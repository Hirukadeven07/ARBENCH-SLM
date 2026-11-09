"""Benchmark tasks (LightEval `LightevalTaskConfig` + prompt functions, re-engineered).

Each filtered question becomes one or more *eval items* (LightEval `Doc`s):

| task          | context given to the model                     | primary metric          |
|---------------|------------------------------------------------|-------------------------|
| grounded_qa   | the gold evidence chunk(s)                     | LLM-jury judge score    |
| rag_qa        | top-k BM25 chunks retrieved from the report    | LLM-jury judge score    |
| multi_choice  | the gold evidence chunk                        | accuracy                |
| unanswerable  | an unrelated chunk that lacks the answer       | abstention accuracy     |
| closed_book   | nothing (optional)                             | LLM-jury judge score    |
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass

from arbench import prompts
from arbench.pipeline.chunk import Chunk
from arbench.retrieval import BM25, content_overlap
from arbench.utils import core_numbers, extract_numbers, numbers_match, stable_hash

log = logging.getLogger("arbench.tasks")


@dataclass(frozen=True)
class TaskSpec:
    name: str
    description: str
    primary_metric: str
    judged: bool


TASKS: dict[str, TaskSpec] = {
    "grounded_qa": TaskSpec(
        "grounded_qa", "Open-book QA with the gold evidence chunk(s) (reading comprehension)", "judge_score", True
    ),
    "rag_qa": TaskSpec(
        "rag_qa", "QA over BM25-retrieved chunks from the whole report (realistic RAG)", "judge_score", True
    ),
    "multi_choice": TaskSpec("multi_choice", "4-option multiple choice with evidence", "accuracy", False),
    "unanswerable": TaskSpec(
        "unanswerable", "Answer is NOT in the given excerpt – model must abstain (hallucination test)", "abstention", False
    ),
    "closed_book": TaskSpec("closed_book", "No context – measures prior knowledge / guessing", "judge_score", True),
}


def format_excerpts(chunks: list[Chunk]) -> str:
    return "\n\n".join(
        f"[Excerpt {i + 1} | {c.page_label}{' | ' + c.section if c.section else ''}]\n{c.text}"
        for i, c in enumerate(chunks)
    )


def _item(task: str, q: dict, context_chunks: list[Chunk], evidence_chunks: list[Chunk], **extra) -> dict:
    return {
        "item_id": stable_hash(task, q["qid"], n=14),
        "task": task,
        "qid": q["qid"],
        "kind": q["kind"],
        "question": q["question"],
        "gold": q["answer"],
        "choices": q.get("choices") or [],
        "gold_index": q.get("gold_index"),
        "question_type": q["question_type"],
        "difficulty": q["difficulty"],
        "generator": q["generator"],
        "doc_id": q["doc_id"],
        "pages": q["pages"],
        "section": q.get("section", ""),
        "context_chunk_ids": [c.chunk_id for c in context_chunks],
        "context": format_excerpts(context_chunks) if context_chunks else "",
        "evidence": format_excerpts(evidence_chunks),
        **extra,
    }


def _answer_absent(gold: str, chunk: Chunk) -> bool:
    """Conservative check that a distractor chunk does not contain the gold answer."""
    if content_overlap(gold, chunk.text) > 0.35:
        return False
    nums = [v for v in core_numbers(gold) if not (v.is_integer() and v <= 10)]
    if nums:
        src = extract_numbers(chunk.text)
        if any(numbers_match(v, s) for v in nums for s in src):
            return False
    return True


def build_items(questions: list[dict], chunks: list[Chunk], eval_cfg, seed: int) -> list[dict]:
    tasks_on = {name for name, enabled in eval_cfg["tasks"].items() if enabled}
    by_id = {c.chunk_id: c for c in chunks}
    bm25 = BM25([c.text for c in chunks])
    k = int(eval_cfg.get("rag_top_k", 4))
    rng = random.Random(seed)
    items: list[dict] = []

    single = [q for q in questions if q["kind"] == "single_hop"]
    n_unans = round(len(single) * float(eval_cfg.get("unanswerable_fraction", 0.35)))
    unans_ids = {q["qid"] for q in rng.sample(single, min(n_unans, len(single)))} if single else set()

    for q in questions:
        gold_chunks = [by_id[c] for c in q["chunk_ids"] if c in by_id]
        if not gold_chunks:
            continue
        if q["kind"] == "multi_choice":
            if "multi_choice" in tasks_on:
                items.append(_item("multi_choice", q, gold_chunks, gold_chunks))
            continue

        if "grounded_qa" in tasks_on:
            items.append(_item("grounded_qa", q, gold_chunks, gold_chunks))

        if "rag_qa" in tasks_on:
            top = bm25.top_k(q["question"], k)
            retrieved = [chunks[i] for i in top]
            gold_set = set(q["chunk_ids"])
            hit = len(gold_set & {c.chunk_id for c in retrieved}) / len(gold_set)
            items.append(_item("rag_qa", q, retrieved, gold_chunks, retrieval_recall=round(hit, 3)))

        if "closed_book" in tasks_on:
            items.append(_item("closed_book", q, [], gold_chunks))

        if "unanswerable" in tasks_on and q["qid"] in unans_ids:
            scores = bm25.scores(q["question"])
            ranked = sorted(range(len(chunks)), key=lambda i: -scores[i])
            gold_ids = set(q["chunk_ids"])
            distractor = None
            # skip the top-3 most relevant chunks (annual reports repeat key facts), then take the most
            # topically similar chunk that provably lacks the answer -> a tempting "hard negative".
            for rank, i in enumerate(ranked):
                c = chunks[i]
                if rank < 3 or c.chunk_id in gold_ids:
                    continue
                if any(c.doc_id == g.doc_id and abs(c.index - g.index) <= 1 for g in gold_chunks):
                    continue
                if _answer_absent(q["answer"], c):
                    distractor = c
                    break
            if distractor is not None:
                items.append(_item("unanswerable", q, [distractor], gold_chunks, expected=prompts.INSUFFICIENT))
    return items


# ----------------------------------------------------------------------------- prompt functions


def build_messages(item: dict) -> list[dict[str, str]]:
    """LightEval-style prompt function: item -> chat messages."""
    task = item["task"]
    if task == "multi_choice":
        options = "\n".join(f"{chr(65 + i)}. {c}" for i, c in enumerate(item["choices"]))
        return [
            {"role": "system", "content": prompts.MCQ_EVAL_SYSTEM},
            {
                "role": "user",
                "content": prompts.MCQ_EVAL_USER.format(
                    excerpts=item["context"], question=item["question"], options=options
                ),
            },
        ]
    if task == "closed_book":
        return [
            {"role": "system", "content": prompts.CLOSED_BOOK_SYSTEM},
            {"role": "user", "content": prompts.CLOSED_BOOK_USER.format(question=item["question"])},
        ]
    return [
        {"role": "system", "content": prompts.QA_SYSTEM},
        {"role": "user", "content": prompts.QA_USER.format(excerpts=item["context"], question=item["question"])},
    ]


def max_tokens_for(item: dict, default: int) -> int:
    return 64 if item["task"] == "multi_choice" else default
