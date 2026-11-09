"""Stage 4 – question generation (YourBench single-hop, multi-hop and multi-choice).

* Chunks are sampled with stratification over the whole report (so questions cover
  the strategic report, financial statements, governance and notes alike) and
  weighted towards information-dense chunks.
* Generation is rotated across several models (YourBench supports multiple
  generators); the generator of each question is recorded so the reports can
  check for self-preference bias ("does a model do better on its own questions?").
* Ollama enforces a JSON schema -> small models produce parseable output.
"""

from __future__ import annotations

import hashlib
import logging
import random
import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from arbench import prompts
from arbench.llm.ollama_client import ChatRequest, OllamaClient
from arbench.pipeline.chunk import Chunk
from arbench.retrieval import BM25
from arbench.utils import extract_json, stable_hash

log = logging.getLogger("arbench.generate")


@dataclass
class GenJob:
    kind: str  # single_hop | multi_hop | multi_choice
    chunks: list[Chunk]
    generator: str
    n_questions: int
    request: ChatRequest | None = None
    raw: str = ""
    error: str | None = None
    questions: list[dict] = field(default_factory=list)


# ----------------------------------------------------------------------------- sampling


def _info_score(c: Chunk) -> float:
    return c.quality + 0.35 * min(1.0, c.n_numbers / 15)


def stratified_sample(chunks: list[Chunk], k: int, seed: int, offset: float = 0.0, exclude: set[str] | None = None):
    """Pick k chunks spread over the document, best-informed chunk per stratum."""
    pool = [c for c in chunks if not exclude or c.chunk_id not in exclude] or list(chunks)
    if k >= len(pool):
        return list(pool)
    rng = random.Random(seed)
    picks: list[Chunk] = []
    n = len(pool)
    for i in range(k):
        lo = int((i + offset) * n / k) % n
        hi = max(lo + 1, int((i + 1 + offset) * n / k))
        stratum = pool[lo:min(hi, n)] or pool[lo : lo + 1]
        best = max(stratum, key=lambda c: (_info_score(c), rng.random()))
        if best not in picks:
            picks.append(best)
    return picks


def sample_multihop_groups(chunks: list[Chunk], n_groups: int, size: int, seed: int) -> list[list[Chunk]]:
    """Pair an anchor chunk with topically related but non-adjacent chunks (BM25)."""
    if len(chunks) < size or n_groups <= 0:
        return []
    bm25 = BM25([c.text for c in chunks])
    anchors = stratified_sample(chunks, n_groups, seed + 7, offset=0.5)
    groups, seen = [], set()
    for anchor in anchors:
        a_idx = chunks.index(anchor)
        scores = bm25.scores(anchor.text[:3000])
        ranked = sorted(range(len(chunks)), key=lambda i: -scores[i])
        group = [anchor]
        for i in ranked:
            c = chunks[i]
            if c is anchor or any(abs(c.index - g.index) < 3 and c.doc_id == g.doc_id for g in group):
                continue
            if i == a_idx:
                continue
            group.append(c)
            if len(group) == size:
                break
        key = tuple(sorted(c.chunk_id for c in group))
        if len(group) == size and key not in seen:
            seen.add(key)
            groups.append(group)
    return groups


# ----------------------------------------------------------------------------- prompts


def _single_request(kind: str, chunk: Chunk, title: str, summary: str, n: int, temperature: float) -> ChatRequest:
    system = (prompts.MCQ_SYSTEM if kind == "multi_choice" else prompts.SINGLE_HOP_SYSTEM).format(n=n)
    user = prompts.SINGLE_HOP_USER.format(
        title=title, summary=summary, chunk=chunk.text, page_label=chunk.page_label
    )
    return ChatRequest(
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        temperature=temperature,
        max_tokens=600 * n + 200,
        format=prompts.MCQ_SCHEMA if kind == "multi_choice" else prompts.OPEN_QA_SCHEMA,
    )


def _multi_request(group: list[Chunk], title: str, summary: str, n: int, temperature: float) -> ChatRequest:
    chunk_xml = "\n\n".join(
        f'<text_chunk id="{i + 1}" source="{c.page_label}">\n{c.text}\n</text_chunk>' for i, c in enumerate(group)
    )
    return ChatRequest(
        messages=[
            {"role": "system", "content": prompts.MULTI_HOP_SYSTEM.format(n=n)},
            {"role": "user", "content": prompts.MULTI_HOP_USER.format(title=title, summary=summary, chunks=chunk_xml)},
        ],
        temperature=temperature,
        max_tokens=600 * n + 200,
        format=prompts.OPEN_QA_SCHEMA,
    )


# ----------------------------------------------------------------------------- parsing

_CHOICE_PREFIX = re.compile(r"^\s*(?:\(?[A-Da-d]\)|[A-Da-d][.:)]\s)\s*")


def shuffle_mcq(choices: list[str], answer_letter: str) -> tuple[list[str], int]:
    """Deterministic shuffle (YourBench `shuffle_mcq`) – removes the generators' position bias."""
    raw = [_CHOICE_PREFIX.sub("", c).strip() for c in choices]
    gold = ord(answer_letter) - 65
    order = list(range(len(raw)))
    seed = int(hashlib.sha256(repr((raw, gold)).encode()).hexdigest(), 16)
    random.Random(seed).shuffle(order)
    return [raw[i] for i in order], order.index(gold)


def _coerce_questions(obj: Any) -> list[dict]:
    if isinstance(obj, dict):
        for key in ("questions", "items", "qa_pairs", "data"):
            if isinstance(obj.get(key), list):
                return [q for q in obj[key] if isinstance(q, dict)]
        if "question" in obj:
            return [obj]
    if isinstance(obj, list):
        return [q for q in obj if isinstance(q, dict)]
    return []


def parse_job(job: GenJob, doc_titles: dict[str, str]) -> list[dict]:
    try:
        items = _coerce_questions(extract_json(job.raw))
    except ValueError:
        job.error = "unparseable JSON"
        return []
    out = []
    for q in items:
        question = str(q.get("question", "")).strip()
        answer = str(q.get("answer", "")).strip()
        if len(question) < 12 or not answer:
            continue
        cits = q.get("citations") or []
        if isinstance(cits, str):
            cits = [cits]
        cits = [str(c).strip().strip('"“”') for c in cits if str(c).strip()]
        try:
            diff = int(q.get("estimated_difficulty", 5))
        except (TypeError, ValueError):
            diff = 5
        qtype = str(q.get("question_type", "factual")).strip().lower() or "factual"
        rec = {
            "qid": "",
            "kind": job.kind,
            "question": question,
            "answer": answer,
            "choices": [],
            "gold_index": None,
            "question_type": qtype,
            "difficulty": max(1, min(10, diff)),
            "citations": cits,
            "chunk_ids": [c.chunk_id for c in job.chunks],
            "doc_id": job.chunks[0].doc_id,
            "doc_title": doc_titles.get(job.chunks[0].doc_id, ""),
            "pages": sorted({p for c in job.chunks for p in c.pages}),
            "section": job.chunks[0].section,
            "generator": job.generator,
        }
        if job.kind == "multi_choice":
            choices = q.get("choices") or []
            letter = answer.strip().upper()[:1]
            if (
                not isinstance(choices, list)
                or len(choices) != 4
                or letter not in "ABCD"
                or not letter
                or not all(isinstance(c, str) and c.strip() for c in choices)
            ):
                continue
            shuffled, gold = shuffle_mcq([str(c) for c in choices], letter)
            rec["choices"] = shuffled
            rec["gold_index"] = gold
            rec["answer"] = shuffled[gold]
        rec["qid"] = stable_hash(rec["kind"], rec["question"], rec["chunk_ids"], n=12)
        out.append(rec)
    return out


# ----------------------------------------------------------------------------- stage


def run_generation(
    client: OllamaClient,
    chunks: list[Chunk],
    summaries: dict[str, str],
    titles: dict[str, str],
    gen_cfg,
    generators: list[str],
    seed: int,
    concurrency: int = 1,
) -> tuple[list[dict], dict]:
    sh, mh, mc = gen_cfg["single_hop"], gen_cfg["multi_hop"], gen_cfg["multi_choice"]
    temp = gen_cfg["temperature"]

    single_chunks = stratified_sample(chunks, sh["num_chunks"], seed) if sh["num_chunks"] > 0 else []
    used = {c.chunk_id for c in single_chunks}
    mcq_chunks = (
        stratified_sample(chunks, mc["num_chunks"], seed + 1, offset=0.5, exclude=used) if mc["num_chunks"] > 0 else []
    )
    groups = sample_multihop_groups(chunks, mh["num_groups"], mh.get("chunks_per_group", 2), seed)

    jobs: list[GenJob] = []
    for c in single_chunks:
        jobs.append(GenJob("single_hop", [c], "", sh["questions_per_chunk"]))
    for c in mcq_chunks:
        jobs.append(GenJob("multi_choice", [c], "", mc["questions_per_chunk"]))
    for g in groups:
        jobs.append(GenJob("multi_hop", g, "", mh["questions_per_group"]))

    # rotate generators across jobs (per kind, so every model writes every kind)
    per_kind: dict[str, int] = defaultdict(int)
    for job in jobs:
        job.generator = generators[per_kind[job.kind] % len(generators)]
        per_kind[job.kind] += 1
        did = job.chunks[0].doc_id
        title, summary = titles.get(did, ""), summaries.get(did, "")
        if job.kind == "multi_hop":
            job.request = _multi_request(job.chunks, title, summary, job.n_questions, temp)
        else:
            job.request = _single_request(job.kind, job.chunks[0], title, summary, job.n_questions, temp)

    log.info(
        "Generating questions: %d single-hop chunks, %d multi-hop groups, %d MCQ chunks, generators=%s",
        len(single_chunks),
        len(groups),
        len(mcq_chunks),
        ", ".join(generators),
    )
    by_model: dict[str, list[GenJob]] = defaultdict(list)
    for job in jobs:
        by_model[job.generator].append(job)
    for model, mjobs in by_model.items():
        results = client.chat_many(model, [j.request for j in mjobs], concurrency, desc=f"generate ({model})")
        for job, res in zip(mjobs, results):
            job.raw, job.error = res.text, res.error
        if len(by_model) > 1:
            client.unload(model)

    questions: list[dict] = []
    stats = {"jobs": len(jobs), "failed_jobs": 0, "unparseable": 0, "by_generator": defaultdict(int)}
    for job in jobs:
        if job.error and not job.raw:
            stats["failed_jobs"] += 1
            continue
        qs = parse_job(job, titles)
        if job.error == "unparseable JSON":
            stats["unparseable"] += 1
        for q in qs:
            q["raw_response"] = job.raw
        questions.extend(qs)
        stats["by_generator"][job.generator] += len(qs)
    stats["by_generator"] = dict(stats["by_generator"])
    stats["raw_questions"] = len(questions)
    log.info("Generated %d raw questions (%d unparseable responses).", len(questions), stats["unparseable"])
    return questions, stats
