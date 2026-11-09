"""Benchmark export: the annual-report benchmark as portable files.

* benchmark/annual_report_benchmark.jsonl – every eval item (task, question, gold, context, metadata)
* benchmark/questions.jsonl              – validated questions with citations + grounding scores
* benchmark/lighteval/*.jsonl + custom_task.py – YourBench `prepare_lighteval` schema, loadable by the
  real LightEval CLI with an Ollama model through LiteLLM (optional interoperability)
* benchmark/BENCHMARK_CARD.md            – what is inside, how it was built
"""

from __future__ import annotations

from pathlib import Path

from arbench.utils import write_jsonl

LIGHTEVAL_TASK_TEMPLATE = '''"""LightEval custom tasks for the ARBench annual-report benchmark (generated file).

Run (requires `pip install lighteval[litellm]` and a running Ollama):

    lighteval endpoint litellm "model_name=ollama/qwen3:4b-instruct,base_url=http://localhost:11434" \\
        "custom|arbench_open|0|0,custom|arbench_mcq|0|0" --custom-tasks "{task_file}"

The open-ended task is graded by an Ollama judge through LiteLLM (change JUDGE_MODEL below).
"""
import re

import numpy as np

from lighteval.metrics.metrics import Metrics
from lighteval.metrics.metrics_sample import JudgeLLM
from lighteval.metrics.utils.metric_utils import CorpusLevelMetricGrouping
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc, SamplingMethod

OPEN_FILE = r"{open_file}"
MCQ_FILE = r"{mcq_file}"
JUDGE_MODEL = "ollama/{judge_model}"
JUDGE_URL = "http://localhost:11434"

QA_PROMPT = """Answer the following question about a company annual report using the excerpt.

<excerpt>
{{chunk}}
</excerpt>

<question>
{{question}}
</question>

Enclose your full answer in <answer> XML tags."""

JUDGE_PROMPT = """Evidence:\\n{{chunk}}\\n\\nQuestion: {{question}}\\nGold answer: {{gold}}\\nModel answer: {{answer}}\\n
Is the model answer correct with respect to the gold answer? Reply with <final_answer>1</final_answer> if correct,
otherwise <final_answer>0</final_answer>."""


def open_prompt(line, task_name: str = ""):
    return Doc(
        task_name=task_name,
        query=QA_PROMPT.format(chunk=line["chunks"][0], question=line["question"]),
        choices=[line["ground_truth_answer"]],
        gold_index=0,
        specific={{"question": line["question"], "chunks": line["chunks"], "document": line["document"]}},
    )


def mcq_prompt(line, task_name: str = ""):
    options = "\\n".join(f"{{chr(65 + i)}}. {{c}}" for i, c in enumerate(line["choices"]))
    query = (
        "Answer the following multiple-choice question by selecting only one letter: A, B, C, or D. "
        f"Do not explain your answer.\\n\\nExcerpt:\\n{{line['chunks'][0]}}\\n\\nQuestion: {{line['question']}}"
        f"\\n\\nChoices:\\n{{options}}\\n\\nAnswer:"
    )
    return Doc(task_name=task_name, query=query, choices=line["choices"], gold_index=line["gold"][0])


def _judge_template(question, answer, gold, **kwargs):
    chunk = kwargs.get("chunks", "")
    return [{{"role": "user", "content": JUDGE_PROMPT.format(chunk=chunk, question=question, gold=gold, answer=answer)}}]


def _judge_parse(response):
    m = re.search(r"<final_answer>\\s*([01])\\s*</final_answer>", str(response))
    return int(m.group(1)) if m else 0


class OllamaJudge(JudgeLLM):
    def __init__(self):
        super().__init__(
            judge_model_name=JUDGE_MODEL,
            template=_judge_template,
            process_judge_response=_judge_parse,
            judge_backend="litellm",
            short_judge_name="arbench_judge",
            url=JUDGE_URL,
        )

    def compute(self, responses, docs, **kwargs):
        questions = [d.specific["question"] for d in docs]
        golds = [d.get_golds()[0] for d in docs]
        preds = [r.final_text[0] for r in responses]
        chunks = [d.specific["chunks"][0] for d in docs]
        scores, _, _ = self.judge.evaluate_answer_batch(
            questions, preds, [None] * len(questions), golds, chunks=chunks
        )
        return [{{"accuracy": s}} for s in scores]


arbench_judge_metric = CorpusLevelMetricGrouping(
    metric_name=["accuracy"],
    higher_is_better={{"accuracy": True}},
    category=SamplingMethod.GENERATIVE,
    sample_level_fn=OllamaJudge(),
    corpus_level_fn={{"accuracy": np.mean}},
)

arbench_open = LightevalTaskConfig(
    name="arbench_open",
    prompt_function=open_prompt,
    hf_repo="json",
    hf_subset="default",
    hf_data_files=OPEN_FILE,
    hf_avail_splits=["train"],
    evaluation_splits=["train"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=512,
    metrics=[arbench_judge_metric],
    stop_sequence=[],
    version=0,
)

arbench_mcq = LightevalTaskConfig(
    name="arbench_mcq",
    prompt_function=mcq_prompt,
    hf_repo="json",
    hf_subset="default",
    hf_data_files=MCQ_FILE,
    hf_avail_splits=["train"],
    evaluation_splits=["train"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=32,
    metrics=[Metrics.gpqa_instruct_metric],
    version=0,
)

TASKS_TABLE = [arbench_open, arbench_mcq]
'''


def _lighteval_row(q: dict, chunk_text: dict[str, str], summary: str) -> dict:
    """YourBench `prepare_lighteval.make_record` schema."""
    is_mcq = q["kind"] == "multi_choice"
    return {
        "question": q["question"],
        "ground_truth_answer": chr(65 + q["gold_index"]) if is_mcq else q["answer"],
        "gold": [q["gold_index"]] if is_mcq else [0],
        "choices": q["choices"] if is_mcq else [q["answer"]],
        "question_category": q["question_type"],
        "kind": q["kind"],
        "estimated_difficulty": q["difficulty"],
        "document_id": q["doc_id"],
        "question_generating_model": q["generator"],
        "chunk_ids": q["chunk_ids"],
        "chunks": [chunk_text[c] for c in q["chunk_ids"] if c in chunk_text],
        "citations": q["citations"],
        "citation_score": q.get("citation_score"),
        "document": summary,  # summary instead of the full report keeps the file small
        "pages": q["pages"],
    }


def export_benchmark(
    bench_dir: Path,
    questions: list[dict],
    items: list[dict],
    chunk_text: dict[str, str],
    summaries: dict[str, str],
    judge_model: str,
    card: str,
) -> dict[str, Path]:
    bench_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    paths["items"] = bench_dir / "annual_report_benchmark.jsonl"
    write_jsonl(paths["items"], items)
    paths["questions"] = bench_dir / "questions.jsonl"
    write_jsonl(paths["questions"], [{k: v for k, v in q.items() if k != "raw_response"} for q in questions])

    le = bench_dir / "lighteval"
    open_rows = [_lighteval_row(q, chunk_text, summaries.get(q["doc_id"], "")) for q in questions if q["kind"] != "multi_choice"]
    mcq_rows = [_lighteval_row(q, chunk_text, summaries.get(q["doc_id"], "")) for q in questions if q["kind"] == "multi_choice"]
    paths["lighteval_open"] = le / "arbench_open.jsonl"
    paths["lighteval_mcq"] = le / "arbench_mcq.jsonl"
    write_jsonl(paths["lighteval_open"], open_rows)
    write_jsonl(paths["lighteval_mcq"], mcq_rows)
    task_file = le / "custom_task.py"
    task_file.write_text(
        LIGHTEVAL_TASK_TEMPLATE.format(
            open_file=paths["lighteval_open"].resolve().as_posix(),
            mcq_file=paths["lighteval_mcq"].resolve().as_posix(),
            judge_model=judge_model,
            task_file=task_file.resolve().as_posix(),
        ),
        encoding="utf-8",
    )
    paths["lighteval_task"] = task_file
    paths["card"] = bench_dir / "BENCHMARK_CARD.md"
    paths["card"].write_text(card, encoding="utf-8")
    return paths
