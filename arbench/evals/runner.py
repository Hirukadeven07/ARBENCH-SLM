"""Evaluation runner + LLM jury (re-engineered from LightEval `pipeline.py` and `JudgeLLMYourBench`).

1. every model answers every eval item (one model loaded at a time -> fits a 16 GB laptop)
2. deterministic metrics are computed per item (metrics.py)
3. free-text answers are graded by an LLM *jury*: by default each answer is judged by the
   OTHER models under test ("cross" strategy) to avoid self-preference bias; verdicts are
   averaged, and inter-judge agreement is recorded.
"""

from __future__ import annotations

import logging
import re
from collections import defaultdict
from pathlib import Path

from arbench import prompts
from arbench.evals.metrics import score_item
from arbench.evals.tasks import TASKS, build_messages, max_tokens_for
from arbench.llm.ollama_client import ChatRequest, OllamaClient
from arbench.utils import extract_json, slug, truncate_tokens, write_jsonl

log = logging.getLogger("arbench.runner")


# ----------------------------------------------------------------------------- answering


def answer_items(client: OllamaClient, model: str, items: list[dict], eval_cfg, concurrency: int) -> list[dict]:
    reqs = [
        ChatRequest(
            messages=build_messages(it),
            temperature=eval_cfg["temperature"],
            max_tokens=max_tokens_for(it, eval_cfg["max_answer_tokens"]),
            tag=it["item_id"],
        )
        for it in items
    ]
    results = client.chat_many(model, reqs, concurrency, desc=f"evaluate ({model})")
    records = []
    for it, res in zip(items, results):
        rec = {
            "item_id": it["item_id"],
            "task": it["task"],
            "model": model,
            "raw_output": res.text,
            "error": res.error,
            "cached": res.cached,
            "stats": {
                "prompt_tokens": res.prompt_tokens,
                "completion_tokens": res.completion_tokens,
                "total_s": round(res.total_s, 3),
                "prompt_eval_s": round(res.prompt_eval_s, 3),
                "eval_s": round(res.eval_s, 3),
                "load_s": round(res.load_s, 3),
                "done_reason": res.done_reason,
            },
        }
        rec["scores"] = score_item(it, res.text if res.ok else "")
        records.append(rec)
    return records


# ----------------------------------------------------------------------------- judging


def judges_for(candidate: str, models: list[str], judge_cfg, fixed_judges: list[str]) -> list[str]:
    strategy = judge_cfg["strategy"]
    if strategy == "self":
        return [candidate]
    if strategy == "fixed":
        return fixed_judges
    others = [m for m in models if m != candidate]
    return others or [candidate]


def _judge_request(item: dict, answer: str, judge_cfg) -> ChatRequest:
    evidence = truncate_tokens(item["evidence"], int(judge_cfg.get("max_evidence_tokens", 1200)))
    return ChatRequest(
        messages=[
            {"role": "system", "content": prompts.JUDGE_SYSTEM},
            {
                "role": "user",
                "content": prompts.JUDGE_USER.format(
                    evidence=evidence,
                    question=item["question"],
                    gold=item["gold"],
                    answer=truncate_tokens(answer, 500),
                ),
            },
        ],
        temperature=judge_cfg["temperature"],
        max_tokens=300,
        format=prompts.JUDGE_SCHEMA,
    )


_VERDICT_RE = re.compile(r"partially[_ ]correct|incorrect|correct", re.IGNORECASE)


def parse_verdict(text: str) -> tuple[str | None, str]:
    try:
        obj = extract_json(text)
        if isinstance(obj, dict):
            v = str(obj.get("verdict", "")).strip().lower().replace(" ", "_")
            if v in prompts.JUDGE_SCORES:
                return v, str(obj.get("reasoning", ""))[:400]
    except ValueError:
        pass
    m = _VERDICT_RE.search(text or "")
    if m:
        return m.group(0).lower().replace(" ", "_"), (text or "")[:400]
    return None, (text or "")[:400]


def run_jury(
    client: OllamaClient,
    items_by_id: dict[str, dict],
    predictions: dict[str, list[dict]],
    models: list[str],
    judge_cfg,
    fixed_judges: list[str],
    concurrency: int,
) -> None:
    """Mutates prediction records: adds `judges` and `scores.judge_score`."""
    work: dict[str, list[tuple[dict, dict]]] = defaultdict(list)  # judge -> [(record, item)]
    for cand, records in predictions.items():
        for rec in records:
            item = items_by_id[rec["item_id"]]
            if not TASKS[item["task"]].judged:
                continue
            rec["judges"] = {}
            s = rec["scores"]
            if rec["error"] or not s.get("prediction"):
                s["judge_score"] = 0.0
                s["judge_note"] = "no answer"
                continue
            if s.get("abstained"):
                s["judge_score"] = 0.0
                s["judge_note"] = "abstained although the answer was available"
                continue
            for j in judges_for(cand, models, judge_cfg, fixed_judges):
                work[j].append((rec, item))

    for judge, pairs in work.items():
        reqs = [_judge_request(item, rec["scores"]["prediction"], judge_cfg) for rec, item in pairs]
        results = client.chat_many(judge, reqs, concurrency, desc=f"judge ({judge})")
        for (rec, _item), res in zip(pairs, results):
            verdict, reasoning = parse_verdict(res.text) if res.ok else (None, res.error or "")
            rec["judges"][judge] = {
                "verdict": verdict,
                "score": prompts.JUDGE_SCORES.get(verdict) if verdict else None,
                "reasoning": reasoning,
            }
        client.unload(judge)

    for records in predictions.values():
        for rec in records:
            if "judges" not in rec or "judge_score" in rec["scores"]:
                continue
            valid = [v["score"] for v in rec["judges"].values() if v["score"] is not None]
            rec["scores"]["judge_score"] = sum(valid) / len(valid) if valid else None
            rec["scores"]["judge_correct"] = (
                float(sum(1 for v in valid if v == 1.0) > len(valid) / 2) if valid else None
            )
            rec["scores"]["judge_agreement"] = float(len(set(valid)) == 1) if len(valid) > 1 else None


# ----------------------------------------------------------------------------- orchestration


def run_evaluation(
    client: OllamaClient,
    items: list[dict],
    models: list[str],
    cfg,
    out_dir: Path,
    fixed_judges: list[str],
) -> dict[str, list[dict]]:
    eval_cfg, judge_cfg = cfg["evaluation"], cfg["judge"]
    concurrency = int(cfg["ollama"]["concurrency"])
    predictions: dict[str, list[dict]] = {}
    for model in models:
        log.info("Evaluating %s on %d items …", model, len(items))
        predictions[model] = answer_items(client, model, items, eval_cfg, concurrency)
        errors = sum(1 for r in predictions[model] if r["error"])
        if errors:
            log.warning("%s: %d / %d calls failed (see predictions file)", model, errors, len(items))
        if len(models) > 1:
            client.unload(model)

    items_by_id = {it["item_id"]: it for it in items}
    if judge_cfg["enabled"]:
        log.info("Judging free-text answers (strategy=%s) …", judge_cfg["strategy"])
        run_jury(client, items_by_id, predictions, models, judge_cfg, fixed_judges, concurrency)

    pred_dir = out_dir / "predictions"
    for model, records in predictions.items():
        rows = []
        for rec in records:
            it = items_by_id[rec["item_id"]]
            rows.append(
                {
                    **rec,
                    "question": it["question"],
                    "gold": it["gold"],
                    "choices": it["choices"],
                    "gold_index": it["gold_index"],
                    "kind": it["kind"],
                    "question_type": it["question_type"],
                    "difficulty": it["difficulty"],
                    "generator": it["generator"],
                    "pages": it["pages"],
                }
            )
        write_jsonl(pred_dir / f"{slug(model)}.jsonl", rows)
    return predictions
