"""Corpus-level aggregation (LightEval `metrics_corpus` + `stderr.py`, re-engineered).

* mean + standard error (LightEval `mean_stderr`) + bootstrap 95 % CI for every metric
* paired bootstrap significance test for every model pair on every task
* breakdowns by question type, difficulty, hop kind, and generator (self vs. other)
* efficiency statistics from Ollama timings
* weighted composite "ARBench score"
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Iterable

import numpy as np

from arbench.evals.tasks import TASKS

QA_METRICS = ["judge_score", "judge_correct", "token_f1", "rouge_l", "exact_match", "numeric_match",
              "number_hallucination", "abstained", "format_ok", "judge_agreement"]
TASK_METRICS = {
    "grounded_qa": QA_METRICS + ["over_refusal"],
    "rag_qa": QA_METRICS + ["retrieval_recall"],
    "closed_book": QA_METRICS,
    "multi_choice": ["accuracy", "parsed"],
    "unanswerable": ["abstention", "number_hallucination", "format_ok"],
}


# ----------------------------------------------------------------------------- statistics


def mean_stderr(values: list[float]) -> float:
    n = len(values)
    if n < 2:
        return 0.0
    mu = sum(values) / n
    return math.sqrt(sum((x - mu) ** 2 for x in values) / (n - 1)) / math.sqrt(n)


def bootstrap_ci(values: list[float], n_boot: int, seed: int = 0) -> tuple[float, float]:
    if len(values) < 2:
        v = values[0] if values else float("nan")
        return v, v
    rng = np.random.default_rng(seed)
    arr = np.asarray(values, dtype=float)
    means = rng.choice(arr, size=(n_boot, len(arr)), replace=True).mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def paired_bootstrap(a: list[float], b: list[float], n_boot: int, seed: int = 0) -> dict:
    """Two-sided paired bootstrap test on the mean difference a-b."""
    diff = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    if len(diff) < 2:
        return {"diff": float(diff.mean()) if len(diff) else 0.0, "p_value": 1.0, "ci95": [0.0, 0.0], "n": len(diff)}
    rng = np.random.default_rng(seed)
    boots = rng.choice(diff, size=(n_boot, len(diff)), replace=True).mean(axis=1)
    obs = float(diff.mean())
    # p-value: probability the bootstrap distribution (centred) is as extreme as observed
    centred = boots - boots.mean()
    p = float((np.abs(centred) >= abs(obs)).mean())
    return {
        "diff": obs,
        "p_value": max(p, 1.0 / n_boot),
        "ci95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
        "n": int(len(diff)),
    }


def summarize(values: Iterable[float | None], n_boot: int, seed: int = 0) -> dict | None:
    vals = [float(v) for v in values if v is not None and not (isinstance(v, float) and math.isnan(v))]
    if not vals:
        return None
    lo, hi = bootstrap_ci(vals, n_boot, seed)
    return {"mean": sum(vals) / len(vals), "stderr": mean_stderr(vals), "ci95": [lo, hi], "n": len(vals)}


# ----------------------------------------------------------------------------- primary score per item


def primary_score(task: str, scores: dict, judge_enabled: bool) -> float | None:
    spec = TASKS[task]
    if spec.primary_metric == "judge_score":
        if judge_enabled and scores.get("judge_score") is not None:
            return scores["judge_score"]
        return scores.get("token_f1")
    return scores.get(spec.primary_metric)


def _bucket_difficulty(d: int) -> str:
    return "easy (1-3)" if d <= 3 else "medium (4-7)" if d <= 7 else "hard (8-10)"


# ----------------------------------------------------------------------------- aggregation


def aggregate(predictions: dict[str, list[dict]], items: list[dict], cfg) -> dict:
    items_by_id = {it["item_id"]: it for it in items}
    judge_enabled = bool(cfg["judge"]["enabled"])
    n_boot = int(cfg["scoring"]["bootstrap_samples"])
    weights = dict(cfg["scoring"]["weights"])
    results: dict = {"models": {}, "pairwise": {}, "meta": {}}

    for model, records in predictions.items():
        by_task: dict[str, list[dict]] = defaultdict(list)
        for r in records:
            by_task[r["task"]].append(r)
        tasks_out = {}
        for task, recs in by_task.items():
            metrics = {}
            for m in TASK_METRICS.get(task, []):
                if m == "over_refusal":
                    vals = [float(r["scores"].get("abstained", False)) for r in recs]
                elif m == "retrieval_recall":
                    vals = [items_by_id[r["item_id"]].get("retrieval_recall") for r in recs]
                else:
                    vals = []
                    for r in recs:
                        v = r["scores"].get(m)
                        vals.append(float(v) if isinstance(v, (bool, int, float)) and v is not None else None)
                s = summarize(vals, n_boot)
                if s:
                    metrics[m] = s
            prim = summarize([primary_score(task, r["scores"], judge_enabled) for r in recs], n_boot)
            tasks_out[task] = {
                "n": len(recs),
                "errors": sum(1 for r in recs if r["error"]),
                "primary_metric": TASKS[task].primary_metric
                if (judge_enabled or TASKS[task].primary_metric != "judge_score")
                else "token_f1",
                "primary": prim,
                "metrics": metrics,
            }

        # ------------------------------------------------ breakdowns over answerable items
        breakdowns: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
        for r in records:
            if r["task"] in ("unanswerable", "closed_book"):
                continue
            it = items_by_id[r["item_id"]]
            ps = primary_score(r["task"], r["scores"], judge_enabled)
            if ps is None:
                continue
            breakdowns["question_type"][it["question_type"]].append(ps)
            breakdowns["difficulty"][_bucket_difficulty(it["difficulty"])].append(ps)
            breakdowns["kind"][it["kind"]].append(ps)
            breakdowns["generator"]["own questions" if it["generator"] == model else "other models' questions"].append(ps)
            pg = min(it["pages"]) if it["pages"] else 0
            breakdowns["report_section"][it.get("section") or f"page {pg}"].append(ps)
        bd_out = {
            dim: {k: {"mean": sum(v) / len(v), "n": len(v)} for k, v in sorted(groups.items())}
            for dim, groups in breakdowns.items()
            if dim != "report_section"
        }
        # keep the 12 largest sections only (reports have many headings)
        sec = breakdowns.get("report_section", {})
        bd_out["report_section"] = {
            k: {"mean": sum(v) / len(v), "n": len(v)}
            for k, v in sorted(sec.items(), key=lambda kv: -len(kv[1]))[:12]
        }

        # ------------------------------------------------ efficiency
        ok = [r for r in records if not r["error"]]
        comp = sum(r["stats"]["completion_tokens"] for r in ok)
        eval_s = sum(r["stats"]["eval_s"] for r in ok)
        prompt_tok = sum(r["stats"]["prompt_tokens"] for r in ok)
        prompt_s = sum(r["stats"]["prompt_eval_s"] for r in ok)
        lat = [r["stats"]["total_s"] for r in ok if r["stats"]["total_s"] > 0]
        efficiency = {
            "calls": len(records),
            "failed_calls": len(records) - len(ok),
            "gen_tokens_per_s": comp / eval_s if eval_s else 0.0,
            "prompt_tokens_per_s": prompt_tok / prompt_s if prompt_s else 0.0,
            "mean_latency_s": float(np.mean(lat)) if lat else 0.0,
            "p95_latency_s": float(np.percentile(lat, 95)) if lat else 0.0,
            "mean_output_tokens": comp / len(ok) if ok else 0.0,
            "total_compute_s": float(sum(lat)),
            "truncated_outputs": sum(1 for r in ok if r["stats"].get("done_reason") == "length"),
        }
        results["models"][model] = {"tasks": tasks_out, "breakdowns": bd_out, "efficiency": efficiency}

    # ------------------------------------------------ composite score
    max_tps = max((m["efficiency"]["gen_tokens_per_s"] for m in results["models"].values()), default=0) or 1.0
    for model, res in results["models"].items():
        parts, total_w = {}, 0.0
        for task, w in weights.items():
            if w <= 0:
                continue
            if task == "efficiency":
                val = res["efficiency"]["gen_tokens_per_s"] / max_tps
            else:
                t = res["tasks"].get(task)
                if not t or not t["primary"]:
                    continue
                val = t["primary"]["mean"]
            parts[task] = val
            total_w += w
        res["composite"] = sum(weights[t] * v for t, v in parts.items()) / total_w if total_w else 0.0
        res["composite_parts"] = parts

    # ------------------------------------------------ pairwise significance
    models = list(predictions)
    for task in sorted({r["task"] for recs in predictions.values() for r in recs}):
        per_model = {
            m: {r["item_id"]: primary_score(task, r["scores"], judge_enabled) for r in predictions[m] if r["task"] == task}
            for m in models
        }
        for i, a in enumerate(models):
            for b in models[i + 1 :]:
                shared = [k for k in per_model[a] if per_model[a][k] is not None and per_model[b].get(k) is not None]
                if not shared:
                    continue
                test = paired_bootstrap([per_model[a][k] for k in shared], [per_model[b][k] for k in shared], n_boot)
                results["pairwise"].setdefault(task, []).append({"a": a, "b": b, **test})

    results["ranking"] = sorted(models, key=lambda m: -results["models"][m]["composite"])
    results["meta"]["weights"] = weights
    results["meta"]["judge_enabled"] = judge_enabled
    return results
