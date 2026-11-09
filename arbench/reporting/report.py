"""Report generation: one Markdown report per SLM, a comparison report, CSV tables and an HTML dashboard.

All narrative text is *data-driven templates* (no LLM writes the conclusions), so every
sentence is traceable to a number in results.json.
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path

from arbench.evals.tasks import TASKS
from arbench.reporting import charts
from arbench.reporting.html import markdown_to_html_page
from arbench.utils import slug

log = logging.getLogger("arbench.report")

TASK_ORDER = ["grounded_qa", "rag_qa", "multi_choice", "unanswerable", "closed_book"]
TASK_LABEL = {
    "grounded_qa": "Grounded QA",
    "rag_qa": "RAG QA",
    "multi_choice": "Multiple choice",
    "unanswerable": "Unanswerable (abstain)",
    "closed_book": "Closed book",
}
METRIC_LABEL = {
    "judge_score": "Judge score (jury, partial credit)",
    "judge_correct": "Judge accuracy (majority 'correct')",
    "token_f1": "Token F1",
    "rouge_l": "ROUGE-L",
    "exact_match": "Exact match",
    "numeric_match": "Numeric match (gold figures reproduced)",
    "number_hallucination": "Number hallucination rate (lower is better)",
    "abstained": "Abstention rate",
    "over_refusal": "Over-refusal rate (lower is better)",
    "format_ok": "Format compliance (<answer> tags)",
    "judge_agreement": "Judge agreement",
    "retrieval_recall": "Retrieval recall@k (gold chunk retrieved)",
    "accuracy": "Accuracy",
    "parsed": "Valid letter produced",
    "abstention": "Correct abstention rate",
}
LOWER_IS_BETTER = {"number_hallucination", "over_refusal"}


def pct(v: float | None, digits: int = 1) -> str:
    return "–" if v is None else f"{v * 100:.{digits}f}%"


def ci(s: dict | None) -> str:
    return "–" if not s else f"[{s['ci95'][0] * 100:.1f}, {s['ci95'][1] * 100:.1f}]"


def _tasks_present(results: dict) -> list[str]:
    present = {t for m in results["models"].values() for t in m["tasks"]}
    return [t for t in TASK_ORDER if t in present]


def _mean(s: dict | None) -> float | None:
    return s["mean"] if s else None


# ============================================================================ per-model


def _strengths(res: dict) -> tuple[list[str], list[str]]:
    qt = {k: v for k, v in res["breakdowns"].get("question_type", {}).items() if v["n"] >= 3}
    ranked = sorted(qt.items(), key=lambda kv: -kv[1]["mean"])
    best = [f"{k} questions ({pct(v['mean'], 0)}, n={v['n']})" for k, v in ranked[:2]]
    worst = [f"{k} questions ({pct(v['mean'], 0)}, n={v['n']})" for k, v in ranked[-2:][::-1]] if len(ranked) > 2 else []
    return best, worst


def model_report(model: str, ctx: dict, chart_path: Path | None) -> str:
    results, labels = ctx["results"], ctx["labels"]
    res = results["models"][model]
    label = labels[model]
    rank = results["ranking"].index(model) + 1
    n_models = len(results["ranking"])
    info = ctx["model_info"].get(model, {})
    eff = res["efficiency"]
    L: list[str] = []
    L.append(f"# {label} — Annual-Report Benchmark Report\n")
    L.append(
        f"**Ollama tag:** `{model}` · **Family:** {info.get('family', '?')} · **Parameters:** "
        f"{info.get('parameter_size', '?')} · **Quantisation:** {info.get('quantization_level', '?')} · "
        f"**Size:** {info.get('size_gb', '?')} GB · **Digest:** `{info.get('digest', '?')}`\n"
    )
    L.append(
        f"**Benchmark:** {ctx['bench_title']} · {ctx['n_questions']} questions → {ctx['n_items']} eval items · "
        f"run `{ctx['run_name']}` · profile `{ctx['profile']}`\n"
    )
    L.append("## Headline\n")
    L.append(
        f"- **ARBench composite score: {res['composite'] * 100:.1f} / 100** — rank **{rank} of {n_models}**."
    )
    for t in _tasks_present(results):
        tr = res["tasks"].get(t)
        if tr and tr["primary"]:
            L.append(f"- {TASK_LABEL[t]}: **{pct(tr['primary']['mean'])}** (95% CI {ci(tr['primary'])}, n={tr['n']})")
    L.append(
        f"- Speed: **{eff['gen_tokens_per_s']:.1f} tokens/s** generation, {eff['prompt_tokens_per_s']:.0f} tokens/s "
        f"prompt processing, mean latency {eff['mean_latency_s']:.1f}s per answer."
    )
    best, worst = _strengths(res)
    if best:
        L.append(f"- Strongest on: {'; '.join(best)}.")
    if worst:
        L.append(f"- Weakest on: {'; '.join(worst)}.")
    L.append("")
    if chart_path:
        L.append(f"![Task scores]({chart_path.as_posix()})\n")

    # ---------------- task table
    L.append("## Results by task\n")
    L.append("| Task | What it tests | n | Primary metric | Score | 95% CI | ± stderr |")
    L.append("|---|---|---:|---|---:|---|---:|")
    for t in _tasks_present(results):
        tr = res["tasks"].get(t)
        if not tr:
            continue
        p = tr["primary"]
        L.append(
            f"| {TASK_LABEL[t]} | {TASKS[t].description} | {tr['n']} | {METRIC_LABEL.get(tr['primary_metric'], tr['primary_metric'])} "
            f"| **{pct(_mean(p))}** | {ci(p)} | {pct(p['stderr'] if p else None)} |"
        )
    L.append("")

    # ---------------- detailed metrics
    L.append("## Detailed metrics\n")
    for t in _tasks_present(results):
        tr = res["tasks"].get(t)
        if not tr or not tr["metrics"]:
            continue
        L.append(f"### {TASK_LABEL[t]}\n")
        L.append("| Metric | Value | 95% CI | n |")
        L.append("|---|---:|---|---:|")
        for m, s in tr["metrics"].items():
            arrow = " ↓" if m in LOWER_IS_BETTER else ""
            L.append(f"| {METRIC_LABEL.get(m, m)}{arrow} | {pct(s['mean'])} | {ci(s)} | {s['n']} |")
        if tr["errors"]:
            L.append(f"\n> ⚠ {tr['errors']} call(s) failed for this task and were scored as wrong.")
        L.append("")

    # ---------------- hallucination section
    L.append("## Hallucination & abstention behaviour\n")
    un = res["tasks"].get("unanswerable")
    gq = res["tasks"].get("grounded_qa")
    if un and un["primary"]:
        L.append(
            f"- When the excerpt did **not** contain the answer, the model correctly abstained "
            f"**{pct(un['primary']['mean'])}** of the time (n={un['n']}); the rest are confident answers that are "
            "not supported by the evidence (hallucinations)."
        )
    if gq and gq["metrics"].get("over_refusal"):
        L.append(
            f"- When the answer **was** available, it wrongly refused {pct(gq['metrics']['over_refusal']['mean'])} "
            "of grounded questions (over-refusal)."
        )
    for t in ("grounded_qa", "rag_qa"):
        tr = res["tasks"].get(t)
        if tr and tr["metrics"].get("number_hallucination"):
            L.append(
                f"- {TASK_LABEL[t]}: {pct(tr['metrics']['number_hallucination']['mean'])} of the figures it stated "
                "appear nowhere in the excerpts it was given (number hallucination)."
            )
    L.append("")

    # ---------------- breakdowns
    L.append("## Breakdowns (answerable items: grounded QA, RAG QA, MCQ)\n")
    names = {
        "question_type": "Question type",
        "difficulty": "Generator-estimated difficulty",
        "kind": "Question kind (single-hop / multi-hop / multiple choice)",
        "generator": "Question author (self-preference check)",
        "report_section": "Report section (largest 12)",
    }
    for dim, title in names.items():
        groups = res["breakdowns"].get(dim)
        if not groups:
            continue
        L.append(f"### {title}\n")
        L.append("| Group | Score | n |")
        L.append("|---|---:|---:|")
        for k, v in sorted(groups.items(), key=lambda kv: -kv[1]["mean"]):
            L.append(f"| {k} | {pct(v['mean'])} | {v['n']} |")
        if dim == "generator" and len(groups) == 2:
            own = groups.get("own questions", {}).get("mean")
            other = groups.get("other models' questions", {}).get("mean")
            if own is not None and other is not None:
                gap = (own - other) * 100
                L.append(
                    f"\n> Self-preference gap: {gap:+.1f} pts on questions this model wrote itself "
                    f"({'notable' if abs(gap) >= 10 else 'small'} — YourBench-style generator bias check)."
                )
        L.append("")

    # ---------------- efficiency
    L.append("## Efficiency (measured by Ollama on this machine)\n")
    L.append("| Metric | Value |")
    L.append("|---|---:|")
    L.append(f"| Generation speed | {eff['gen_tokens_per_s']:.1f} tokens/s |")
    L.append(f"| Prompt processing speed | {eff['prompt_tokens_per_s']:.0f} tokens/s |")
    L.append(f"| Mean latency per answer | {eff['mean_latency_s']:.2f} s |")
    L.append(f"| p95 latency | {eff['p95_latency_s']:.2f} s |")
    L.append(f"| Mean output length | {eff['mean_output_tokens']:.0f} tokens |")
    L.append(f"| Total compute time (answers only) | {eff['total_compute_s'] / 60:.1f} min |")
    L.append(f"| Outputs cut off at max tokens | {eff['truncated_outputs']} |")
    L.append(f"| Failed calls | {eff['failed_calls']} / {eff['calls']} |")
    L.append("")

    # ---------------- examples
    preds = ctx["predictions"][model]
    items = ctx["items_by_id"]
    judged = [r for r in preds if r["task"] in ("grounded_qa", "rag_qa") and r["scores"].get("judge_score") is not None]
    fails = sorted(
        [r for r in judged if r["scores"]["judge_score"] < 1.0], key=lambda r: (r["scores"]["judge_score"], r["item_id"])
    )[:4]
    wins = [r for r in judged if r["scores"]["judge_score"] == 1.0][:2]
    if fails:
        L.append("## Example errors\n")
        for r in fails:
            L.extend(_example(r, items[r["item_id"]]))
    if wins:
        L.append("## Example correct answers\n")
        for r in wins:
            L.extend(_example(r, items[r["item_id"]]))
    un_fail = [r for r in preds if r["task"] == "unanswerable" and not r["scores"].get("abstained")][:2]
    if un_fail:
        L.append("## Example hallucinations (answer not in the excerpt)\n")
        for r in un_fail:
            it = items[r["item_id"]]
            L.append(f"**Q:** {it['question']}  \n**Model said:** {_clip(r['scores']['prediction'])}  \n"
                     f"*(The excerpt shown was from {it['context'][:60].splitlines()[0]} … and did not contain the answer.)*\n")
    return "\n".join(L)


def _clip(text: str, n: int = 400) -> str:
    text = " ".join((text or "").split())
    return text if len(text) <= n else text[:n] + " …"


def _example(r: dict, it: dict) -> list[str]:
    verdicts = ", ".join(f"{j}: {v['verdict']}" for j, v in (r.get("judges") or {}).items()) or r["scores"].get(
        "judge_note", ""
    )
    reasons = " | ".join(_clip(v["reasoning"], 200) for v in (r.get("judges") or {}).values() if v.get("reasoning"))
    return [
        f"**[{TASK_LABEL[r['task']]} · {it['question_type']} · difficulty {it['difficulty']} · "
        f"p. {', '.join(map(str, it['pages']))}]**  ",
        f"**Q:** {it['question']}  ",
        f"**Gold:** {_clip(it['gold'])}  ",
        f"**Model:** {_clip(r['scores'].get('prediction', ''))}  ",
        f"**Judges:** {verdicts}" + (f" — _{reasons}_" if reasons else ""),
        "",
    ]


# ============================================================================ comparison


def comparison_report(ctx: dict, chart_paths: dict[str, Path]) -> str:
    results, labels = ctx["results"], ctx["labels"]
    models = results["ranking"]
    tasks = _tasks_present(results)
    L: list[str] = []
    L.append("# SLM Comparison — Annual-Report Benchmark (ARBench)\n")
    L.append(
        f"**Document:** {ctx['bench_title']} ({ctx['doc_pages']} pages, {ctx['n_chunks']} chunks) · "
        f"**Benchmark:** {ctx['n_questions']} validated questions → {ctx['n_items']} eval items · "
        f"**Judging:** {ctx['judge_desc']} · **Run:** `{ctx['run_name']}` ({ctx['profile']} profile) · "
        f"**Ollama:** {ctx['ollama_version']} · **Host:** {ctx['host']}\n"
    )
    # leaderboard
    L.append("## Leaderboard\n")
    head = "| Rank | Model | **ARBench score** | " + " | ".join(TASK_LABEL[t] for t in tasks) + " | Tokens/s | Latency |"
    L.append(head)
    L.append("|---:|---|---:|" + "---:|" * len(tasks) + "---:|---:|")
    for i, m in enumerate(models, 1):
        r = results["models"][m]
        cells = [pct(_mean(r["tasks"].get(t, {}).get("primary"))) for t in tasks]
        L.append(
            f"| {i} | {labels[m]} | **{r['composite'] * 100:.1f}** | " + " | ".join(cells)
            + f" | {r['efficiency']['gen_tokens_per_s']:.1f} | {r['efficiency']['mean_latency_s']:.1f}s |"
        )
    w = results["meta"]["weights"]
    L.append(
        "\n*ARBench score = weighted mean of the task scores ("
        + ", ".join(f"{k} {v:.0%}" for k, v in w.items() if v)
        + "); efficiency = tokens/s normalised to the fastest model.*\n"
    )
    for key in ("leaderboard", "tasks", "quality_speed"):
        if key in chart_paths:
            L.append(f"![{key}]({chart_paths[key].as_posix()})\n")

    # verdict
    L.append("## Verdict\n")
    L.extend(_verdict(ctx))
    L.append("")

    # significance
    L.append("## Are the differences significant? (paired bootstrap, 95%)\n")
    L.append("| Task | Model A | Model B | n | A − B | 95% CI | p-value | Significant |")
    L.append("|---|---|---|---:|---:|---|---:|:---:|")
    for t in tasks:
        for p in results["pairwise"].get(t, []):
            if p["n"] < 5:
                sig, ci_txt, p_txt = "too few items", "–", "–"
            else:
                sig = "yes" if p["p_value"] < 0.05 else "no"
                ci_txt = f"[{p['ci95'][0] * 100:+.1f}, {p['ci95'][1] * 100:+.1f}]"
                p_txt = f"{p['p_value']:.3f}"
            L.append(
                f"| {TASK_LABEL[t]} | {labels[p['a']]} | {labels[p['b']]} | {p['n']} | {p['diff'] * 100:+.1f} pts | "
                f"{ci_txt} | {p_txt} | {sig} |"
            )
    L.append("\n*Small benchmarks give wide intervals: treat non-significant gaps as ties.*\n")

    # detailed metric comparison
    L.append("## Metric-by-metric comparison\n")
    for t in tasks:
        all_metrics = []
        for m in models:
            for k in results["models"][m]["tasks"].get(t, {}).get("metrics", {}):
                if k not in all_metrics:
                    all_metrics.append(k)
        if not all_metrics:
            continue
        L.append(f"### {TASK_LABEL[t]}\n")
        L.append("| Metric | " + " | ".join(labels[m] for m in models) + " |")
        L.append("|---|" + "---:|" * len(models))
        for k in all_metrics:
            vals = [_mean(results["models"][m]["tasks"].get(t, {}).get("metrics", {}).get(k)) for m in models]
            valid = [v for v in vals if v is not None]
            best = (min(valid) if k in LOWER_IS_BETTER else max(valid)) if valid else None
            unique_best = valid.count(best) < len(valid)  # bold only when it actually stands out
            cells = [f"**{pct(v)}**" if v is not None and v == best and unique_best else pct(v) for v in vals]
            arrow = " ↓" if k in LOWER_IS_BETTER else ""
            L.append(f"| {METRIC_LABEL.get(k, k)}{arrow} | " + " | ".join(cells) + " |")
        L.append("")

    for key, title in (("qtype", "By question type"), ("difficulty", "By difficulty"), ("hallucination", "Hallucination profile")):
        if key in chart_paths:
            L.append(f"### {title}\n\n![{key}]({chart_paths[key].as_posix()})\n")

    # efficiency
    L.append("## Efficiency\n")
    L.append("| Model | Gen tokens/s | Prompt tokens/s | Mean latency | p95 latency | Mean output tokens | Size |")
    L.append("|---|---:|---:|---:|---:|---:|---:|")
    for m in models:
        e = results["models"][m]["efficiency"]
        L.append(
            f"| {labels[m]} | {e['gen_tokens_per_s']:.1f} | {e['prompt_tokens_per_s']:.0f} | {e['mean_latency_s']:.2f}s "
            f"| {e['p95_latency_s']:.2f}s | {e['mean_output_tokens']:.0f} | {ctx['model_info'].get(m, {}).get('size_gb', '?')} GB |"
        )
    L.append("")

    # judge analysis
    if ctx.get("judge_stats"):
        js = ctx["judge_stats"]
        L.append("## Judge (LLM-jury) diagnostics\n")
        agreement = (
            f"**{pct(js['agreement'])}**" if js.get("agreement") is not None else "n/a (each answer had a single judge)"
        )
        L.append(
            f"Inter-judge agreement on identical answers: {agreement}. "
            "Leniency = mean score a judge awards to the answers it grades (higher = more lenient).\n"
        )
        L.append("| Judge | Answers graded | Mean score given | Unparseable verdicts |")
        L.append("|---|---:|---:|---:|")
        for j, s in js["judges"].items():
            L.append(f"| {labels.get(j, j)} | {s['n']} | {pct(s['mean'])} | {s['invalid']} |")
        L.append("")

    # self preference
    L.append("## Self-preference check (questions written by the model itself vs. others)\n")
    L.append("| Model | Own questions | Others' questions | Gap |")
    L.append("|---|---:|---:|---:|")
    for m in models:
        g = results["models"][m]["breakdowns"].get("generator", {})
        own, oth = g.get("own questions", {}).get("mean"), g.get("other models' questions", {}).get("mean")
        gap = f"{(own - oth) * 100:+.1f} pts" if own is not None and oth is not None else "–"
        L.append(f"| {labels[m]} | {pct(own)} | {pct(oth)} | {gap} |")
    L.append("")

    L.append("## Benchmark composition\n")
    bs = ctx["bench_stats"]
    L.append(f"- Questions by kind: {bs.get('kept_by_kind')}")
    L.append(f"- Questions by type: {bs.get('kept_by_type')}")
    L.append(f"- Questions by generator: {bs.get('kept_by_generator')}")
    L.append(f"- Eval items by task: {ctx['items_by_task']}")
    L.append(f"- Filter: kept {bs.get('kept')} of {bs.get('input')} generated; rejections {bs.get('rejected')}")
    L.append("")
    L.append(METHODOLOGY)
    return "\n".join(L)


def _verdict(ctx: dict) -> list[str]:
    results, labels = ctx["results"], ctx["labels"]
    models = results["ranking"]
    out = []
    best = models[0]
    out.append(f"- **Best overall:** {labels[best]} (ARBench {results['models'][best]['composite'] * 100:.1f}).")

    for task, phrase in (
        ("grounded_qa", "Most accurate reader of the report (grounded QA)"),
        ("rag_qa", "Best for retrieval-augmented Q&A over the full report"),
        ("multi_choice", "Best at multiple choice"),
        ("unanswerable", "Most honest when the answer is missing (least hallucination)"),
    ):
        cands = [(m, _mean(results["models"][m]["tasks"].get(task, {}).get("primary"))) for m in models]
        cands = [c for c in cands if c[1] is not None]
        if not cands:
            continue
        top = max(v for _, v in cands)
        leaders = [m for m, v in cands if abs(v - top) < 1e-9]
        if len(leaders) > 1:
            out.append(f"- **{phrase}:** tie between {', '.join(labels[m] for m in leaders)} ({pct(top)}).")
            continue
        if len(cands) == 1:
            out.append(f"- **{phrase}:** {labels[leaders[0]]} ({pct(top)}).")
            continue
        sig = _is_sig_best(results, task, leaders[0])
        out.append(
            f"- **{phrase}:** {labels[leaders[0]]} ({pct(top)})"
            f"{' — significantly ahead of every other model' if sig else ' — not significantly ahead of the runner-up'}."
        )
    fastest = max(models, key=lambda m: results["models"][m]["efficiency"]["gen_tokens_per_s"])
    out.append(
        f"- **Fastest:** {labels[fastest]} ({results['models'][fastest]['efficiency']['gen_tokens_per_s']:.1f} tokens/s)."
    )
    worst_h = max(
        models,
        key=lambda m: (_mean(results["models"][m]["tasks"].get("grounded_qa", {}).get("metrics", {}).get("number_hallucination")) or 0),
    )
    v = _mean(results["models"][worst_h]["tasks"].get("grounded_qa", {}).get("metrics", {}).get("number_hallucination"))
    if v:
        out.append(f"- **Watch out:** {labels[worst_h]} states the most unsupported figures ({pct(v)} of its numbers in grounded QA).")
    return out


def _is_sig_best(results: dict, task: str, model: str) -> bool:
    pairs = [p for p in results["pairwise"].get(task, []) if model in (p["a"], p["b"])]
    if not pairs:
        return False
    for p in pairs:
        diff = p["diff"] if p["a"] == model else -p["diff"]
        if diff <= 0 or p["p_value"] >= 0.05:
            return False
    return True


METHODOLOGY = """## Methodology

**Pipeline (YourBench → LightEval, re-engineered to run 100% locally on Ollama):**

1. **Ingestion** – the annual report is converted to Markdown (tables preserved), running headers/footers and page numbers removed.
2. **Chunking** – structure-aware packing of paragraphs/tables into ~700-token chunks with overlap; tables of contents and boilerplate are filtered by a quality score.
3. **Summarisation** – a short hierarchical summary orients the question generator (never shown to evaluated models).
4. **Question generation** – single-hop, multi-hop (2 related chunks) and multiple-choice questions with verbatim citations, written in rotation by the models under test (JSON-schema constrained decoding).
5. **Filtering** – YourBench citation-grounding score, number grounding (gold figures must exist in the source), hygiene checks and near-duplicate removal.
5b. **Verification** – every item is re-checked against its evidence by the models that did not write it; items they unanimously reject (e.g. a mislabelled MCQ answer or an unsupported gold answer) are removed.
6. **Evaluation** – LightEval-style tasks: grounded QA, RAG QA (BM25 top-k), multiple choice, unanswerable (hallucination probe) and optional closed-book; greedy decoding (temperature 0, fixed seed).
7. **Judging** – free-text answers are graded correct / partially correct / incorrect by an LLM jury; with the default *cross* strategy each answer is graded by the *other* models to avoid self-grading.
8. **Statistics** – means with standard errors and bootstrap 95% CIs; paired bootstrap tests between models.

**Caveats:** questions and judgements come from small models, so the benchmark measures *relative* ability on this report; inspect the example errors and the `benchmark/` files. Confidence intervals are wide when the benchmark is small – use the `full`/`colab` profile for decisions.
"""


# ============================================================================ orchestration


def judge_statistics(predictions: dict[str, list[dict]]) -> dict | None:
    judges: dict[str, dict] = {}
    agree, agree_n = 0, 0
    for recs in predictions.values():
        for r in recs:
            js = r.get("judges") or {}
            for j, v in js.items():
                s = judges.setdefault(j, {"n": 0, "sum": 0.0, "invalid": 0})
                if v.get("score") is None:
                    s["invalid"] += 1
                else:
                    s["n"] += 1
                    s["sum"] += v["score"]
            if r["scores"].get("judge_agreement") is not None:
                agree_n += 1
                agree += r["scores"]["judge_agreement"]
    if not judges:
        return None
    return {
        "judges": {j: {"n": s["n"], "mean": s["sum"] / s["n"] if s["n"] else None, "invalid": s["invalid"]} for j, s in judges.items()},
        "agreement": agree / agree_n if agree_n else None,
    }


def write_reports(ctx: dict, out_dir: Path) -> dict[str, Path]:
    results, labels = ctx["results"], ctx["labels"]
    models_cfg_order = ctx["models"]
    colors = charts.model_colors(models_cfg_order)
    rep = out_dir / "reports"
    chart_dir = rep / "charts"
    tasks = _tasks_present(results)
    paths: dict[str, Path] = {}

    # ---------------- charts
    chart_paths: dict[str, Path] = {}
    try:
        chart_paths["leaderboard"] = charts.leaderboard(
            models_cfg_order, {m: results["models"][m]["composite"] for m in models_cfg_order}, labels, colors,
            chart_dir / "leaderboard.png",
        )
        series = {m: [_mean(results["models"][m]["tasks"].get(t, {}).get("primary")) for t in tasks] for m in models_cfg_order}
        errs = {
            m: [tuple(results["models"][m]["tasks"][t]["primary"]["ci95"]) if results["models"][m]["tasks"].get(t, {}).get("primary") else None for t in tasks]
            for m in models_cfg_order
        }
        chart_paths["tasks"] = charts.grouped_bars(
            [TASK_LABEL[t] for t in tasks], series, labels, colors, "Score by task (with 95% CI)", chart_dir / "tasks.png", errs
        )
        qtypes = sorted({k for m in models_cfg_order for k, v in results["models"][m]["breakdowns"].get("question_type", {}).items()})
        if qtypes:
            chart_paths["qtype"] = charts.grouped_bars(
                qtypes,
                {m: [results["models"][m]["breakdowns"]["question_type"].get(q, {}).get("mean") for q in qtypes] for m in models_cfg_order},
                labels, colors, "Score by question type", chart_dir / "question_types.png",
            )
        diffs = ["easy (1-3)", "medium (4-7)", "hard (8-10)"]
        chart_paths["difficulty"] = charts.grouped_bars(
            diffs,
            {m: [results["models"][m]["breakdowns"].get("difficulty", {}).get(d, {}).get("mean") for d in diffs] for m in models_cfg_order},
            labels, colors, "Score by difficulty", chart_dir / "difficulty.png",
        )
        hall_groups = ["Correct abstention ↑", "Over-refusal ↓", "Number hallucination ↓"]

        def hv(m):
            t = results["models"][m]["tasks"]
            return [
                _mean(t.get("unanswerable", {}).get("primary")),
                _mean(t.get("grounded_qa", {}).get("metrics", {}).get("over_refusal")),
                _mean(t.get("grounded_qa", {}).get("metrics", {}).get("number_hallucination")),
            ]

        chart_paths["hallucination"] = charts.grouped_bars(
            hall_groups, {m: hv(m) for m in models_cfg_order}, labels, colors, "Hallucination profile",
            chart_dir / "hallucination.png",
        )
        quality = {}
        for m in models_cfg_order:
            vals = [_mean(results["models"][m]["tasks"].get(t, {}).get("primary")) for t in tasks]
            vals = [v for v in vals if v is not None]
            quality[m] = sum(vals) / len(vals) if vals else 0
        chart_paths["quality_speed"] = charts.quality_vs_speed(
            models_cfg_order, quality, {m: results["models"][m]["efficiency"]["gen_tokens_per_s"] for m in models_cfg_order},
            labels, colors, chart_dir / "quality_vs_speed.png",
        )
    except Exception as exc:  # charts must never break the run
        log.warning("Chart generation failed: %s", exc)

    rel = {k: p.relative_to(rep) for k, p in chart_paths.items()}

    # ---------------- comparison
    comp_md = comparison_report(ctx, rel)
    paths["comparison"] = rep / "comparison_report.md"
    paths["comparison"].write_text(comp_md, encoding="utf-8")

    # ---------------- per model
    model_mds = {}
    for m in models_cfg_order:
        cp = None
        try:
            r = results["models"][m]
            cp = charts.single_model_tasks(
                [TASK_LABEL[t] for t in tasks],
                [_mean(r["tasks"].get(t, {}).get("primary")) for t in tasks],
                [tuple(r["tasks"][t]["primary"]["ci95"]) if r["tasks"].get(t, {}).get("primary") else None for t in tasks],
                colors[m],
                f"{labels[m]} — score by task",
                chart_dir / f"{slug(m)}_tasks.png",
            ).relative_to(rep)
        except Exception as exc:
            log.warning("Chart for %s failed: %s", m, exc)
        md = model_report(m, ctx, cp)
        model_mds[m] = md
        p = rep / f"{slug(m)}_report.md"
        p.write_text(md, encoding="utf-8")
        paths[m] = p

    # ---------------- HTML dashboard (self-contained)
    sections = [("Comparison", comp_md)] + [(labels[m], model_mds[m]) for m in models_cfg_order]
    html = markdown_to_html_page("ARBench Results", sections, rep)
    paths["html"] = rep / "report.html"
    paths["html"].write_text(html, encoding="utf-8")

    # ---------------- CSV
    csv_path = out_dir / "results" / "summary.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["model", "label", "composite"] + [f"{t}_{results['models'][models_cfg_order[0]]['tasks'].get(t, {}).get('primary_metric', '')}" for t in tasks]
                   + ["gen_tokens_per_s", "mean_latency_s"])
        for m in results["ranking"]:
            r = results["models"][m]
            w.writerow([m, labels[m], round(r["composite"], 4)]
                       + [round(_mean(r["tasks"].get(t, {}).get("primary")) or 0, 4) for t in tasks]
                       + [round(r["efficiency"]["gen_tokens_per_s"], 2), round(r["efficiency"]["mean_latency_s"], 2)])
    paths["csv"] = csv_path
    return paths
