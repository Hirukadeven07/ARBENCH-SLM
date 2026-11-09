"""End-to-end orchestration: one call runs the whole benchmark.

    preflight -> ingest -> chunk -> summarise -> generate -> filter -> build tasks
              -> evaluate -> judge -> score -> reports

Every stage writes its artefacts to outputs/<run>/ and every LLM call is cached, so an
interrupted run continues where it stopped when the same command is executed again.
"""

from __future__ import annotations

import logging
import platform
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

from arbench import __version__
from arbench.config import PROJECT_ROOT, Config, model_label, model_names, resolve_models, resolve_path
from arbench.evals.runner import run_evaluation
from arbench.evals.scoring import aggregate
from arbench.evals.tasks import build_items
from arbench.export import export_benchmark
from arbench.llm.cache import LLMCache
from arbench.llm.ollama_client import OllamaClient
from arbench.pipeline.chunk import Chunk, run_chunking
from arbench.pipeline.filter import run_filtering
from arbench.pipeline.generate import run_generation
from arbench.pipeline.ingest import run_ingestion
from arbench.pipeline.summarize import summarize
from arbench.pipeline.verify import run_verification
from arbench.reporting.report import judge_statistics, write_reports
from arbench.utils import read_jsonl, write_json, write_jsonl

log = logging.getLogger("arbench")


class StageTimer:
    def __init__(self):
        self.times: dict[str, float] = {}

    def __call__(self, name: str):
        timer = self

        class _Ctx:
            def __enter__(self_inner):
                log.info("━━━ %s ━━━", name)
                self_inner.t0 = time.perf_counter()

            def __exit__(self_inner, *exc):
                timer.times[name] = round(time.perf_counter() - self_inner.t0, 1)
                log.info("    done in %s", _fmt_s(timer.times[name]))

        return _Ctx()


def _fmt_s(s: float) -> str:
    return f"{s:.0f}s" if s < 90 else f"{s / 60:.1f} min" if s < 5400 else f"{s / 3600:.1f} h"


def setup_logging(run_dir: Path, verbose: bool = False) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s  %(levelname)-7s %(message)s", "%H:%M:%S")
    root = logging.getLogger()
    root.setLevel(logging.DEBUG if verbose else logging.INFO)
    for h in list(root.handlers):
        root.removeHandler(h)
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    fh = logging.FileHandler(run_dir / "run.log", encoding="utf-8")
    fh.setFormatter(fmt)
    root.addHandler(sh)
    root.addHandler(fh)
    for noisy in ("urllib3", "matplotlib", "PIL", "httpx"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def _chunks_from_rows(rows: list[dict]) -> list[Chunk]:
    return [Chunk(**r) for r in rows]


def benchmark_card(ctx: dict) -> str:
    bs = ctx["bench_stats"]
    return f"""# Annual-Report Benchmark — card

**Source document(s):** {', '.join(ctx['doc_files'])}
**Title:** {ctx['bench_title']} · {ctx['doc_pages']} pages · {ctx['n_chunks']} usable chunks
**Created:** {ctx['created']} with ARBench {__version__} (profile `{ctx['profile']}`)

## Contents
- `questions.jsonl` — {bs['kept']} validated questions (kind: {bs['kept_by_kind']})
- `annual_report_benchmark.jsonl` — {ctx['n_items']} evaluation items (task: {ctx['items_by_task']})
- `lighteval/` — the same questions in the YourBench → LightEval schema + `custom_task.py`

## How it was built
- Question generators (rotated): {', '.join(ctx['generators'])}
- Generated {bs['input']} candidates → kept {bs['kept']} after citation grounding (≥ {ctx['filter_cfg']['min_citation_score']}),
  number grounding (≥ {ctx['filter_cfg']['min_number_grounding']}), hygiene checks, near-duplicate removal and
  cross-model verification (items unanimously rejected by the non-author models are dropped).
- Rejections: {bs['rejected']}
- Question types: {bs['kept_by_type']}

## Item fields
`task`, `question`, `gold`, `choices`/`gold_index` (MCQ), `context` (what the model sees), `evidence` (gold chunks used by
the judge), `question_type`, `difficulty`, `generator`, `pages`, `section`, `retrieval_recall` (RAG), `expected` (unanswerable).
"""


def run(cfg: Config, run_name: str | None = None, reuse_benchmark: str | None = None, generate_only: bool = False,
        verbose: bool = False) -> Path:
    started = time.time()
    run_name = run_name or datetime.now().strftime("%Y%m%d_%H%M%S") + f"_{cfg['profile']}"
    out_root = resolve_path(cfg["output_dir"])
    run_dir = out_root / run_name
    setup_logging(run_dir, verbose)
    timer = StageTimer()
    log.info("ARBench %s · run '%s' · profile '%s'", __version__, run_name, cfg["profile"])
    log.info("Outputs → %s", run_dir)
    write_json(run_dir / "config_used.json", cfg)

    models = model_names(cfg)
    labels = {m: model_label(cfg, m) for m in models}
    generators = resolve_models(cfg, cfg["generation"].get("models"))
    fixed_judges = resolve_models(cfg, cfg["judge"].get("models")) if cfg["judge"]["strategy"] == "fixed" else []
    summary_model = cfg["summarization"].get("model")
    summary_model = generators[0] if summary_model in (None, "auto") else summary_model

    cache = LLMCache(resolve_path(cfg["cache_path"]) if cfg.get("cache_path") else None)
    client = OllamaClient.from_config(cfg, cache)

    # ------------------------------------------------------------------ preflight
    with timer("0 · Preflight (Ollama + models)"):
        client.ensure_server(cfg["ollama"]["auto_start"])
        needed = list(dict.fromkeys(models + ([] if reuse_benchmark else generators + [summary_model]) + fixed_judges))
        model_info = client.ensure_models(needed, cfg["ollama"]["auto_pull"])
        for m in needed:
            i = model_info[m]
            log.info("  ✓ %-28s %s · %s · %s GB", m, i.get("parameter_size", "?"), i.get("quantization_level", "?"), i.get("size_gb"))

    bench_dir = run_dir / "benchmark"
    if reuse_benchmark:
        # ---------------------------------------------------------- reuse an existing benchmark
        src = Path(reuse_benchmark)
        src = src if (src / "benchmark").exists() else resolve_path(reuse_benchmark)
        log.info("Reusing benchmark from %s", src)
        items = read_jsonl(src / "benchmark" / "annual_report_benchmark.jsonl")
        questions = read_jsonl(src / "benchmark" / "questions.jsonl")
        meta = __import__("json").loads((src / "benchmark_meta.json").read_text(encoding="utf-8"))
        bench_dir.mkdir(parents=True, exist_ok=True)
        for f in (src / "benchmark").rglob("*"):
            if f.is_file():
                dest = bench_dir / f.relative_to(src / "benchmark")
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(f.read_bytes())
        write_json(run_dir / "benchmark_meta.json", meta)
    else:
        # ---------------------------------------------------------- 1 ingestion
        with timer("1 · Ingestion (extract text from the annual report)"):
            docs = run_ingestion(resolve_path(cfg["input_dir"]), cfg["ingestion"], PROJECT_ROOT, run_dir / "01_extracted")
        titles = {d.doc_id: d.title for d in docs}

        # ---------------------------------------------------------- 2 chunking
        with timer("2 · Chunking"):
            chunks = run_chunking(docs, cfg["chunking"])
            write_jsonl(run_dir / "02_chunks.jsonl", [c.to_dict() for c in chunks])
            log.info("  %d chunks total", len(chunks))

        # ---------------------------------------------------------- 3 summarisation
        summaries: dict[str, str] = {}
        with timer("3 · Summarisation"):
            for d in docs:
                dchunks = [c for c in chunks if c.doc_id == d.doc_id]
                if cfg["summarization"]["enabled"] and dchunks:
                    summaries[d.doc_id] = summarize(client, summary_model, d.title, dchunks, cfg["summarization"]["max_chunks"])
                else:
                    summaries[d.doc_id] = d.title
            write_json(run_dir / "03_summaries.json", {titles[k]: v for k, v in summaries.items()})

        # ---------------------------------------------------------- 4 generation
        with timer("4 · Question generation (YourBench)"):
            raw_questions, gen_stats = run_generation(
                client, chunks, summaries, titles, cfg["generation"], generators, cfg["seed"], int(cfg["ollama"]["concurrency"])
            )
            write_jsonl(run_dir / "04_questions_raw.jsonl", raw_questions)
            if not raw_questions:
                raise RuntimeError("No questions could be generated – check run.log for model errors.")

        # ---------------------------------------------------------- 5 filtering
        chunk_text = {c.chunk_id: c.text for c in chunks}
        with timer("5 · Filtering (citation + number grounding, dedup)"):
            questions, filt_stats = run_filtering(raw_questions, chunk_text, cfg["filtering"])
            if not questions:
                raise RuntimeError("All generated questions were filtered out – lower filtering thresholds in the config.")

        if cfg["filtering"].get("verify", True):
            with timer("5b · Cross-model verification of questions"):
                verifiers = resolve_models(cfg, cfg["filtering"].get("verifiers"))
                questions, vstats = run_verification(
                    client, questions, chunk_text, verifiers, int(cfg["ollama"]["concurrency"])
                )
                filt_stats["verification"] = vstats
                for reason, n in vstats["dropped"].items():
                    filt_stats["rejected"][reason] = filt_stats["rejected"].get(reason, 0) + n
                filt_stats["kept"] = len(questions)
                filt_stats["kept_by_kind"] = dict(Counter(q["kind"] for q in questions))
                filt_stats["kept_by_generator"] = dict(Counter(q["generator"] for q in questions))
                filt_stats["kept_by_type"] = dict(Counter(q["question_type"] for q in questions))
                if not questions:
                    raise RuntimeError("Verification rejected every question – check run.log.")

        # ---------------------------------------------------------- 6 tasks
        with timer("6 · Building benchmark tasks (LightEval)"):
            items = build_items(questions, chunks, cfg["evaluation"], cfg["seed"])
            items_by_task = dict(Counter(i["task"] for i in items))
            log.info("  %d eval items: %s", len(items), items_by_task)
            meta = {
                "doc_files": [d.filename for d in docs],
                "bench_title": " + ".join(d.title for d in docs),
                "doc_pages": sum(d.meta["pages"] for d in docs),
                "n_chunks": len(chunks),
                "generators": generators,
                "gen_stats": gen_stats,
                "bench_stats": filt_stats,
                "summaries": summaries,
                "created": datetime.now().isoformat(timespec="seconds"),
            }
            write_json(run_dir / "benchmark_meta.json", meta)
            card_ctx = {**meta, "profile": cfg["profile"], "n_items": len(items), "items_by_task": items_by_task,
                        "filter_cfg": cfg["filtering"]}
            judge_for_export = fixed_judges[0] if fixed_judges else models[-1]
            export_benchmark(bench_dir, questions, items, chunk_text, summaries, judge_for_export, benchmark_card(card_ctx))
            log.info("  Benchmark written to %s", bench_dir)

    if generate_only:
        log.info("--generate-only: stopping after benchmark creation (%s).", _fmt_s(time.time() - started))
        _write_latest(out_root, run_name)
        return run_dir

    # ------------------------------------------------------------------ evaluation
    n_judge_calls = 0
    if cfg["judge"]["enabled"]:
        judged = sum(1 for i in items if i["task"] in ("grounded_qa", "rag_qa", "closed_book"))
        per = 1 if cfg["judge"]["strategy"] in ("self", "fixed") and len(fixed_judges) <= 1 else max(1, len(models) - 1)
        n_judge_calls = judged * len(models) * (len(fixed_judges) if cfg["judge"]["strategy"] == "fixed" else per)
    log.info(
        "Workload: %d answer calls + up to %d judge calls (cached calls are free).", len(items) * len(models), n_judge_calls
    )
    with timer("7 · Evaluation + LLM jury (LightEval)"):
        predictions = run_evaluation(client, items, models, cfg, run_dir, fixed_judges)

    # ------------------------------------------------------------------ scoring + reports
    with timer("8 · Scoring + reports"):
        results = aggregate(predictions, items, cfg)
        js = judge_statistics(predictions)
        judge_desc = {
            "cross": "cross-model jury (each answer graded by the other models)",
            "fixed": f"fixed judge(s): {', '.join(fixed_judges)}",
            "self": "self-grading",
        }[cfg["judge"]["strategy"]] if cfg["judge"]["enabled"] else "disabled (token F1 used)"
        ctx = {
            "results": results,
            "labels": labels,
            "models": models,
            "model_info": model_info,
            "predictions": predictions,
            "items_by_id": {i["item_id"]: i for i in items},
            "bench_title": meta["bench_title"],
            "doc_pages": meta["doc_pages"],
            "n_chunks": meta["n_chunks"],
            "n_questions": len(questions),
            "n_items": len(items),
            "items_by_task": dict(Counter(i["task"] for i in items)),
            "bench_stats": meta["bench_stats"],
            "run_name": run_name,
            "profile": cfg["profile"],
            "judge_desc": judge_desc,
            "judge_stats": js,
            "ollama_version": client.version(),
            "host": f"{platform.system()} {platform.machine()} · {platform.processor() or 'cpu'}",
        }
        write_json(
            run_dir / "results" / "results.json",
            {
                "run": run_name,
                "profile": cfg["profile"],
                "arbench_version": __version__,
                "document": meta["bench_title"],
                "models": {m: {"label": labels[m], **model_info.get(m, {})} for m in models},
                "ranking": results["ranking"],
                "scores": results["models"],
                "pairwise": results["pairwise"],
                "judge_stats": js,
                "stage_times_s": timer.times,
            },
        )
        paths = write_reports(ctx, run_dir)

    _write_latest(out_root, run_name)
    total = time.time() - started
    log.info("━━━ Finished in %s ━━━", _fmt_s(total))
    log.info("Leaderboard:")
    for i, m in enumerate(results["ranking"], 1):
        log.info("  %d. %-22s ARBench %.1f", i, labels[m], results["models"][m]["composite"] * 100)
    log.info("Comparison report : %s", paths["comparison"])
    log.info("HTML dashboard    : %s", paths["html"])
    log.info("Benchmark (evals) : %s", bench_dir / "annual_report_benchmark.jsonl")
    cache.close()
    return run_dir


def _write_latest(out_root: Path, run_name: str) -> None:
    (out_root / "LATEST.txt").write_text(run_name + "\n", encoding="utf-8")
