#!/usr/bin/env python
"""ARBench – one command to benchmark Granite 4.1, Gemma 4 and Qwen 3 on your annual report.

    python run_benchmark.py                      # standard profile, models from config/default.yaml
    python run_benchmark.py --profile quick      # faster, fewer questions
    python run_benchmark.py --profile colab      # Google Colab L4 GPU
    python run_benchmark.py --models qwen3:4b-instruct granite4.1:3b
    python run_benchmark.py --reuse-benchmark outputs/<run>   # evaluate new models on an existing benchmark
    python run_benchmark.py --generate-only      # only build the benchmark file

Re-running the same command resumes: every model call is cached in outputs/.cache.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from arbench.config import load_config  # noqa: E402


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Local annual-report benchmark for small language models (Ollama).")
    p.add_argument("--config", default=None, help="YAML config (default: config/default.yaml)")
    p.add_argument("--profile", default="standard", help="smoke | quick | standard | full | colab")
    p.add_argument("--input", default=None, help="folder containing the annual report(s)")
    p.add_argument("--models", nargs="+", default=None, help="override Ollama model tags to benchmark")
    p.add_argument("--run-name", default=None, help="name of the output folder (reuse it to resume)")
    p.add_argument("--reuse-benchmark", default=None, help="path of a previous run whose benchmark should be reused")
    p.add_argument("--generate-only", action="store_true", help="stop after building the benchmark")
    p.add_argument("--no-judge", action="store_true", help="skip the LLM jury (use token-F1 instead)")
    p.add_argument("--judge", default=None, help="use one fixed judge model (e.g. qwen3:8b) instead of the cross jury")
    p.add_argument("--concurrency", type=int, default=None, help="parallel requests per model (GPU: 4)")
    p.add_argument("--host", default=None, help="Ollama URL (default http://127.0.0.1:11434)")
    p.add_argument("--verbose", action="store_true")
    return p.parse_args(argv)


def main(argv=None) -> int:
    for stream in (sys.stdout, sys.stderr):  # safe Unicode logging on Windows consoles / redirects
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    args = parse_args(argv)
    overrides: dict = {}
    if args.input:
        overrides["input_dir"] = args.input
    if args.models:
        overrides["models"] = [{"name": m, "label": m} for m in args.models]
    if args.no_judge:
        overrides.setdefault("judge", {})["enabled"] = False
    if args.judge:
        overrides.setdefault("judge", {}).update({"strategy": "fixed", "models": [args.judge]})
    if args.concurrency:
        overrides.setdefault("ollama", {})["concurrency"] = args.concurrency
    if args.host:
        overrides.setdefault("ollama", {})["host"] = args.host

    cfg = load_config(args.config, args.profile, overrides)
    if args.models:  # keep friendly labels for known families
        known = {"granite4.1": "Granite 4.1", "gemma4": "Gemma 4", "qwen3": "Qwen 3"}
        for m in cfg["models"]:
            fam = next((v for k, v in known.items() if m["name"].startswith(k)), None)
            if fam:
                m["label"] = f"{fam} ({m['name'].split(':', 1)[-1]})"

    from arbench.orchestrator import run

    try:
        run(cfg, args.run_name, args.reuse_benchmark, args.generate_only, args.verbose)
    except KeyboardInterrupt:
        print("\nInterrupted – re-run the same command (with --run-name) to resume from the cache.")
        return 130
    except Exception as exc:  # show a clean message, full traceback in run.log
        import logging

        logging.getLogger("arbench").exception("Run failed: %s", exc)
        print(f"\nERROR: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
