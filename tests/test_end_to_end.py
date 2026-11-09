"""End-to-end run of the whole pipeline with a fake Ollama client (no network, seconds)."""

from __future__ import annotations

import json

from arbench import orchestrator
from arbench.config import load_config
from arbench.utils import read_jsonl
from tests.conftest import FakeClient, make_sample_pdf


def _cfg(tmp_path, input_dir, **extra):
    overrides = {
        "input_dir": str(input_dir),
        "output_dir": str(tmp_path / "outputs"),
        "cache_path": None,
        "models": [{"name": "alpha:1b", "label": "Alpha"}, {"name": "beta:2b", "label": "Beta"}, {"name": "gamma:3b", "label": "Gamma"}],
        "chunking": {"target_tokens": 120, "max_tokens": 200, "overlap_tokens": 20, "min_tokens": 40, "min_quality": 0.2},
        "generation": {"single_hop": {"num_chunks": 4, "questions_per_chunk": 2}, "multi_hop": {"num_groups": 1},
                       "multi_choice": {"num_chunks": 2, "questions_per_chunk": 1}},
        "evaluation": {"unanswerable_fraction": 0.5},
        "scoring": {"bootstrap_samples": 200},
    }
    overrides.update(extra)
    return load_config(profile="smoke", overrides=overrides)


def test_full_pipeline_with_fake_models(tmp_path, monkeypatch, sample_report_dir):
    monkeypatch.setattr(orchestrator, "OllamaClient", FakeClient)
    cfg = _cfg(tmp_path, sample_report_dir)
    run_dir = orchestrator.run(cfg, run_name="e2e")

    # benchmark file with evals
    items = read_jsonl(run_dir / "benchmark" / "annual_report_benchmark.jsonl")
    tasks = {i["task"] for i in items}
    assert {"grounded_qa", "rag_qa", "multi_choice"} <= tasks
    assert (run_dir / "benchmark" / "lighteval" / "custom_task.py").exists()
    compile((run_dir / "benchmark" / "lighteval" / "custom_task.py").read_text(encoding="utf-8"), "custom_task.py", "exec")

    # results + reports
    res = json.loads((run_dir / "results" / "results.json").read_text(encoding="utf-8"))
    assert set(res["ranking"]) == {"alpha:1b", "beta:2b", "gamma:3b"}
    for m in res["ranking"]:
        assert 0 <= res["scores"][m]["composite"] <= 1
    reports = run_dir / "reports"
    assert (reports / "comparison_report.md").exists()
    assert (reports / "report.html").stat().st_size > 10_000
    for m in ("alpha_1b", "beta_2b", "gamma_3b"):
        text = (reports / f"{m}_report.md").read_text(encoding="utf-8")
        assert "Results by task" in text and "Efficiency" in text
    assert (run_dir / "results" / "summary.csv").exists()

    # cross-jury: no model judges itself
    for pfile in (run_dir / "predictions").glob("*.jsonl"):
        for rec in read_jsonl(pfile):
            assert rec["model"] not in (rec.get("judges") or {})


def test_reuse_benchmark_and_pdf_input(tmp_path, monkeypatch):
    monkeypatch.setattr(orchestrator, "OllamaClient", FakeClient)
    pdf_dir = tmp_path / "pdfs"
    pdf_dir.mkdir()
    make_sample_pdf(pdf_dir / "report.pdf")
    cfg = _cfg(tmp_path, pdf_dir)
    first = orchestrator.run(cfg, run_name="gen", generate_only=True)
    assert (first / "benchmark" / "questions.jsonl").exists()
    assert not (first / "results").exists()

    cfg2 = _cfg(tmp_path, pdf_dir, judge={"strategy": "fixed", "models": ["alpha:1b"]})
    second = orchestrator.run(cfg2, run_name="eval", reuse_benchmark=str(first))
    res = json.loads((second / "results" / "results.json").read_text(encoding="utf-8"))
    assert res["judge_stats"]["judges"].keys() == {"alpha:1b"}
