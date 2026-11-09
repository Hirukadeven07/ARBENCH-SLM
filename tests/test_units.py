"""Unit tests for the deterministic building blocks."""

from __future__ import annotations

import pytest

from arbench.config import load_config
from arbench.evals import metrics as M
from arbench.evals.scoring import bootstrap_ci, mean_stderr, paired_bootstrap
from arbench.pipeline.chunk import chunk_document, quality_score
from arbench.pipeline.filter import citation_score, number_grounding, run_filtering
from arbench.pipeline.generate import parse_job, shuffle_mcq, GenJob
from arbench.pipeline.ingest import Document, Page, clean_pages
from arbench.retrieval import BM25
from arbench.utils import extract_json, extract_numbers, numbers_match


# ----------------------------------------------------------------------------- utils
def test_extract_numbers_formats():
    vals = extract_numbers("Revenue was £2,415.6 million (2023: £2,228.4m), margin 12.9%, loss (35.2)")
    assert 2415.6 in vals and 2415.6e6 in vals
    assert 2228.4e6 in vals
    assert 12.9 in vals
    assert -35.2 in vals


def test_numbers_match_scale_and_rounding():
    assert numbers_match(1.2e9, 1200e6)
    assert numbers_match(12.34, 12.3)
    assert not numbers_match(100, 110)


def test_extract_json_variants():
    assert extract_json('{"a": 1}') == {"a": 1}
    assert extract_json('```json\n{"a": 2}\n```') == {"a": 2}
    assert extract_json('Sure! Here it is: {"a": [1, {"b": "}"}]} trailing') == {"a": [1, {"b": "}"}]}
    with pytest.raises(ValueError):
        extract_json("no json here")


# ----------------------------------------------------------------------------- metrics
def test_helm_normalizer_and_em():
    assert M.helm_normalizer("The Revenue, was 1,000!") == "revenue was 1000.0"
    assert M.helm_normalizer("1,000") == M.helm_normalizer("1000")
    assert M.exact_match("The answer.", "answer") == 1.0


def test_token_f1_and_rouge():
    assert M.token_f1("revenue rose to 10 million", "revenue rose to 10 million") == 1.0
    assert 0 < M.token_f1("revenue rose to 10 million", "revenue fell") < 1
    assert M.rouge_l("a b c d", "a b c d") == 1.0
    assert M.rouge_l("revenue grew strongly", "costs fell") == 0.0


def test_numeric_match_and_hallucination():
    assert M.numeric_match("Revenue was £2,415.6 million", "It was 2415.6m pounds") == 1.0
    assert M.numeric_match("Revenue was £2,415.6 million", "It was 2,000 million") == 0.0
    assert M.numeric_match("It grew strongly", "anything") is None
    assert M.number_hallucination("Profit was 999.9 million", "Profit was 312.9 million") == 1.0
    assert M.number_hallucination("Profit was 312.9 million", "Profit was £312.9 million") == 0.0


def test_answer_extraction_and_abstention():
    assert M.extract_answer("blah <answer> 42 </answer>") == ("42", True)
    assert M.extract_answer("just text") == ("just text", False)
    assert M.is_abstention("INSUFFICIENT_CONTEXT")
    assert M.is_abstention("The excerpts do not mention the dividend.")
    assert not M.is_abstention("Revenue was £2.4bn, up 8.4%.")


@pytest.mark.parametrize(
    "text,expected",
    [("Answer: C", 2), ("B", 1), ("(D) because", 3), ("The answer is A.", 0), ("I think option B", 1)],
)
def test_extract_letter(text, expected):
    assert M.extract_letter(text, ["w", "x", "y", "z"]) == expected


def test_score_item_tasks():
    qa = {"task": "grounded_qa", "gold": "Revenue was £2,415.6 million", "context": "Revenue was £2,415.6 million", "question": "q"}
    s = M.score_item(qa, "<answer>Revenue was £2,415.6 million</answer>")
    assert s["token_f1"] == 1.0 and s["numeric_match"] == 1.0 and s["format_ok"] and not s["abstained"]
    un = {"task": "unanswerable", "gold": "x", "context": "other text", "question": "q"}
    assert M.score_item(un, "<answer>INSUFFICIENT_CONTEXT</answer>")["abstention"] == 1.0
    mc = {"task": "multi_choice", "choices": ["a", "b", "c", "d"], "gold_index": 2}
    assert M.score_item(mc, "Answer: C")["accuracy"] == 1.0


# ----------------------------------------------------------------------------- stats
def test_stats():
    assert mean_stderr([1, 1, 1]) == 0
    lo, hi = bootstrap_ci([0, 1] * 20, 500)
    assert lo < 0.5 < hi
    res = paired_bootstrap([1] * 30, [0] * 30, 500)
    assert res["diff"] == 1 and res["p_value"] < 0.05
    res = paired_bootstrap([1, 0] * 15, [0, 1] * 15, 500)
    assert res["p_value"] > 0.05


# ----------------------------------------------------------------------------- ingestion / chunking
def test_clean_pages_removes_headers_and_page_numbers():
    pages = [Page(i, f"Northwind Annual Report 2024\nBody text {i} about revenue.\n{i}") for i in range(1, 7)]
    out = clean_pages(pages)
    assert all("Northwind Annual Report" not in p.text for p in out)
    assert all(not p.text.strip().endswith(str(p.number)) for p in out)


def test_chunking_respects_sizes_and_pages():
    para = "Revenue increased by 8.4% to £2,415.6 million as contract logistics grew strongly across Europe. " * 6
    doc = Document("d1", "r.md", "Report", [Page(i, f"# Section {i}\n\n{para}\n\n{para}") for i in range(1, 9)])
    chunks = chunk_document(doc, target_tokens=300, max_tokens=450, overlap_tokens=40, min_tokens=50)
    assert len(chunks) > 3
    assert all(c.n_tokens <= 450 + 60 for c in chunks)
    assert chunks[0].pages[0] == 1 and chunks[-1].pages[-1] == 8
    assert chunks[0].section.startswith("Section")


def test_quality_score_penalises_toc():
    toc = "\n".join(f"Chapter {i} .......... {i * 3}" for i in range(1, 30))
    prose = "The Group delivered strong revenue growth driven by automation and new customers in Europe. " * 5
    assert quality_score(toc) < 0.35 < quality_score(prose)


# ----------------------------------------------------------------------------- generation parsing / filtering
def test_shuffle_mcq_keeps_gold():
    choices, gold = shuffle_mcq(["(A) right", "B) wrong1", "wrong2", "wrong3"], "A")
    assert choices[gold] == "right"
    assert shuffle_mcq(["right", "w1", "w2", "w3"], "A") == shuffle_mcq(["right", "w1", "w2", "w3"], "A")


def test_parse_job_and_filter():
    from arbench.pipeline.chunk import Chunk

    src = "Group revenue increased by 8.4% to £2,415.6 million in 2024."
    c = Chunk("d_0000", "d", 0, src, [2], "Chair", 30, 0.8, 3)
    job = GenJob("single_hop", [c], "m1", 2)
    job.raw = (
        '{"questions": [{"question_type": "numerical", "question": "What was Northwind group revenue in 2024?", '
        '"citations": ["revenue increased by 8.4% to £2,415.6 million"], "answer": "£2,415.6 million", "estimated_difficulty": 2},'
        '{"question_type": "numerical", "question": "What was Northwind group profit in 2024?", '
        '"citations": ["profit was huge"], "answer": "£999 million", "estimated_difficulty": 2},'
        '{"question_type": "factual", "question": "According to the text, what rose?", "citations": [], "answer": "revenue", "estimated_difficulty": 1}]}'
    )
    qs = parse_job(job, {"d": "Report"})
    assert len(qs) == 3
    kept, stats = run_filtering(qs, {"d_0000": src}, {"min_citation_score": 0.6, "min_number_grounding": 0.5, "dedup_similarity": 0.9, "max_questions": 0})
    assert len(kept) == 1 and kept[0]["answer"] == "£2,415.6 million"
    assert stats["rejected"].get("meta_reference") == 1


def test_citation_and_number_grounding():
    src = "Net debt fell to £498.0 million and leverage was 1.1 times EBITDA."
    assert citation_score(["Net debt fell to £498.0 million"], src, "£498.0 million") > 0.8
    assert citation_score(["Totally invented sentence here"], src, "x") < 0.6
    assert number_grounding("£498.0 million", src) == 1.0
    assert number_grounding("£520 million", src) == 0.0


def test_bm25_ranks_relevant_first():
    bm = BM25(["dividend of 14.2 pence per share", "scope 1 emissions fell", "driver turnover reduced"])
    assert bm.top_k("What dividend per share?", 1) == [0]


def test_config_profiles():
    cfg = load_config(profile="smoke")
    assert cfg["generation"]["single_hop"]["num_chunks"] == 3
    assert cfg["generation"]["single_hop"]["questions_per_chunk"] == 2
    assert [m["name"] for m in cfg["models"]] == ["granite4.1:3b", "gemma4:e2b-it-qat", "qwen3:4b-instruct"]
    with pytest.raises(ValueError):
        load_config(profile="nope")


# ----------------------------------------------------------------------------- verification
class _ScriptedClient:
    """Returns a fixed verifier reply per model."""

    def __init__(self, replies):
        self.replies = replies

    def chat_many(self, model, reqs, concurrency=1, desc="", on_result=None):
        from arbench.llm.ollama_client import ChatResult

        return [ChatResult(text=self.replies[model], model=model) for _ in reqs]

    def unload(self, name):
        pass


def _mcq(gold_index=1, generator="gen"):
    return {"qid": "q", "kind": "multi_choice", "question": "Which risk?", "answer": "x", "choices": ["a", "b", "c", "d"],
            "gold_index": gold_index, "chunk_ids": ["c1"], "generator": generator, "question_type": "factual"}


def test_verification_mcq_consensus_rules():
    from arbench.pipeline.verify import run_verification

    models = ["gen", "v1", "v2"]
    # both verifiers say "none supported" -> dropped
    kept, st = run_verification(_ScriptedClient({"v1": '{"reasoning":"","answer":"E"}', "v2": '{"reasoning":"","answer":"E"}'}),
                                [_mcq()], {"c1": "text"}, models)
    assert kept == [] and st["dropped"] == {"mcq_consensus_no_valid_option": 1}
    # verifiers disagree with gold AND with each other -> kept (no consensus)
    kept, _ = run_verification(_ScriptedClient({"v1": '{"reasoning":"","answer":"A"}', "v2": '{"reasoning":"","answer":"C"}'}),
                               [_mcq()], {"c1": "text"}, models)
    assert len(kept) == 1
    # one verifier agrees with gold -> kept
    kept, _ = run_verification(_ScriptedClient({"v1": '{"reasoning":"","answer":"B"}', "v2": '{"reasoning":"","answer":"E"}'}),
                               [_mcq()], {"c1": "text"}, models)
    assert len(kept) == 1


def test_verification_open_answers():
    from arbench.pipeline.verify import run_verification

    q = {"qid": "o", "kind": "single_hop", "question": "Revenue?", "answer": "£9bn", "chunk_ids": ["c1"],
         "generator": "gen", "question_type": "numerical"}
    no = '{"reasoning":"","supported":false}'
    kept, st = run_verification(_ScriptedClient({"v1": no, "v2": no}), [dict(q)], {"c1": "text"}, ["gen", "v1", "v2"])
    assert kept == [] and st["dropped"] == {"answer_not_supported": 1}
    kept, _ = run_verification(_ScriptedClient({"v1": no, "v2": '{"reasoning":"","supported":true}'}), [dict(q)],
                               {"c1": "text"}, ["gen", "v1", "v2"])
    assert len(kept) == 1
