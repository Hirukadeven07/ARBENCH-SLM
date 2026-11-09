"""Stage 5 – question quality filtering.

Re-engineered from YourBench `citation_score_filtering.py` + `_remove_duplicate_questions`:

1. citation grounding   – fuzzy partial-ratio of each citation against the source chunk(s)
2. number grounding     – figures in the gold answer must appear in the source (no invented numbers)
3. hygiene              – no meta references ("according to the text"), sane lengths, valid MCQs
4. near-duplicate removal (token-set similarity)
"""

from __future__ import annotations

import logging
import re
from collections import Counter

from rapidfuzz import fuzz

from arbench.retrieval import content_overlap
from arbench.utils import core_numbers, extract_numbers, numbers_match

log = logging.getLogger("arbench.filter")

_META = re.compile(
    r"\b(according to|based on|in|from|within)\s+(the|this)\s+(text|passage|chunk|excerpt|section above|given text|"
    r"provided text|source)\b|\btext chunk\b",
    re.IGNORECASE,
)


def citation_score(citations: list[str], source: str, answer: str, alpha: float = 0.7, beta: float = 0.3) -> float:
    """YourBench-style: alpha * avg(citation vs chunk) + beta * avg(citation vs answer), in [0, 1].

    Without usable citations we fall back to lexical support of the answer by the source.
    """
    cits = [c for c in citations if len(c) >= 8]
    if not cits:
        return round(0.9 * content_overlap(answer, source), 3)
    src = source.lower()
    chunk_scores = [fuzz.partial_ratio(c.lower(), src) / 100 for c in cits]
    ans_scores = [fuzz.partial_ratio(c.lower(), answer.lower()) / 100 for c in cits]
    chunk = sum(chunk_scores) / len(cits)
    ans = sum(ans_scores) / len(cits)
    # The answer term rewards citations that actually support the answer; never let it
    # pull a perfectly grounded citation set below the threshold on its own.
    return round(max(chunk * 0.9, alpha * chunk + beta * ans), 3)


def number_grounding(answer: str, source: str) -> float:
    nums = [v for v in core_numbers(answer) if not (v.is_integer() and 1 <= v <= 10 and "%" not in answer)]
    if not nums:
        return 1.0
    src_nums = extract_numbers(source)
    hits = sum(1 for v in nums if any(numbers_match(v, s) for s in src_nums))
    return hits / len(nums)


def _reject_reason(q: dict, source: str, cfg) -> str | None:
    question, answer = q["question"], q["answer"]
    if _META.search(question):
        return "meta_reference"
    if len(question.split()) < 4:
        return "question_too_short"
    if len(answer.split()) > 150:
        return "answer_too_long"
    if fuzz.ratio(question.lower(), answer.lower()) > 85:
        return "answer_repeats_question"
    if q["kind"] == "multi_choice":
        norm = [re.sub(r"\W+", " ", c.lower()).strip() for c in q["choices"]]
        if len(set(norm)) < 4 or any(not c for c in norm):
            return "duplicate_choices"
    q["citation_score"] = citation_score(q["citations"], source, answer)
    q["number_grounding"] = round(number_grounding(answer, source), 3)
    if q["citation_score"] < cfg["min_citation_score"]:
        return "low_citation_score"
    if q["number_grounding"] < cfg["min_number_grounding"]:
        return "ungrounded_numbers"
    return None


def run_filtering(questions: list[dict], chunk_text: dict[str, str], cfg) -> tuple[list[dict], dict]:
    reasons: Counter[str] = Counter()
    kept: list[dict] = []
    for q in questions:
        source = "\n\n".join(chunk_text[c] for c in q["chunk_ids"] if c in chunk_text)
        reason = _reject_reason(q, source, cfg)
        if reason:
            reasons[reason] += 1
            continue
        kept.append(q)

    # near-duplicate removal: keep the better-grounded question
    kept.sort(key=lambda q: (-q["citation_score"], q["qid"]))
    unique: list[dict] = []
    for q in kept:
        if any(
            fuzz.token_set_ratio(q["question"].lower(), u["question"].lower()) >= cfg["dedup_similarity"] * 100
            for u in unique
        ):
            reasons["near_duplicate"] += 1
            continue
        unique.append(q)

    max_q = cfg.get("max_questions", 0) or 0
    if max_q and len(unique) > max_q:
        reasons["over_max_questions"] += len(unique) - max_q
        unique = unique[:max_q]

    unique.sort(key=lambda q: (q["kind"], q["doc_id"], q["chunk_ids"][0], q["qid"]))
    stats = {
        "input": len(questions),
        "kept": len(unique),
        "rejected": dict(reasons),
        "kept_by_kind": dict(Counter(q["kind"] for q in unique)),
        "kept_by_generator": dict(Counter(q["generator"] for q in unique)),
        "kept_by_type": dict(Counter(q["question_type"] for q in unique)),
    }
    log.info("Filtering kept %d / %d questions. Rejections: %s", len(unique), len(questions), dict(reasons) or "none")
    return unique, stats
