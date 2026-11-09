"""Sample-level metrics (re-engineered from LightEval `metrics_sample.py` / `normalizations.py`).

Ported:   helm_normalizer, exact match, bag-of-words F1, ROUGE-L, letter extractive match (gpqa-style)
Added for annual reports:
  * numeric_match      – share of the gold figures reproduced in the answer (tolerant to formats/scales)
  * number_hallucination – share of figures in the answer that appear nowhere in the given context
  * abstention detection – for the unanswerable (hallucination) task and over-refusal rate
"""

from __future__ import annotations

import re
import string

from arbench.prompts import INSUFFICIENT
from arbench.utils import core_numbers, extract_numbers, numbers_match

# ----------------------------------------------------------------------------- normalisation (LightEval)


def helm_normalizer(text: str) -> str:
    """Lower text and remove punctuation, articles and extra whitespace (LightEval `helm_normalizer`)."""

    def remove_articles(t: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", t)

    def homogeneize_numbers(t: str) -> str:
        try:
            return str(float(t))
        except ValueError:
            return t

    def remove_punc(t: str) -> str:
        exclude = set(string.punctuation)
        return "".join(ch for ch in t if ch not in exclude)

    tokens = [
        " ".join(remove_articles(homogeneize_numbers(remove_punc(t.lower()))).split())
        for t in re.split(" |-", text.replace(",", ""))
    ]
    return " ".join(t for t in tokens if t).strip()


# ----------------------------------------------------------------------------- answer extraction

_ANSWER_TAG = re.compile(r"<answer>(.*?)(?:</answer>|$)", re.DOTALL | re.IGNORECASE)


def extract_answer(text: str) -> tuple[str, bool]:
    """Return (answer, format_ok). format_ok = the model used the requested <answer> tags."""
    if not text:
        return "", False
    m = _ANSWER_TAG.search(text)
    if m and m.group(1).strip():
        ans = re.sub(r"</answer\s*$", "", m.group(1).strip(), flags=re.IGNORECASE).strip()
        return ans, "</answer" in text.lower()
    return text.strip(), False


_ABSTAIN = re.compile(
    r"insufficient[_ ]context|not (?:mentioned|provided|stated|specified|available|included|found|disclosed|given)"
    r"|(?:do|does) not (?:contain|provide|mention|include|specify|state|say)|no (?:information|mention|data|details)"
    r"|cannot (?:be )?(?:determine|determined|answer|answered|find|found)|can't (?:determine|answer)"
    r"|unable to (?:determine|answer|find)|not possible to (?:determine|answer)|insufficient information",
    re.IGNORECASE,
)


def is_abstention(answer: str) -> bool:
    if not answer:
        return False
    if INSUFFICIENT.lower() in answer.lower():
        return True
    # only treat as abstention when the reply is short and dominated by a refusal phrase
    return len(answer.split()) <= 40 and bool(_ABSTAIN.search(answer))


# ----------------------------------------------------------------------------- lexical metrics


def exact_match(gold: str, pred: str) -> float:
    return float(helm_normalizer(gold) == helm_normalizer(pred))


def token_f1(gold: str, pred: str) -> float:
    """LightEval F1_score: f-measure over the *sets* of normalised words."""
    g, p = set(helm_normalizer(gold).split()), set(helm_normalizer(pred).split())
    if not g or not p:
        return 0.0
    inter = len(g & p)
    if inter == 0:
        return 0.0
    precision, recall = inter / len(p), inter / len(g)
    return 2 * precision * recall / (precision + recall)


def rouge_l(gold: str, pred: str) -> float:
    """ROUGE-L F-measure on normalised tokens (LCS)."""
    g, p = helm_normalizer(gold).split(), helm_normalizer(pred).split()
    if not g or not p:
        return 0.0
    if len(g) * len(p) > 250_000:  # guard against pathological lengths
        p = p[: max(1, 250_000 // len(g))]
    prev = [0] * (len(p) + 1)
    for gi in g:
        cur = [0] * (len(p) + 1)
        for j, pj in enumerate(p, 1):
            cur[j] = prev[j - 1] + 1 if gi == pj else max(prev[j], cur[j - 1])
        prev = cur
    lcs = prev[-1]
    if lcs == 0:
        return 0.0
    r, pr = lcs / len(g), lcs / len(p)
    return 2 * r * pr / (r + pr)


# ----------------------------------------------------------------------------- numeric metrics


def _informative_numbers(text: str) -> list[float]:
    return [v for v in core_numbers(text) if not (v.is_integer() and 0 <= v <= 10)]


def numeric_match(gold: str, pred: str) -> float | None:
    """Share of gold figures present in the prediction (None if the gold has no figures)."""
    g = _informative_numbers(gold)
    if not g:
        return None
    p = extract_numbers(pred)
    return sum(1 for v in g if any(numbers_match(v, x) for x in p)) / len(g)


def number_hallucination(pred: str, context: str, question: str = "") -> float | None:
    """Share of figures in the prediction that appear neither in the context nor in the question."""
    nums = _informative_numbers(pred)
    if not nums:
        return None
    pool = extract_numbers(context) + extract_numbers(question)
    unsupported = 0
    for v in nums:
        if any(numbers_match(v, x) for x in pool):
            continue
        # allow derived figures that are simple differences / ratios of two context numbers? too costly – count it
        unsupported += 1
    return unsupported / len(nums)


# ----------------------------------------------------------------------------- MCQ (LightEval extractive match)

_LETTER_PATTERNS = [
    re.compile(r"answer\s*(?:is|:)?\s*\(?\**\s*([A-D])\b", re.IGNORECASE),
    re.compile(r"^\s*\(?\**([A-D])\**\)?(?:[.):\s]|$)"),
    re.compile(r"\(([A-D])\)"),
    re.compile(r"\b(?:option|choice)\s*\(?([A-D])\b", re.IGNORECASE),
    re.compile(r"\b([A-D])\b"),
]


def extract_letter(text: str, choices: list[str]) -> int | None:
    if not text:
        return None
    t = text.strip()
    for pat in _LETTER_PATTERNS:
        m = pat.search(t)
        if m:
            idx = ord(m.group(1).upper()) - 65
            if 0 <= idx < len(choices):
                return idx
    # fall back to matching the choice text
    norm = helm_normalizer(t)
    hits = [i for i, c in enumerate(choices) if helm_normalizer(c) and helm_normalizer(c) in norm]
    return hits[0] if len(hits) == 1 else None


# ----------------------------------------------------------------------------- per-item scoring


def score_item(item: dict, raw_output: str) -> dict:
    task = item["task"]
    if task == "multi_choice":
        pred = extract_letter(raw_output, item["choices"])
        return {
            "prediction": raw_output.strip(),
            "pred_index": pred,
            "parsed": pred is not None,
            "accuracy": float(pred == item["gold_index"]),
        }

    answer, fmt = extract_answer(raw_output)
    abstained = is_abstention(answer)
    out = {"prediction": answer, "format_ok": fmt, "abstained": abstained}
    if task == "unanswerable":
        out["abstention"] = float(abstained)
        out["number_hallucination"] = None if abstained else number_hallucination(answer, item["context"], item["question"])
        return out
    out.update(
        {
            "exact_match": exact_match(item["gold"], answer),
            "token_f1": 0.0 if abstained else token_f1(item["gold"], answer),
            "rouge_l": 0.0 if abstained else rouge_l(item["gold"], answer),
            "numeric_match": None if numeric_match(item["gold"], "") is None else (0.0 if abstained else numeric_match(item["gold"], answer)),
            "number_hallucination": None
            if abstained or task == "closed_book"
            else number_hallucination(answer, item["context"], item["question"]),
        }
    )
    return out
