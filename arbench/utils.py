"""Shared helpers: token counting, JSON(L) IO, robust JSON extraction, number parsing."""

from __future__ import annotations

import hashlib
import json
import logging
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable

log = logging.getLogger("arbench")

# ----------------------------------------------------------------------------- tokens


@lru_cache(maxsize=1)
def _encoder():
    try:
        import tiktoken

        return tiktoken.get_encoding("cl100k_base")
    except Exception:  # offline / not installed -> heuristic
        log.debug("tiktoken unavailable – using heuristic token counts")
        return None


_WORDISH = re.compile(r"\w+|[^\w\s]", re.UNICODE)


def count_tokens(text: str) -> int:
    enc = _encoder()
    if enc is not None:
        return len(enc.encode(text, disallowed_special=()))
    return int(len(_WORDISH.findall(text)) * 1.15) + 1


def truncate_tokens(text: str, max_tokens: int) -> str:
    enc = _encoder()
    if enc is not None:
        toks = enc.encode(text, disallowed_special=())
        return text if len(toks) <= max_tokens else enc.decode(toks[:max_tokens]) + " …"
    words = text.split()
    limit = int(max_tokens / 1.3)
    return text if len(words) <= limit else " ".join(words[:limit]) + " …"


# ----------------------------------------------------------------------------- IO


def write_jsonl(path: str | Path, rows: Iterable[dict]) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    return n


def read_jsonl(path: str | Path) -> list[dict]:
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def write_json(path: str | Path, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, ensure_ascii=False, indent=2, default=str)


def stable_hash(*parts: Any, n: int = 12) -> str:
    return hashlib.sha256("\x1f".join(str(p) for p in parts).encode("utf-8")).hexdigest()[:n]


def slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("_")


# ----------------------------------------------------------------------------- JSON from LLMs

_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


def extract_json(text: str) -> Any:
    """Parse JSON from an LLM reply: strict first, then fenced, then first balanced block.

    YourBench rejects anything that is not exactly one JSON value; small models
    need a bit more tolerance, but we never 'repair' content – only locate it.
    """
    if text is None:
        raise ValueError("empty response")
    s = text.strip()
    try:
        return json.loads(s)
    except ValueError:
        pass
    for m in _FENCE.finditer(s):
        try:
            return json.loads(m.group(1).strip())
        except ValueError:
            continue
    for opener, closer in (("{", "}"), ("[", "]")):
        start = s.find(opener)
        while start != -1:
            depth, in_str, esc = 0, False, False
            for i in range(start, len(s)):
                ch = s[i]
                if in_str:
                    if esc:
                        esc = False
                    elif ch == "\\":
                        esc = True
                    elif ch == '"':
                        in_str = False
                    continue
                if ch == '"':
                    in_str = True
                elif ch == opener:
                    depth += 1
                elif ch == closer:
                    depth -= 1
                    if depth == 0:
                        try:
                            return json.loads(s[start : i + 1])
                        except ValueError:
                            break
            start = s.find(opener, start + 1)
    raise ValueError("no JSON value found in response")


# ----------------------------------------------------------------------------- numbers

_SCALE = {
    "thousand": 1e3,
    "k": 1e3,
    "mn": 1e6,
    "m": 1e6,
    "million": 1e6,
    "millions": 1e6,
    "bn": 1e9,
    "b": 1e9,
    "billion": 1e9,
    "billions": 1e9,
    "tn": 1e12,
    "trillion": 1e12,
    "crore": 1e7,
    "crores": 1e7,
    "lakh": 1e5,
    "lakhs": 1e5,
}
_NUM_RE = re.compile(
    r"(?<![\w.])(\(?-?\d{1,3}(?:[,\s]\d{3})+(?:\.\d+)?\)?|\(?-?\d+(?:\.\d+)?\)?)\s*"
    r"(%|per\s*cent|percent|thousand|million|millions|billion|billions|trillion|crores?|lakhs?|mn|bn|tn|k|m|b)?(?![\w])",
    re.IGNORECASE,
)


def extract_numbers(text: str, include_years: bool = True) -> list[float]:
    """Extract numeric values (handles 1,234.5 / (1,234) negatives / 12% / 3.4 bn).

    Returns both the raw value and, if a scale word is present, the scaled value
    is *also* added so that '1.2 billion' matches '1,200 million'.
    """
    values: list[float] = []
    for m in _NUM_RE.finditer(text or ""):
        raw, unit = m.group(1), (m.group(2) or "").lower().replace(" ", "")
        neg = raw.startswith("(") and raw.endswith(")")
        cleaned = raw.strip("()").replace(",", "").replace(" ", "")
        try:
            v = float(cleaned)
        except ValueError:
            continue
        if neg:
            v = -abs(v)
        if not include_years and unit == "" and v.is_integer() and 1900 <= v <= 2100:
            continue
        values.append(v)
        scale = _SCALE.get(unit)
        if scale:
            values.append(v * scale)
    return values


def numbers_match(a: float, b: float, rel_tol: float = 0.005) -> bool:
    if a == b:
        return True
    if abs(a) < 1e-9 or abs(b) < 1e-9:
        return abs(a - b) < 1e-9
    if abs(a - b) / max(abs(a), abs(b)) <= rel_tol:
        return True
    # tolerate rounding of the presented figure, e.g. 12.3 vs 12.34
    return round(a, 1) == round(b, 1) and abs(a) >= 1


def core_numbers(text: str) -> list[float]:
    """Numbers that carry information: drop list ordinals and 4-digit years only if other numbers exist."""
    vals = extract_numbers(text)
    non_year = [v for v in vals if not (v.is_integer() and 1900 <= v <= 2100)]
    return non_year or vals
