"""Dependency-free Okapi BM25 over report chunks (used for RAG-QA, multi-hop pairing and distractors)."""

from __future__ import annotations

import math
import re
from collections import Counter

STOPWORDS = set(
    """a an the and or of to in on for by with at from as is are was were be been being this that these those it its
    which who whom what when where why how than then there their our we us you your they them he she his her not no
    do does did done has have had will would shall should can could may might must also into over under about after
    before during per such other more most less least very year years company group annual report""".split()
)
_TOKEN = re.compile(r"[a-z0-9]+(?:[.,][0-9]+)*")


def tokenize(text: str) -> list[str]:
    toks = _TOKEN.findall(text.lower())
    return [t.replace(",", "") for t in toks if t not in STOPWORDS and (len(t) > 1 or t.isdigit())]


class BM25:
    def __init__(self, docs: list[str], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.docs_tokens = [tokenize(d) for d in docs]
        self.doc_len = [len(t) for t in self.docs_tokens]
        self.avgdl = sum(self.doc_len) / max(1, len(self.doc_len))
        self.tf = [Counter(t) for t in self.docs_tokens]
        df: Counter[str] = Counter()
        for toks in self.docs_tokens:
            df.update(set(toks))
        n = len(docs)
        self.idf = {w: math.log(1 + (n - f + 0.5) / (f + 0.5)) for w, f in df.items()}

    def scores(self, query: str) -> list[float]:
        q = tokenize(query)
        out = []
        for tf, dl in zip(self.tf, self.doc_len):
            s = 0.0
            for w in q:
                f = tf.get(w)
                if not f:
                    continue
                s += self.idf.get(w, 0.0) * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * dl / self.avgdl))
            out.append(s)
        return out

    def top_k(self, query: str, k: int) -> list[int]:
        sc = self.scores(query)
        return sorted(range(len(sc)), key=lambda i: (-sc[i], i))[:k]


def content_overlap(a: str, b: str) -> float:
    """Share of content tokens of `a` that occur in `b`."""
    ta = set(tokenize(a))
    if not ta:
        return 0.0
    tb = set(tokenize(b))
    return len(ta & tb) / len(ta)
