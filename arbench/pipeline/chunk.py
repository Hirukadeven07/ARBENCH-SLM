"""Stage 2 – chunking (re-engineered from YourBench `pipeline/chunking.py`).

YourBench slices raw token windows (which can cut sentences and tables in half).
For annual reports we use *structure-aware* packing instead:

* paragraphs / Markdown tables / headings are atomic blocks
* blocks are packed into ~target_tokens chunks with a token overlap
* each chunk remembers its pages and the nearest section heading
* a quality score filters tables of contents, cover pages, and legal boilerplate

Multi-hop groups (YourBench `multihop_chunks`) are sampled later during generation.
"""

from __future__ import annotations

import logging
import re
from dataclasses import asdict, dataclass, field

from arbench.pipeline.ingest import Document
from arbench.utils import count_tokens, extract_numbers

log = logging.getLogger("arbench.chunk")


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    index: int
    text: str
    pages: list[int] = field(default_factory=list)
    section: str = ""
    n_tokens: int = 0
    quality: float = 0.0
    n_numbers: int = 0

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def page_label(self) -> str:
        if not self.pages:
            return "?"
        a, b = min(self.pages), max(self.pages)
        return f"p. {a}" if a == b else f"pp. {a}-{b}"


_HEADING = re.compile(r"^(#{1,6}\s+.+|[A-Z][A-Z0-9 &,'’\-–]{4,80})$")
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9“\"(])")


def _blocks_from_page(text: str) -> list[str]:
    """Split a page into paragraphs, keeping Markdown tables together."""
    blocks: list[str] = []
    buf: list[str] = []
    in_table = False
    for line in text.splitlines():
        is_table = line.strip().startswith("|")
        if is_table != in_table and buf:
            blocks.append("\n".join(buf))
            buf = []
        in_table = is_table
        if not line.strip() and not in_table:
            if buf:
                blocks.append("\n".join(buf))
                buf = []
            continue
        buf.append(line)
    if buf:
        blocks.append("\n".join(buf))
    return [b.strip() for b in blocks if b.strip()]


def _split_oversized(block: str, max_tokens: int) -> list[str]:
    """Split a too-long block by sentences (prose) or by rows (tables)."""
    if block.lstrip().startswith("|"):
        units = block.splitlines()
        header = units[:2] if len(units) > 2 and set(units[1].replace("|", "").strip()) <= set("-: ") else []
        body = units[len(header) :]
        joiner = "\n"
    else:
        units = _SENT_SPLIT.split(block)
        header, body, joiner = [], units, " "
    pieces, cur, cur_tok = [], list(header), count_tokens(joiner.join(header)) if header else 0
    for u in body:
        t = count_tokens(u)
        if cur and cur_tok + t > max_tokens and len(cur) > len(header):
            pieces.append(joiner.join(cur))
            cur, cur_tok = list(header), count_tokens(joiner.join(header)) if header else 0
        if t > max_tokens:  # single giant sentence / row – hard split on words
            words = u.split()
            step = max(50, int(len(words) * max_tokens / max(t, 1)))
            for i in range(0, len(words), step):
                pieces.append(" ".join(words[i : i + step]))
            continue
        cur.append(u)
        cur_tok += t
    if len(cur) > len(header):
        pieces.append(joiner.join(cur))
    return pieces


def quality_score(text: str) -> float:
    """0-1 heuristic: informative prose/tables score high; TOCs, covers, and boilerplate score low."""
    words = re.findall(r"[A-Za-z]{2,}", text)
    if len(words) < 25:
        return 0.0
    lines = [ln for ln in text.splitlines() if ln.strip()]
    toc_lines = sum(1 for ln in lines if re.search(r"(\.{3,}|…{2,}|\s{2,})\s*\d{1,4}\s*$", ln))
    toc_ratio = toc_lines / max(1, len(lines))
    alpha_ratio = sum(len(w) for w in words) / max(1, len(re.sub(r"\s", "", text)))
    lower = text.lower()
    boiler = sum(
        lower.count(k)
        for k in ("forward-looking statement", "all rights reserved", "this page intentionally", "registered office")
    )
    score = 0.5 * min(1.0, alpha_ratio / 0.6) + 0.3 * min(1.0, len(words) / 120) + 0.2
    score -= 0.8 * toc_ratio + 0.15 * min(boiler, 3)
    return round(max(0.0, min(1.0, score)), 3)


def chunk_document(doc: Document, target_tokens: int, max_tokens: int, overlap_tokens: int, min_tokens: int) -> list[Chunk]:
    # 1. blocks with page + section provenance
    items: list[tuple[str, int, str, int]] = []  # text, page, section, tokens
    section = ""
    for page in doc.pages:
        for block in _blocks_from_page(page.text):
            first = block.splitlines()[0].strip()
            if _HEADING.match(first) and len(block.splitlines()) <= 2 and len(first) < 90:
                section = first.lstrip("#").strip().strip("*")
            t = count_tokens(block)
            if t > max_tokens:
                for piece in _split_oversized(block, max_tokens):
                    items.append((piece, page.number, section, count_tokens(piece)))
            else:
                items.append((block, page.number, section, t))

    # 2. greedy packing with overlap
    chunks: list[Chunk] = []
    cur: list[tuple[str, int, str, int]] = []
    cur_tok = 0

    def flush() -> None:
        nonlocal cur, cur_tok
        if not cur:
            return
        text = "\n\n".join(x[0] for x in cur)
        idx = len(chunks)
        sec = next((x[2] for x in cur if x[2]), "")
        chunks.append(
            Chunk(
                chunk_id=f"{doc.doc_id}_{idx:04d}",
                doc_id=doc.doc_id,
                index=idx,
                text=text,
                pages=sorted({x[1] for x in cur}),
                section=sec[:120],
                n_tokens=cur_tok,
                quality=quality_score(text),
                n_numbers=len(extract_numbers(text, include_years=False)),
            )
        )
        # overlap: carry trailing blocks totalling <= overlap_tokens
        carry, carry_tok = [], 0
        for x in reversed(cur):
            if carry_tok + x[3] > overlap_tokens:
                break
            carry.insert(0, x)
            carry_tok += x[3]
        if len(carry) == len(cur):  # never carry the whole chunk
            carry, carry_tok = [], 0
        cur, cur_tok = carry, carry_tok

    for item in items:
        if cur and cur_tok + item[3] > target_tokens:
            flush()
        cur.append(item)
        cur_tok += item[3]
    if cur and cur_tok > overlap_tokens:
        flush()

    # 3. merge a too-small tail into the previous chunk
    if len(chunks) >= 2 and chunks[-1].n_tokens < min_tokens:
        last = chunks.pop()
        prev = chunks[-1]
        prev.text += "\n\n" + last.text
        prev.pages = sorted(set(prev.pages) | set(last.pages))
        prev.n_tokens += last.n_tokens
        prev.quality = quality_score(prev.text)
        prev.n_numbers = len(extract_numbers(prev.text, include_years=False))
    return chunks


def run_chunking(docs: list[Document], cfg_chunk) -> list[Chunk]:
    all_chunks: list[Chunk] = []
    for doc in docs:
        chunks = chunk_document(
            doc,
            cfg_chunk["target_tokens"],
            cfg_chunk["max_tokens"],
            cfg_chunk["overlap_tokens"],
            cfg_chunk["min_tokens"],
        )
        kept = [c for c in chunks if c.quality >= cfg_chunk["min_quality"] and c.n_tokens >= cfg_chunk["min_tokens"]]
        log.info(
            "  %s: %d chunks (%d kept after quality filter, avg %d tokens)",
            doc.filename,
            len(chunks),
            len(kept),
            sum(c.n_tokens for c in kept) / max(1, len(kept)),
        )
        all_chunks.extend(kept)
    if not all_chunks:
        raise ValueError("Chunking produced no usable chunks – is the document text extractable?")
    return all_chunks
