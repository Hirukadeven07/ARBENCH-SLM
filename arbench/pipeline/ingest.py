"""Stage 1 – ingestion (re-engineered from YourBench `pipeline/ingestion.py`).

Converts every annual report in the input folder to clean Markdown, keeping page
boundaries so each chunk/question can cite the page numbers it came from.

PDF  -> pymupdf4llm (Markdown incl. tables) with a plain PyMuPDF fallback
DOCX -> python-docx (paragraphs + tables)
HTML -> BeautifulSoup text
MD / TXT -> as-is
"""

from __future__ import annotations

import logging
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from arbench.utils import stable_hash

log = logging.getLogger("arbench.ingest")

ALT_INPUT_DIRS = [
    "data/annual_report",
    "data/annual_reports",
    "data/anual_report",
    "data/anual report",
    "data/annual report",
    "data anual report",
    "data annual report",
    "data_annual_report",
    "data",
]


@dataclass
class Page:
    number: int  # 1-based
    text: str


@dataclass
class Document:
    doc_id: str
    filename: str
    title: str
    pages: list[Page] = field(default_factory=list)
    meta: dict = field(default_factory=dict)

    @property
    def text(self) -> str:
        return "\n\n".join(p.text for p in self.pages)


# ----------------------------------------------------------------------------- discovery


def discover_inputs(input_dir: Path, extensions: list[str], project_root: Path) -> tuple[Path, list[Path]]:
    """Find report files; tolerate the common folder-name variants (e.g. 'data anual report')."""
    exts = {e.lower() for e in extensions}
    candidates = [input_dir] + [project_root / d for d in ALT_INPUT_DIRS]
    seen = set()
    for folder in candidates:
        folder = folder.resolve()
        if folder in seen or not folder.is_dir():
            continue
        seen.add(folder)
        files = sorted(
            p
            for p in folder.rglob("*")
            if p.is_file()
            and p.suffix.lower() in exts
            and not p.name.startswith(("~$", "."))
            and p.name.lower() != "readme.md"
            and "outputs" not in p.parts
        )
        if files:
            if folder != input_dir.resolve():
                log.info("Using annual reports found in %s", folder)
            return folder, files
    raise FileNotFoundError(
        f"No annual report found. Put a PDF/DOCX/HTML/MD/TXT file in '{input_dir}' and run again."
    )


# ----------------------------------------------------------------------------- converters


def _pdf_pages(path: Path, engine: str) -> list[Page]:
    try:
        import pymupdf as fitz  # PyMuPDF >= 1.24
    except ImportError:  # pragma: no cover
        import fitz

    pages: list[Page] = []
    if engine in ("auto", "pymupdf4llm"):
        try:
            import pymupdf4llm

            chunks = pymupdf4llm.to_markdown(str(path), page_chunks=True, show_progress=False)
            for i, ch in enumerate(chunks):
                num = (ch.get("metadata") or {}).get("page", i + 1)
                pages.append(Page(number=int(num) if num else i + 1, text=ch.get("text", "")))
            if sum(len(p.text.strip()) for p in pages) > 0:
                return pages
        except Exception as exc:  # pragma: no cover - depends on PDF
            log.warning("pymupdf4llm failed on %s (%s) – falling back to plain PyMuPDF", path.name, exc)
    with fitz.open(path) as doc:
        return [Page(number=i + 1, text=page.get_text("text")) for i, page in enumerate(doc)]


def _docx_pages(path: Path) -> list[Page]:
    import docx

    d = docx.Document(str(path))
    parts: list[str] = []
    for p in d.paragraphs:
        t = p.text.strip()
        if not t:
            continue
        style = (p.style.name or "").lower() if p.style is not None else ""
        if style.startswith("heading"):
            level = re.findall(r"\d+", style)
            parts.append("#" * int(level[0] if level else 2) + " " + t)
        else:
            parts.append(t)
    for table in d.tables:
        rows = []
        for r in table.rows:
            rows.append("| " + " | ".join(c.text.strip().replace("\n", " ") for c in r.cells) + " |")
        if rows:
            header_sep = "|" + "---|" * len(table.rows[0].cells)
            parts.append("\n".join([rows[0], header_sep, *rows[1:]]))
    # DOCX has no fixed pages: emit pseudo-pages of ~40 blocks
    pages, step = [], 40
    for i in range(0, len(parts), step):
        pages.append(Page(number=i // step + 1, text="\n\n".join(parts[i : i + step])))
    return pages


def _html_pages(path: Path) -> list[Page]:
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(path.read_text(encoding="utf-8", errors="ignore"), "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header", "noscript"]):
        tag.decompose()
    text = soup.get_text("\n")
    return [Page(number=1, text=text)]


def _text_pages(path: Path) -> list[Page]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    # honour explicit form-feed page breaks if present
    parts = text.split("\f") if "\f" in text else [text]
    return [Page(number=i + 1, text=t) for i, t in enumerate(parts)]


# ----------------------------------------------------------------------------- cleaning


_PAGE_NUM = re.compile(r"^\s*(page\s*)?\d{1,4}(\s*(of|/)\s*\d{1,4})?\s*$", re.IGNORECASE)


def _norm_line(line: str) -> str:
    return re.sub(r"\d+", "#", line.strip().lower())


def clean_pages(pages: list[Page], strip_headers_footers: bool = True) -> list[Page]:
    """Remove running headers/footers, page numbers, and layout noise."""
    if strip_headers_footers and len(pages) >= 4:
        counter: Counter[str] = Counter()
        for p in pages:
            lines = [ln for ln in p.text.splitlines() if ln.strip()]
            edge = lines[:3] + lines[-3:]
            counter.update({_norm_line(ln) for ln in edge if len(ln.strip()) < 120})
        threshold = max(3, int(0.4 * len(pages)))
        repeated = {ln for ln, c in counter.items() if c >= threshold and ln}
    else:
        repeated = set()

    cleaned: list[Page] = []
    for p in pages:
        out_lines = []
        for ln in p.text.splitlines():
            s = ln.rstrip()
            if _PAGE_NUM.match(s):
                continue
            if repeated and _norm_line(s) in repeated:
                continue
            out_lines.append(s)
        text = "\n".join(out_lines)
        text = re.sub(r"[ \t]+\n", "\n", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)  # de-hyphenate line breaks
        text = text.replace(" ", " ").replace("ﬁ", "fi").replace("ﬂ", "fl")
        cleaned.append(Page(number=p.number, text=text.strip()))
    return cleaned


def _guess_title(path: Path, pages: list[Page]) -> str:
    for p in pages[:3]:
        for ln in p.text.splitlines():
            s = ln.strip().lstrip("#").strip().strip("*").strip()
            if 8 <= len(s) <= 120 and re.search(r"[A-Za-z]{3}", s) and "annual" in s.lower():
                return s
    return re.sub(r"[_-]+", " ", path.stem).strip()


def ingest_file(path: Path, root: Path, cfg_ing) -> Document:
    ext = path.suffix.lower()
    if ext == ".pdf":
        pages = _pdf_pages(path, cfg_ing.get("pdf_engine", "auto"))
    elif ext == ".docx":
        pages = _docx_pages(path)
    elif ext in (".html", ".htm"):
        pages = _html_pages(path)
    else:
        pages = _text_pages(path)
    raw_chars = sum(len(p.text) for p in pages)
    pages = clean_pages(pages, cfg_ing.get("strip_headers_footers", True))
    pages = [p for p in pages if p.text.strip()]
    if not pages:
        raise ValueError(
            f"{path.name}: no extractable text. If it is a scanned PDF, run OCR first "
            "(e.g. `ocrmypdf in.pdf out.pdf`) and place the OCR'd file in the input folder."
        )
    n_pages = max(p.number for p in pages)
    avg = raw_chars / max(1, n_pages)
    if ext == ".pdf" and avg < cfg_ing.get("min_chars_per_page_warning", 200):
        log.warning(
            "%s has only ~%d characters/page – it may be scanned. Consider OCR (ocrmypdf) for better results.",
            path.name,
            avg,
        )
    rel = path.relative_to(root).as_posix()
    text_all = "\n".join(p.text for p in pages)
    return Document(
        doc_id=stable_hash(rel, len(text_all), text_all[:2000], n=16),
        filename=rel,
        title=_guess_title(path, pages),
        pages=pages,
        meta={"pages": n_pages, "chars": len(text_all), "bytes": path.stat().st_size, "type": ext.lstrip(".")},
    )


def run_ingestion(input_dir: Path, cfg_ing, project_root: Path, out_dir: Path) -> list[Document]:
    root, files = discover_inputs(input_dir, cfg_ing["extensions"], project_root)
    docs = []
    for f in files:
        log.info("Extracting %s …", f.name)
        doc = ingest_file(f, root, cfg_ing)
        docs.append(doc)
        md_path = out_dir / f"{Path(doc.filename).stem}.md"
        md_path.parent.mkdir(parents=True, exist_ok=True)
        with open(md_path, "w", encoding="utf-8") as fh:
            fh.write(f"# {doc.title}\n\n")
            for p in doc.pages:
                fh.write(f"\n\n<!-- page {p.number} -->\n\n{p.text}")
        log.info(
            "  %s: %d pages, %s characters, title='%s'", f.name, doc.meta["pages"], f"{doc.meta['chars']:,}", doc.title
        )
    return docs
