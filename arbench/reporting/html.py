"""Tiny Markdown -> self-contained HTML converter for the reports this project writes.

Supports exactly what report.py emits: headings, tables, lists, block quotes, images
(embedded as base64 so report.html is a single portable file), bold/italic/code.
"""

from __future__ import annotations

import base64
import html
import re
from pathlib import Path

CSS = """
:root{--bg:#fcfcfb;--panel:#ffffff;--text:#0b0b0b;--muted:#52514e;--line:#e4e3df;--accent:#2a78d6;--quote:#f4f3ef}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#1a1a19;--panel:#222221;--text:#f0efec;--muted:#c3c2b7;--line:#3a3a38;--accent:#3987e5;--quote:#2a2a28}}
:root[data-theme="dark"]{--bg:#1a1a19;--panel:#222221;--text:#f0efec;--muted:#c3c2b7;--line:#3a3a38;--accent:#3987e5;--quote:#2a2a28}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--text);font:15px/1.55 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif}
header{padding:20px 16px 0;max-width:1100px;margin:0 auto}
header h1{margin:0 0 4px;font-size:22px}
header p{margin:0;color:var(--muted);font-size:13px}
nav{display:flex;gap:6px;flex-wrap:wrap;max-width:1100px;margin:14px auto 0;padding:0 16px}
nav button{border:1px solid var(--line);background:var(--panel);color:var(--text);padding:7px 14px;border-radius:999px;cursor:pointer;font:inherit;font-size:13px}
nav button[aria-selected="true"]{background:var(--accent);border-color:var(--accent);color:#fff}
main{max-width:1100px;margin:0 auto;padding:8px 16px 48px}
section{display:none}section.active{display:block}
h1{font-size:24px;margin:24px 0 8px}h2{font-size:19px;margin:28px 0 8px;border-bottom:1px solid var(--line);padding-bottom:4px}h3{font-size:16px;margin:20px 0 6px}
.table-wrap{overflow-x:auto;margin:8px 0 14px}
table{border-collapse:collapse;font-size:13px;min-width:50%}
th,td{border-bottom:1px solid var(--line);padding:6px 10px;text-align:left;vertical-align:top}
th{color:var(--muted);font-weight:600}
td.num,th.num{text-align:right;font-variant-numeric:tabular-nums}
img{display:block;width:100%;max-width:760px;height:auto;border:1px solid var(--line);border-radius:8px;margin:8px 0;background:#fcfcfb}
blockquote{margin:10px 0;padding:8px 12px;background:var(--quote);border-left:3px solid var(--accent);border-radius:4px}
code{background:var(--quote);padding:1px 5px;border-radius:4px;font-size:12.5px}
ul{padding-left:20px}li{margin:3px 0}
"""

JS = """
const btns=[...document.querySelectorAll('nav button')];
function show(i){btns.forEach((b,j)=>b.setAttribute('aria-selected',i===j));document.querySelectorAll('section').forEach((s,j)=>s.classList.toggle('active',i===j));try{localStorage.setItem('arbench-tab',i)}catch(e){}}
btns.forEach((b,i)=>b.addEventListener('click',()=>show(i)));
let t=0;try{t=+localStorage.getItem('arbench-tab')||0}catch(e){}show(t<btns.length?t:0);
"""


def _inline(text: str) -> str:
    t = html.escape(text, quote=False)
    t = re.sub(r"`([^`]+)`", r"<code>\1</code>", t)
    t = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", t)
    t = re.sub(r"(?<![\w*])\*(?!\s)(.+?)(?<!\s)\*(?![\w*])", r"<em>\1</em>", t)
    t = re.sub(r"(?<![\w])_(?!\s)(.+?)(?<!\s)_(?![\w])", r"<em>\1</em>", t)
    return t


def _img(src: str, alt: str, base: Path) -> str:
    p = (base / src).resolve()
    if p.exists():
        data = base64.b64encode(p.read_bytes()).decode()
        return f'<img alt="{html.escape(alt)}" src="data:image/png;base64,{data}">'
    return f"<p><em>missing chart {html.escape(src)}</em></p>"


def md_to_html(md: str, base: Path) -> str:
    out: list[str] = []
    lines = md.splitlines()
    i = 0
    para: list[str] = []

    def flush_para():
        if para:
            out.append("<p>" + "<br>".join(_inline(x.rstrip()) for x in para) + "</p>")
            para.clear()

    while i < len(lines):
        line = lines[i]
        s = line.strip()
        if not s:
            flush_para()
            i += 1
            continue
        m = re.match(r"^(#{1,3})\s+(.*)", s)
        if m:
            flush_para()
            lvl = len(m.group(1))
            out.append(f"<h{lvl}>{_inline(m.group(2))}</h{lvl}>")
            i += 1
            continue
        m = re.match(r"^!\[(.*?)\]\((.*?)\)$", s)
        if m:
            flush_para()
            out.append(_img(m.group(2), m.group(1), base))
            i += 1
            continue
        if s.startswith("|"):
            flush_para()
            rows = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                rows.append(lines[i].strip())
                i += 1
            cells = [[c.strip() for c in r.strip("|").split("|")] for r in rows]
            aligns = []
            if len(cells) > 1 and all(re.fullmatch(r":?-+:?", c) for c in cells[1]):
                aligns = ["num" if c.endswith(":") and not c.startswith(":") else "" for c in cells[1]]
                header, body = cells[0], cells[2:]
            else:
                header, body = cells[0], cells[1:]
            h = "".join(f'<th class="{aligns[k] if k < len(aligns) else ""}">{_inline(c)}</th>' for k, c in enumerate(header))
            b = "".join(
                "<tr>" + "".join(f'<td class="{aligns[k] if k < len(aligns) else ""}">{_inline(c)}</td>' for k, c in enumerate(r)) + "</tr>"
                for r in body
            )
            out.append(f'<div class="table-wrap"><table><thead><tr>{h}</tr></thead><tbody>{b}</tbody></table></div>')
            continue
        if s.startswith("- "):
            flush_para()
            items = []
            while i < len(lines) and lines[i].strip().startswith("- "):
                items.append(f"<li>{_inline(lines[i].strip()[2:])}</li>")
                i += 1
            out.append("<ul>" + "".join(items) + "</ul>")
            continue
        if s.startswith(">"):
            flush_para()
            q = []
            while i < len(lines) and lines[i].strip().startswith(">"):
                q.append(_inline(lines[i].strip().lstrip(">").strip()))
                i += 1
            out.append("<blockquote>" + "<br>".join(q) + "</blockquote>")
            continue
        para.append(line)
        i += 1
    flush_para()
    return "\n".join(out)


def markdown_to_html_page(title: str, sections: list[tuple[str, str]], base: Path) -> str:
    nav = "".join(f'<button role="tab" aria-selected="false">{html.escape(name)}</button>' for name, _ in sections)
    body = "".join(f"<section>{md_to_html(md, base)}</section>" for _, md in sections)
    return (
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width,initial-scale=1'>"
        f"<title>{html.escape(title)}</title><style>{CSS}</style></head><body>"
        f"<header><h1>{html.escape(title)}</h1><p>Annual-report benchmark of small language models · "
        "YourBench-style generation + LightEval-style evaluation · 100% local (Ollama)</p></header>"
        f"<nav role='tablist'>{nav}</nav><main>{body}</main><script>{JS}</script></body></html>"
    )
