"""Stage 3 – hierarchical summary (YourBench `pipeline/summarization.py`).

The summary only orients the question generator (company name, period, themes);
it is never used as answer evidence or shown to models under evaluation.
"""

from __future__ import annotations

import logging

from arbench import prompts
from arbench.llm.ollama_client import ChatRequest, OllamaClient
from arbench.pipeline.chunk import Chunk
from arbench.utils import extract_json, truncate_tokens

log = logging.getLogger("arbench.summarize")


def _evenly_spaced(n_total: int, k: int) -> list[int]:
    if k >= n_total:
        return list(range(n_total))
    return sorted({round(i * (n_total - 1) / max(1, k - 1)) for i in range(k)})


def _parse_summary(text: str) -> str:
    try:
        obj = extract_json(text)
        if isinstance(obj, dict) and isinstance(obj.get("summary"), str):
            return obj["summary"].strip()
    except ValueError:
        pass
    return text.strip()


def summarize(client: OllamaClient, model: str, title: str, chunks: list[Chunk], max_chunks: int) -> str:
    # always include the opening chunks (highlights / chair letter) plus evenly spaced ones
    picks = _evenly_spaced(len(chunks), max_chunks)
    reqs = [
        ChatRequest(
            messages=[{"role": "user", "content": prompts.SUMMARIZE_PROMPT.format(text=chunks[i].text)}],
            temperature=0.2,
            max_tokens=400,
            format=prompts.SUMMARY_SCHEMA,
        )
        for i in picks
    ]
    results = client.chat_many(model, reqs, desc=f"summarise ({model})")
    parts = [_parse_summary(r.text) for r in results if r.ok and r.text.strip()]
    if not parts:
        log.warning("Summarisation failed – continuing with the document title only.")
        return title
    if len(parts) == 1:
        return parts[0]
    joined = truncate_tokens("\n\n".join(parts), 3000)
    res = client.chat(
        model,
        ChatRequest(
            messages=[{"role": "user", "content": prompts.COMBINE_SUMMARIES_PROMPT.format(summaries=joined)}],
            temperature=0.2,
            max_tokens=600,
            format=prompts.SUMMARY_SCHEMA,
        ),
    )
    return _parse_summary(res.text) if res.ok and res.text.strip() else "\n".join(parts)
