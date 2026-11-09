"""Shared fixtures: a synthetic annual report and a fake Ollama client (no network, deterministic)."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from arbench.llm.ollama_client import ChatResult  # noqa: E402
from arbench.prompts import INSUFFICIENT  # noqa: E402

SAMPLE_PAGES = [
    "Northwind Logistics PLC\nAnnual Report 2024\n\nDelivering resilient supply chains",
    """# Chair's Statement

In 2024 Northwind Logistics delivered a resilient performance despite softer freight volumes in Europe.
Group revenue increased by 8.4% to £2,415.6 million, compared with £2,228.4 million in 2023.
Underlying operating profit rose to £312.9 million, and the Board recommends a final dividend of 14.2 pence per share,
bringing the total dividend for the year to 21.0 pence. Our strategy, Network 2027, focuses on automation,
cross-border e-commerce and decarbonising our fleet.""",
    """# Financial Review

Revenue by segment: Contract Logistics generated £1,320.5 million (2023: £1,190.2 million), Freight Forwarding
£745.8 million (2023: £781.0 million) and E-commerce Fulfilment £349.3 million (2023: £257.2 million).
The decline in Freight Forwarding reflected lower ocean freight rates, while E-commerce Fulfilment grew 35.8%
following the opening of two automated hubs in Rotterdam and Leipzig. Net debt fell to £498.0 million and leverage
was 1.1 times EBITDA. Capital expenditure was £188.4 million, of which 40% related to automation projects.""",
    """# Principal Risks

The principal risks identified by the Board are cyber security, driver availability, fuel price volatility and
customer concentration. A ransomware attack on a peer in 2024 led the Group to invest £12.5 million in network
segmentation and staff training. Driver turnover reduced to 18% from 24% after the introduction of a new pay
framework. The largest customer accounted for 9% of revenue.""",
    """# Sustainability

Scope 1 and 2 emissions decreased by 11% to 412,000 tonnes CO2e. The Group now operates 640 electric vans and
75 hydrogen-ready heavy goods vehicles. Northwind targets a 50% reduction in Scope 1 and 2 emissions by 2030 against
a 2019 baseline and net zero by 2045. Employee engagement reached 78%, the highest level recorded.""",
    """# Governance

The Board comprises nine directors, six of whom are independent non-executive directors. Sarah Okafor became Chief
Executive Officer on 1 March 2024, succeeding David Lindqvist, who retired after twelve years. The Audit Committee
met six times during the year and oversaw the tender of the external audit, which will move to Crowe LLP from 2026.""",
]


def make_sample_pdf(path: Path) -> Path:
    import pymupdf as fitz

    doc = fitz.open()
    for i, text in enumerate(SAMPLE_PAGES):
        page = doc.new_page()
        page.insert_textbox(fitz.Rect(50, 50, 545, 780), text.replace("# ", ""), fontsize=10)
        page.insert_text((290, 815), str(i + 1), fontsize=8)
    doc.save(path)
    return path


@pytest.fixture
def sample_report_dir(tmp_path: Path) -> Path:
    d = tmp_path / "data" / "annual_report"
    d.mkdir(parents=True)
    (d / "northwind_annual_report_2024.md").write_text("\f".join(SAMPLE_PAGES), encoding="utf-8")
    return d


# ------------------------------------------------------------------------------- fake LLM


def _first_sentence_with_number(text: str) -> str:
    for s in re.split(r"(?<=[.])\s+", text):
        if re.search(r"\d", s) and len(s) > 40:
            return s.strip()
    return text.strip().split("\n")[0]


class FakeClient:
    """Imitates OllamaClient: valid structured JSON for generation/judging, simple QA answers."""

    def __init__(self, *_, **__):
        self.calls = 0

    @classmethod
    def from_config(cls, cfg, cache):
        return cls()

    def ensure_server(self, *a, **k):
        pass

    def ensure_models(self, names, *a, **k):
        return {n: {"family": "fake", "parameter_size": "1B", "quantization_level": "Q4", "size_gb": 1.0, "digest": "abc"} for n in names}

    def unload(self, name):
        pass

    def version(self):
        return "fake"

    def chat(self, model, req):
        self.calls += 1
        system = req.messages[0]["content"] if req.messages[0]["role"] == "system" else ""
        user = req.messages[-1]["content"]
        text = self._respond(model, system, user, req)
        return ChatResult(text=text, model=model, prompt_tokens=100, completion_tokens=20, total_s=1.0,
                          eval_s=0.5, prompt_eval_s=0.2, done_reason="stop")

    def chat_many(self, model, reqs, concurrency=1, desc="", on_result=None):
        return [self.chat(model, r) for r in reqs]

    def _respond(self, model, system, user, req):
        if "summary" in json.dumps(req.format or {}) and "questions" not in json.dumps(req.format or {}):
            if "verdict" not in json.dumps(req.format or {}):
                return json.dumps({"summary": "Northwind Logistics PLC annual report 2024 summary."})
        if "verdict" in json.dumps(req.format or {}):
            ans = re.search(r"<model_answer>\s*(.*?)\s*</model_answer>", user, re.S).group(1)
            gold = re.search(r"<gold_answer>\s*(.*?)\s*</gold_answer>", user, re.S).group(1)
            v = "correct" if ans[:30] == gold[:30] else ("partially_correct" if model.endswith("b") else "incorrect")
            return json.dumps({"reasoning": "compared", "verdict": v})
        if system.startswith("You check multiple-choice"):
            return json.dumps({"reasoning": "checked", "answer": "A"})
        if system.startswith("You check question-answer"):
            return json.dumps({"reasoning": "checked", "supported": True})
        if "multiple-choice evaluation questions" in system:
            chunk = re.search(r"<text_chunk[^>]*>\s*(.*?)\s*</text_chunk>", user, re.S).group(1)
            s = _first_sentence_with_number(chunk)
            return json.dumps({"questions": [{
                "question_type": "factual", "question": f"In Northwind's 2024 annual report, which statement is correct about: {s[:60]}?",
                "citations": [s[:80]], "choices": [s[:70], "Revenue fell sharply", "No dividend was paid", "The CEO resigned"],
                "answer": "A", "estimated_difficulty": 3}]})
        if "evaluation questions" in system:
            chunks = re.findall(r"<text_chunk[^>]*>\s*(.*?)\s*</text_chunk>", user, re.S)
            out = []
            for k, ch in enumerate(chunks[:2]):
                s = _first_sentence_with_number(ch)
                out.append({
                    "question_type": "numerical",
                    "question": f"What did Northwind Logistics report in 2024 regarding item {k} of {ch[:25]!r}?",
                    "citations": [s[:90]], "answer": s, "estimated_difficulty": 4 + k})
            return json.dumps({"questions": out})
        if "Choices:" in user:
            return "Answer: A" if model.startswith("a") else "B"
        # QA evaluation
        excerpt = re.search(r"<report_excerpts>\s*(.*?)\s*</report_excerpts>", user, re.S)
        if not excerpt:
            return f"<answer>{INSUFFICIENT}</answer>"
        body = excerpt.group(1).split("\n", 1)[-1]
        return f"<answer>{_first_sentence_with_number(body)}</answer>"
