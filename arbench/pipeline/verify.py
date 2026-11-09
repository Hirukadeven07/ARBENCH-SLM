"""Stage 5b – cross-model verification of generated items (extension beyond YourBench).

Small generators sometimes mislabel the correct MCQ option or write a gold answer that
the evidence does not support – lexical filters cannot see that. Each item is re-checked
by the models that did NOT write it:

* MCQ: verifiers pick A-D, or E = "none of the options is supported".
  Drop only when every verifier rejects the gold option AND they agree with each other
  (all E, or all the same other letter) -> a consensus that the item is broken.
* Open-ended: verifiers judge whether the gold answer is supported by the evidence.
  Drop only when every verifier says "not supported".

Requiring unanimous rejection keeps hard-but-valid questions (a single verifier failing a
question is not enough to drop it) while removing clearly broken items.
"""

from __future__ import annotations

import logging
from collections import Counter, defaultdict

from arbench import prompts
from arbench.llm.ollama_client import ChatRequest, OllamaClient
from arbench.utils import extract_json, truncate_tokens

log = logging.getLogger("arbench.verify")


def _verifiers(q: dict, models: list[str]) -> list[str]:
    others = [m for m in models if m != q["generator"]]
    return others or models[:1]


def _request(q: dict, evidence: str) -> ChatRequest:
    ev = truncate_tokens(evidence, 1500)
    if q["kind"] == "multi_choice":
        a, b, c, d = q["choices"]
        return ChatRequest(
            messages=[
                {"role": "system", "content": prompts.VERIFY_MCQ_SYSTEM},
                {"role": "user", "content": prompts.VERIFY_MCQ_USER.format(evidence=ev, question=q["question"], a=a, b=b, c=c, d=d)},
            ],
            temperature=0.0,
            max_tokens=200,
            format=prompts.VERIFY_MCQ_SCHEMA,
        )
    return ChatRequest(
        messages=[
            {"role": "system", "content": prompts.VERIFY_QA_SYSTEM},
            {"role": "user", "content": prompts.VERIFY_QA_USER.format(evidence=ev, question=q["question"], answer=q["answer"])},
        ],
        temperature=0.0,
        max_tokens=200,
        format=prompts.VERIFY_QA_SCHEMA,
    )


def _parse(q: dict, text: str):
    try:
        obj = extract_json(text)
    except ValueError:
        return None
    if not isinstance(obj, dict):
        return None
    if q["kind"] == "multi_choice":
        ans = str(obj.get("answer", "")).strip().upper()[:1]
        return ans if ans in "ABCDE" and ans else None
    sup = obj.get("supported")
    if isinstance(sup, str):
        sup = sup.strip().lower() in ("true", "yes")
    return bool(sup) if sup is not None else None


def run_verification(
    client: OllamaClient, questions: list[dict], chunk_text: dict[str, str], models: list[str], concurrency: int = 1
) -> tuple[list[dict], dict]:
    work: dict[str, list[dict]] = defaultdict(list)
    for q in questions:
        q["verification"] = {}
        for v in _verifiers(q, models):
            work[v].append(q)

    for model, qs in work.items():
        reqs = [_request(q, "\n\n".join(chunk_text[c] for c in q["chunk_ids"] if c in chunk_text)) for q in qs]
        results = client.chat_many(model, reqs, concurrency, desc=f"verify ({model})")
        for q, res in zip(qs, results):
            q["verification"][model] = _parse(q, res.text) if res.ok else None
        if len(work) > 1:
            client.unload(model)

    kept, dropped = [], Counter()
    for q in questions:
        votes = [v for v in q["verification"].values() if v is not None]
        if not votes:  # verifiers failed to answer – keep (no evidence against the item)
            kept.append(q)
            continue
        if q["kind"] == "multi_choice":
            gold = chr(65 + q["gold_index"])
            rejected = all(v != gold for v in votes) and len(set(votes)) == 1
            if rejected:
                dropped["mcq_consensus_" + ("no_valid_option" if votes[0] == "E" else "other_option")] += 1
                continue
        else:
            if all(v is False for v in votes):
                dropped["answer_not_supported"] += 1
                continue
        kept.append(q)
    stats = {"verified": len(questions), "kept": len(kept), "dropped": dict(dropped)}
    log.info("Verification kept %d / %d items. Dropped: %s", len(kept), len(questions), dict(dropped) or "none")
    return kept, stats
