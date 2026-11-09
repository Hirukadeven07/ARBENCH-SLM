"""Minimal, robust Ollama client.

Replaces YourBench's `inference_core` (HF Inference / OpenAI) and LightEval's
model backends (LiteLLM / vLLM / TGI) with a single local backend:

* starts `ollama serve` if needed and pulls missing models
* structured outputs (JSON schema) via Ollama's `format` field
* disables "thinking" for models that support it, transparently falls back otherwise
* persistent request cache (resumable runs), retries with back-off
* records latency / token statistics for efficiency metrics
"""

from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

import requests
from tqdm import tqdm

from arbench.llm.cache import LLMCache, request_key

log = logging.getLogger("arbench.ollama")

THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


class OllamaError(RuntimeError):
    pass


@dataclass
class ChatResult:
    text: str = ""
    thinking: str = ""
    model: str = ""
    error: str | None = None
    cached: bool = False
    done_reason: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_s: float = 0.0
    load_s: float = 0.0
    prompt_eval_s: float = 0.0
    eval_s: float = 0.0
    wall_s: float = 0.0
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.error is None

    @property
    def tokens_per_s(self) -> float:
        return self.completion_tokens / self.eval_s if self.eval_s > 0 else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "thinking": self.thinking,
            "model": self.model,
            "error": self.error,
            "done_reason": self.done_reason,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_s": self.total_s,
            "load_s": self.load_s,
            "prompt_eval_s": self.prompt_eval_s,
            "eval_s": self.eval_s,
            "wall_s": self.wall_s,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any], cached: bool = False) -> "ChatResult":
        known = {k: d.get(k) for k in cls.__dataclass_fields__ if k in d and k not in ("cached", "meta")}
        return cls(**known, cached=cached)


@dataclass
class ChatRequest:
    messages: list[dict[str, str]]
    temperature: float = 0.0
    max_tokens: int = 512
    format: dict | str | None = None
    tag: str = ""


class OllamaClient:
    def __init__(
        self,
        host: str = "http://127.0.0.1:11434",
        cache: LLMCache | None = None,
        timeout_s: float = 900,
        max_retries: int = 3,
        keep_alive: str = "30m",
        num_ctx: int = 8192,
        seed: int = 42,
        think: bool = False,
    ):
        self.host = host.rstrip("/")
        self.cache = cache or LLMCache(None)
        self.timeout_s = timeout_s
        self.max_retries = max_retries
        self.keep_alive = keep_alive
        self.num_ctx = num_ctx
        self.seed = seed
        self.think = think
        self._no_think_models: set[str] = set()
        self._digests: dict[str, str] = {}
        self._session = requests.Session()
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ server
    @classmethod
    def from_config(cls, cfg, cache: LLMCache) -> "OllamaClient":
        o = cfg["ollama"]
        host = os.environ.get("OLLAMA_HOST_URL") or o["host"]
        return cls(
            host=host,
            cache=cache,
            timeout_s=o["request_timeout_s"],
            max_retries=o["max_retries"],
            keep_alive=o["keep_alive"],
            num_ctx=o["num_ctx"],
            seed=cfg["seed"],
            think=cfg["evaluation"].get("think", False),
        )

    def is_up(self) -> bool:
        try:
            return self._session.get(f"{self.host}/api/version", timeout=5).ok
        except requests.RequestException:
            return False

    def version(self) -> str:
        try:
            return self._session.get(f"{self.host}/api/version", timeout=5).json().get("version", "?")
        except Exception:  # pragma: no cover
            return "?"

    def ensure_server(self, auto_start: bool = True, wait_s: int = 60) -> None:
        if self.is_up():
            return
        if not auto_start:
            raise OllamaError(f"Ollama is not reachable at {self.host}. Start it with `ollama serve`.")
        exe = shutil.which("ollama")
        if not exe:
            raise OllamaError(
                "Ollama is not installed or not on PATH. Install it from https://ollama.com/download "
                "and re-run this command."
            )
        log.info("Ollama is not running – starting `ollama serve` in the background …")
        kwargs: dict[str, Any] = {"stdout": subprocess.DEVNULL, "stderr": subprocess.DEVNULL}
        if sys.platform == "win32":
            kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS
        else:
            kwargs["start_new_session"] = True
        subprocess.Popen([exe, "serve"], **kwargs)
        deadline = time.time() + wait_s
        while time.time() < deadline:
            if self.is_up():
                log.info("Ollama server is up (%s).", self.version())
                return
            time.sleep(1)
        raise OllamaError(f"Started `ollama serve` but {self.host} did not respond within {wait_s}s.")

    def list_models(self) -> dict[str, dict]:
        r = self._session.get(f"{self.host}/api/tags", timeout=30)
        r.raise_for_status()
        out = {}
        for m in r.json().get("models", []):
            out[m["name"]] = m
            if m["name"].endswith(":latest"):
                out[m["name"][: -len(":latest")]] = m
        return out

    def pull(self, name: str, stall_s: int = 180, attempts: int = 6) -> None:
        """Pull a model with progress bar; restarts the download if it stalls (resumes partial data)."""
        import json as _json

        log.info("Pulling model %s (first run only) …", name)
        for attempt in range(1, attempts + 1):
            # Ollama interleaves progress lines for several layers -> track each digest separately
            layers: dict[str, list[int]] = {}
            bar, last_done, last_change, stalled = None, -1, time.time(), False
            try:
                with self._session.post(
                    f"{self.host}/api/pull", json={"model": name, "stream": True}, stream=True, timeout=(15, stall_s)
                ) as r:
                    r.raise_for_status()
                    for line in r.iter_lines():
                        if not line:
                            continue
                        msg = _json.loads(line)
                        if msg.get("error"):
                            raise OllamaError(f"pull {name}: {msg['error']}")
                        if msg.get("status") == "success":
                            break
                        total, done = msg.get("total"), msg.get("completed")
                        if total and done is not None:
                            layers[msg.get("digest", "?")] = [int(done), int(total)]
                            agg_done = sum(v[0] for v in layers.values())
                            agg_total = sum(v[1] for v in layers.values())
                            if bar is None:
                                bar = tqdm(total=agg_total, unit="B", unit_scale=True, desc=f"pull {name}", leave=False)
                            bar.total = agg_total
                            bar.n = agg_done
                            bar.refresh()
                            if agg_done != last_done:
                                last_done, last_change = agg_done, time.time()
                        if time.time() - last_change > stall_s:
                            stalled = True
                            break
            except requests.RequestException as exc:  # read timeout = no data at all
                stalled = True
                log.debug("pull %s: %s", name, exc)
            finally:
                if bar:
                    bar.close()
            if not stalled:
                log.info("Pulled %s.", name)
                return
            log.warning("Download of %s stalled – restarting (attempt %d/%d, progress is kept) …", name, attempt, attempts)
        raise OllamaError(f"Could not download {name}. Check your connection and run `ollama pull {name}`.")

    def ensure_models(self, names: Sequence[str], auto_pull: bool = True) -> dict[str, dict]:
        available = self.list_models()
        missing = [n for n in names if n not in available]
        if missing and not auto_pull:
            raise OllamaError(f"Models not available locally: {missing}. Run `ollama pull <name>`.")
        for name in missing:
            self.pull(name)
        if missing:
            available = self.list_models()
        info = {}
        for name in names:
            m = available.get(name)
            if not m:
                raise OllamaError(f"Model {name} is still unavailable after pulling.")
            self._digests[name] = m.get("digest", "")
            info[name] = {
                "digest": m.get("digest", "")[:12],
                "size_gb": round(m.get("size", 0) / 1e9, 2),
                **(m.get("details") or {}),
            }
        return info

    def unload(self, name: str) -> None:
        """Free RAM/VRAM before loading the next model (important on 16 GB laptops)."""
        try:
            self._session.post(f"{self.host}/api/generate", json={"model": name, "keep_alive": 0}, timeout=60)
        except requests.RequestException:
            pass

    # -------------------------------------------------------------------- chat
    def _payload(self, model: str, req: ChatRequest, think: bool | None) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": model,
            "messages": req.messages,
            "stream": False,
            "keep_alive": self.keep_alive,
            "options": {
                "temperature": req.temperature,
                "num_predict": req.max_tokens,
                "num_ctx": self.num_ctx,
                "seed": self.seed,
            },
        }
        if req.format is not None:
            payload["format"] = req.format
        if think is not None:
            payload["think"] = think
        return payload

    def chat(self, model: str, req: ChatRequest) -> ChatResult:
        think = None if model in self._no_think_models else self.think
        key = request_key(
            {
                "model": model,
                "digest": self._digests.get(model, ""),
                "messages": req.messages,
                "temperature": req.temperature,
                "max_tokens": req.max_tokens,
                "num_ctx": self.num_ctx,
                "seed": self.seed,
                "format": req.format,
                "think": self.think,
            }
        )
        hit = self.cache.get(key)
        if hit is not None:
            return ChatResult.from_dict(hit, cached=True)

        last_err = ""
        for attempt in range(1, self.max_retries + 2):
            t0 = time.perf_counter()
            try:
                r = self._session.post(
                    f"{self.host}/api/chat", json=self._payload(model, req, think), timeout=self.timeout_s
                )
                if r.status_code == 400 and think is not None and "think" in r.text.lower():
                    # model does not support the `think` flag – remember and retry without it
                    with self._lock:
                        self._no_think_models.add(model)
                    think = None
                    continue
                if r.status_code >= 400:
                    raise OllamaError(f"HTTP {r.status_code}: {r.text[:300]}")
                data = r.json()
                msg = data.get("message") or {}
                content = msg.get("content") or ""
                thinking = msg.get("thinking") or ""
                inline = THINK_RE.findall(content)
                if inline:
                    thinking = (thinking + "\n" + "\n".join(inline)).strip()
                    content = THINK_RE.sub("", content)
                # unterminated <think> (generation cut off while thinking)
                if "<think>" in content.lower() and "</think>" not in content.lower():
                    thinking, content = content, ""
                res = ChatResult(
                    text=content.strip(),
                    thinking=thinking,
                    model=model,
                    done_reason=data.get("done_reason", ""),
                    prompt_tokens=int(data.get("prompt_eval_count") or 0),
                    completion_tokens=int(data.get("eval_count") or 0),
                    total_s=(data.get("total_duration") or 0) / 1e9,
                    load_s=(data.get("load_duration") or 0) / 1e9,
                    prompt_eval_s=(data.get("prompt_eval_duration") or 0) / 1e9,
                    eval_s=(data.get("eval_duration") or 0) / 1e9,
                    wall_s=time.perf_counter() - t0,
                )
                self.cache.put(key, model, res.to_dict())
                return res
            except (requests.RequestException, OllamaError, ValueError) as exc:
                last_err = f"{type(exc).__name__}: {exc}"
                if attempt <= self.max_retries:
                    wait = min(60, 5 * attempt)
                    log.warning("%s call failed (attempt %d): %s – retrying in %ds", model, attempt, last_err, wait)
                    time.sleep(wait)
        return ChatResult(model=model, error=last_err or "unknown error")

    def chat_many(
        self,
        model: str,
        requests_: Sequence[ChatRequest],
        concurrency: int = 1,
        desc: str = "",
        on_result: Callable[[int, ChatResult], None] | None = None,
    ) -> list[ChatResult]:
        """Run many requests against one model (keeps the model loaded) preserving order."""
        results: list[ChatResult | None] = [None] * len(requests_)
        bar = tqdm(total=len(requests_), desc=desc or model, unit="call", dynamic_ncols=True)

        def run(i: int) -> None:
            res = self.chat(model, requests_[i])
            results[i] = res
            if on_result:
                on_result(i, res)
            bar.update(1)
            if not res.cached and res.tokens_per_s:
                bar.set_postfix_str(f"{res.tokens_per_s:.1f} tok/s")

        if concurrency <= 1:
            for i in range(len(requests_)):
                run(i)
        else:
            with ThreadPoolExecutor(max_workers=concurrency) as pool:
                list(pool.map(run, range(len(requests_))))
        bar.close()
        return [r if r is not None else ChatResult(model=model, error="not executed") for r in results]
