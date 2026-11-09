"""Configuration loading: YAML defaults + profile overrides + CLI overrides."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "default.yaml"


class Config(dict):
    """A dict with attribute access (recursive) – keeps the code readable."""

    def __getattr__(self, item: str) -> Any:
        try:
            value = self[item]
        except KeyError as exc:  # pragma: no cover - defensive
            raise AttributeError(item) from exc
        return Config(value) if isinstance(value, dict) and not isinstance(value, Config) else value

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value


def deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def _to_config(obj: Any) -> Any:
    if isinstance(obj, dict):
        return Config({k: _to_config(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [_to_config(v) for v in obj]
    return obj


def load_config(path: str | Path | None = None, profile: str | None = None, overrides: dict | None = None) -> Config:
    path = Path(path) if path else DEFAULT_CONFIG
    with open(path, encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    profiles = raw.pop("profiles", {}) or {}
    if profile:
        if profile not in profiles:
            raise ValueError(f"Unknown profile '{profile}'. Available: {', '.join(profiles) or 'none'}")
        raw = deep_merge(raw, profiles[profile] or {})
    if overrides:
        raw = deep_merge(raw, overrides)
    raw["profile"] = profile or "standard"
    cfg = _to_config(raw)
    validate(cfg)
    return cfg


def validate(cfg: Config) -> None:
    if not cfg.get("models"):
        raise ValueError("config: at least one model is required under `models:`")
    for m in cfg["models"]:
        if not isinstance(m, dict) or not m.get("name"):
            raise ValueError("config: every model needs a `name` (an Ollama tag such as qwen3:4b-instruct)")
        m.setdefault("label", m["name"])
    ch = cfg["chunking"]
    if not 0 <= ch["overlap_tokens"] < ch["target_tokens"] <= ch["max_tokens"]:
        raise ValueError("config: require 0 <= overlap_tokens < target_tokens <= max_tokens")
    strategy = cfg["judge"]["strategy"]
    if strategy not in {"cross", "fixed", "self"}:
        raise ValueError("config: judge.strategy must be cross | fixed | self")
    weights = cfg["scoring"]["weights"]
    if any(w < 0 for w in weights.values()) or sum(weights.values()) <= 0:
        raise ValueError("config: scoring.weights must be non-negative and not all zero")


def model_names(cfg: Config) -> list[str]:
    return [m["name"] for m in cfg["models"]]


def model_label(cfg: Config, name: str) -> str:
    for m in cfg["models"]:
        if m["name"] == name:
            return m.get("label", name)
    return name


def resolve_models(cfg: Config, value: Any) -> list[str]:
    """`auto` -> all models under test; a string -> [string]; a list -> list."""
    if value in (None, "auto", []):
        return model_names(cfg)
    if isinstance(value, str):
        return [value]
    return list(value)


def resolve_path(p: str | Path) -> Path:
    p = Path(p)
    return p if p.is_absolute() else PROJECT_ROOT / p
