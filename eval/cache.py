"""Answer cache so duplicate Q/A pairs under the same config don't re-generate.

Key = a hash of the knobs that actually affect retrieval + generation. Change
any of them (chunk size, top-k, prompt, model, …) and you get a fresh cache
bucket; re-run the same config and every answer is served from disk for free.
"""
from __future__ import annotations

import hashlib
import json

from rag.config import resolve

_SIGNIFICANT = ("corpus", "chunking", "embedding", "retrieval", "generator")


def signature(cfg: dict) -> str:
    payload = {k: cfg[k] for k in _SIGNIFICANT}
    blob = json.dumps(payload, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


def _path(cfg: dict):
    d = resolve(cfg, ".cache/answers")
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{signature(cfg)}.json"


def load(cfg: dict) -> dict:
    p = _path(cfg)
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return {}


def save(cfg: dict, data: dict) -> None:
    _path(cfg).write_text(json.dumps(data), encoding="utf-8")
