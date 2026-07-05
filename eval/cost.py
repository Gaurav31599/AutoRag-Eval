"""Token → USD accounting, using the per-model rates in config/rag.yaml."""
from __future__ import annotations


def usd(cfg: dict, model: str, usage: dict) -> float:
    rates = cfg.get("cost", {}).get(model)
    if not rates:
        return 0.0
    return usage["input"] / 1e6 * rates["input"] + usage["output"] / 1e6 * rates["output"]
