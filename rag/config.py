"""Load the single YAML config. Path override via RAG_CONFIG env or argument."""
from __future__ import annotations

import os
from pathlib import Path

import yaml
from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = ROOT / "config" / "rag.yaml"


def load_config(path: str | os.PathLike | None = None) -> dict:
    path = Path(path or os.environ.get("RAG_CONFIG", DEFAULT_CONFIG))
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["_path"] = str(path)
    cfg["_root"] = str(ROOT)
    return cfg


def resolve(cfg: dict, relpath: str) -> Path:
    """Resolve a config-relative path (e.g. persist_dir) against the repo root."""
    p = Path(relpath)
    return p if p.is_absolute() else ROOT / p
