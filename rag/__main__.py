"""CLI entrypoint.

  python -m rag ingest          # build corpus + golden set + vector store
  python -m rag "your question" # retrieve + answer
"""
from __future__ import annotations

import sys

from rag.config import load_config


def main() -> None:
    args = sys.argv[1:]
    if not args:
        print('Usage: python -m rag ingest  |  python -m rag "your question"')
        raise SystemExit(1)

    cfg = load_config()

    if args[0] == "ingest":
        from rag.ingest import ingest

        ingest(cfg)
        return

    question = " ".join(args)
    from rag.pipeline import RAGPipeline

    out = RAGPipeline(cfg).answer(question)

    print("\n=== ANSWER ===")
    print(out["answer"])
    print("\n=== RETRIEVED CONTEXT ===")
    for i, ctx in enumerate(out["contexts"], 1):
        snippet = ctx[:400] + ("…" if len(ctx) > 400 else "")
        print(f"\n[{i}] {snippet}")
    u = out["usage"]
    print(f"\n=== TOKENS ===  in={u['input']}  out={u['output']}")


if __name__ == "__main__":
    main()
