"""Wires config → retriever + generator into one object with .answer(q)."""
from __future__ import annotations

from rag.generator import generate
from rag.retriever import Retriever


class RAGPipeline:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.retriever = Retriever(cfg)

    def answer(self, question: str) -> dict:
        contexts = self.retriever.retrieve(question)
        answer, usage = generate(self.cfg, question, contexts)
        return {
            "question": question,
            "answer": answer,
            "contexts": contexts,
            "usage": usage,
        }
