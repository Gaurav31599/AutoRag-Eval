"""Config-driven retriever: embed query → Chroma top-k → optional reranker."""
from __future__ import annotations

import chromadb
from sentence_transformers import SentenceTransformer

from rag.config import resolve


class Retriever:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.model = SentenceTransformer(
            cfg["embedding"]["model"], device=cfg["embedding"]["device"]
        )
        client = chromadb.PersistentClient(
            path=str(resolve(cfg, cfg["vector_store"]["persist_dir"]))
        )
        self.coll = client.get_collection(cfg["vector_store"]["collection"])

        self.reranker = None
        rr = cfg["retrieval"]["reranker"]
        if rr["enabled"]:
            from sentence_transformers import CrossEncoder

            self.reranker = CrossEncoder(rr["model"], device=cfg["embedding"]["device"])

    def retrieve(self, question: str) -> list[str]:
        rcfg = self.cfg["retrieval"]
        top_k = rcfg["top_k"]
        emb = self.model.encode(
            [question], normalize_embeddings=True, convert_to_numpy=True
        ).tolist()

        if self.reranker:
            top_n = rcfg["reranker"]["top_n"]
            fetch = max(top_k, top_n) * 3
            docs = self.coll.query(query_embeddings=emb, n_results=fetch)["documents"][0]
            scores = self.reranker.predict([[question, d] for d in docs])
            ranked = [d for _, d in sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)]
            return ranked[:top_n]

        return self.coll.query(query_embeddings=emb, n_results=top_k)["documents"][0]
