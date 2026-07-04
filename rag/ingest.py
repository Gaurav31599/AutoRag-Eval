"""Corpus → chunk → embed → Chroma. Also writes the golden Q/A slice.

The corpus is the pool of paragraphs across the selected HotpotQA questions.
Because HotpotQA is multi-hop with distractor paragraphs, retrieval quality
genuinely matters — which is what makes the eval metrics move when knobs change.
"""
from __future__ import annotations

import json
from pathlib import Path

import chromadb
from datasets import load_dataset
from sentence_transformers import SentenceTransformer

from rag.config import resolve

GOLDEN_PATH = "data/golden/golden.jsonl"


def chunk_text(text: str, size: int, overlap: int) -> list[str]:
    """Character-based chunking with overlap."""
    if size <= 0:
        return [text]
    step = max(size - overlap, 1)
    chunks, start = [], 0
    while start < len(text):
        chunks.append(text[start : start + size])
        if start + size >= len(text):
            break
        start += step
    return chunks


def _build_corpus(examples) -> tuple[dict[str, str], list[dict]]:
    """Return {title: paragraph_text} (deduped) and the golden Q/A list."""
    docs: dict[str, str] = {}
    golden: list[dict] = []
    for ex in examples:
        golden.append(
            {"id": ex["id"], "question": ex["question"], "ground_truth": ex["answer"]}
        )
        titles = ex["context"]["title"]
        sentences = ex["context"]["sentences"]
        for title, sents in zip(titles, sentences):
            if title not in docs:
                docs[title] = " ".join(sents).strip()
    return docs, golden


def ingest(cfg: dict) -> dict:
    c = cfg["corpus"]
    ds = load_dataset(c["dataset"], c["subset"], split=c["split"])
    ds = ds.shuffle(seed=c["seed"]).select(range(c["num_questions"]))

    docs, golden = _build_corpus(ds)

    golden_file = resolve(cfg, GOLDEN_PATH)
    golden_file.parent.mkdir(parents=True, exist_ok=True)
    with open(golden_file, "w", encoding="utf-8") as f:
        for row in golden:
            f.write(json.dumps(row) + "\n")

    # Chunk every paragraph.
    ids, texts, metas = [], [], []
    size = cfg["chunking"]["chunk_size"]
    overlap = cfg["chunking"]["chunk_overlap"]
    for title, text in docs.items():
        if not text:
            continue
        for i, ch in enumerate(chunk_text(text, size, overlap)):
            ids.append(f"{title}::{i}")
            texts.append(ch)
            metas.append({"title": title, "chunk": i})

    model = SentenceTransformer(cfg["embedding"]["model"], device=cfg["embedding"]["device"])
    embeddings = model.encode(
        texts, show_progress_bar=True, normalize_embeddings=True, convert_to_numpy=True
    ).tolist()

    client = chromadb.PersistentClient(path=str(resolve(cfg, cfg["vector_store"]["persist_dir"])))
    name = cfg["vector_store"]["collection"]
    try:
        client.delete_collection(name)
    except Exception:
        pass
    coll = client.create_collection(name, metadata={"hnsw:space": "cosine"})
    coll.add(ids=ids, documents=texts, embeddings=embeddings, metadatas=metas)

    summary = {
        "questions": len(golden),
        "documents": len(docs),
        "chunks": len(texts),
        "golden_file": str(golden_file),
    }
    print(f"Ingested: {summary}")
    return summary


if __name__ == "__main__":
    from rag.config import load_config

    ingest(load_config())
