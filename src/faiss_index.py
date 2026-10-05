"""FAISS index and JSON metadata helpers."""

from __future__ import annotations

import json
from pathlib import Path

import faiss
import numpy as np


def _normalized(embeddings: np.ndarray) -> np.ndarray:
    values = np.asarray(embeddings, dtype=np.float32)
    if values.ndim != 2:
        raise ValueError("embeddings must have shape (n, dim)")
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.maximum(norms, 1e-12)


def build_index(embeddings: np.ndarray) -> faiss.Index:
    """Build an inner-product index over L2-normalized vectors."""
    values = _normalized(embeddings)
    if values.shape[0] == 0:
        raise ValueError("cannot build an index with no embeddings")
    index = faiss.IndexFlatIP(values.shape[1])
    index.add(values)
    return index


def save_index(index: faiss.Index, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(path))


def load_index(path: str | Path) -> faiss.Index:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"FAISS index does not exist: {path}")
    return faiss.read_index(str(path))


def search(index: faiss.Index, query_embedding: np.ndarray, k: int = 5) -> list[tuple[int, float]]:
    """Return (index position, cosine similarity) pairs in rank order."""
    if k < 1:
        raise ValueError("k must be positive")
    query = np.asarray(query_embedding, dtype=np.float32)
    if query.ndim == 1:
        query = query[None, :]
    query = _normalized(query)
    scores, positions = index.search(query, min(k, index.ntotal))
    return [(int(position), float(score)) for position, score in zip(positions[0], scores[0]) if position >= 0]


def save_metadata(metadata: list[dict], path: str | Path) -> None:
    """Save position-ordered metadata as a JSON object keyed by position."""
    payload = {str(position): item for position, item in enumerate(metadata)}
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_metadata(path: str | Path) -> list[dict]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Metadata file does not exist: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [payload[key] for key in sorted(payload, key=lambda value: int(value))]
