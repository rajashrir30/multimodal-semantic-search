import numpy as np

from src.faiss_index import build_index, load_index, save_index, search


def test_build_save_load_roundtrip(tmp_path):
    embeddings = np.eye(3, dtype=np.float32)
    path = tmp_path / "images.faiss"
    save_index(build_index(embeddings), path)
    loaded = load_index(path)
    assert loaded.ntotal == 3
    assert loaded.d == 3


def test_search_returns_top_k_count():
    index = build_index(np.eye(4, dtype=np.float32))
    results = search(index, np.array([1, 0, 0, 0], dtype=np.float32), k=2)
    assert len(results) == 2


def test_exact_match_ranks_highest():
    embeddings = np.array([[0, 1, 0], [1, 0, 0], [0.99, 0.1, 0]], dtype=np.float32)
    results = search(build_index(embeddings), embeddings[2], k=3)
    assert results[0][0] == 2
    assert results[0][1] > results[1][1]
