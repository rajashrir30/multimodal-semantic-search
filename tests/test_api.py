from io import BytesIO

from fastapi.testclient import TestClient

from src import api


client = TestClient(api.app)


def test_health_returns_ok():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_search_returns_separate_result_shapes(monkeypatch):
    monkeypatch.setattr(api, "_search_index", lambda query, media, k: [{"filename": f"{media}.x", "score": 0.9, "path": f"/{media}.x"}])
    response = client.post("/search", json={"query": "dog", "media": "both", "k": 5})
    assert response.status_code == 200
    assert response.json() == {
        "image_results": [{"filename": "image.x", "score": 0.9, "path": "/image.x"}],
        "audio_results": [{"filename": "audio.x", "score": 0.9, "path": "/audio.x"}],
    }


def test_upload_rejects_unsupported_extension():
    response = client.post("/upload", files={"file": ("notes.txt", BytesIO(b"not media"), "text/plain")})
    assert response.status_code == 400
    assert "Unsupported file type" in response.json()["detail"]


def test_stats_has_expected_shape(monkeypatch):
    monkeypatch.setattr(api, "_index_count", lambda index_path, metadata_path: 3 if "images" in str(index_path) else 2)
    response = client.get("/stats")
    assert response.status_code == 200
    assert response.json() == {"images_indexed": 3, "audio_indexed": 2}
