"""HTTP API for multimodal semantic search and incremental uploads."""

from __future__ import annotations

import hashlib
import shutil
import uuid
from pathlib import Path
from threading import Lock

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .embed_audio import embed_audio, embed_text_for_audio
from .embed_images import embed_images, embed_text


ROOT = Path(__file__).resolve().parent.parent
INDEX_DIR = ROOT / "faiss_index"
UPLOAD_ROOT = ROOT / "data" / "uploads"
IMAGE_INDEX = INDEX_DIR / "images.index"
AUDIO_INDEX = INDEX_DIR / "audio.index"
IMAGE_METADATA = INDEX_DIR / "images_metadata.json"
AUDIO_METADATA = INDEX_DIR / "audio_metadata.json"

for directory in (INDEX_DIR, UPLOAD_ROOT / "images", UPLOAD_ROOT / "audio", ROOT / "data" / "sample"):
    directory.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="Multimodal Semantic Search API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501", "http://127.0.0.1:8501"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/media/sample", StaticFiles(directory=str(ROOT / "data" / "sample")), name="sample-media")
app.mount("/media/uploads", StaticFiles(directory=str(UPLOAD_ROOT)), name="uploaded-media")

_INDEX_LOCK = Lock()
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
AUDIO_EXTENSIONS = {".wav", ".mp3"}


class SearchRequest(BaseModel):
    query: str = Field(min_length=1)
    media: str = "image"
    k: int = Field(default=5, ge=1, le=100)


def _validate_media(media: str) -> None:
    if media not in {"image", "audio", "both"}:
        raise HTTPException(status_code=400, detail="media must be image, audio, or both")


def _public_path(path: str) -> str:
    """Translate an indexed filesystem path into a browser-served URL."""
    candidate = Path(path)
    try:
        relative = candidate.resolve().relative_to((ROOT / "data" / "sample").resolve())
        return "/media/sample/" + relative.as_posix()
    except ValueError:
        pass
    try:
        relative = candidate.resolve().relative_to(UPLOAD_ROOT.resolve())
        return "/media/uploads/" + relative.as_posix()
    except ValueError:
        return path


def _search_index(query: str, media: str, k: int) -> list[dict]:
    from .faiss_index import load_index, load_metadata, search as faiss_search

    if media == "image":
        embedding = embed_text(query)
        index_path, metadata_path = IMAGE_INDEX, IMAGE_METADATA
    else:
        embedding = embed_text_for_audio(query)
        index_path, metadata_path = AUDIO_INDEX, AUDIO_METADATA
    index = load_index(index_path)
    metadata = load_metadata(metadata_path)
    results = []
    for position, score in faiss_search(index, embedding, k):
        item = metadata[position]
        results.append({"filename": item["filename"], "score": float(score), "path": _public_path(item["filepath"])})
    return results


def _index_count(index_path: Path, metadata_path: Path) -> int:
    from .faiss_index import load_index

    try:
        return int(load_index(index_path).ntotal)
    except FileNotFoundError:
        return 0


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/stats")
def stats() -> dict[str, int]:
    return {
        "images_indexed": _index_count(IMAGE_INDEX, IMAGE_METADATA),
        "audio_indexed": _index_count(AUDIO_INDEX, AUDIO_METADATA),
    }


@app.post("/search")
def search_endpoint(request: SearchRequest) -> dict[str, list[dict] | None]:
    _validate_media(request.media)
    response: dict[str, list[dict] | None] = {"image_results": None, "audio_results": None}
    try:
        if request.media in {"image", "both"}:
            response["image_results"] = _search_index(request.query, "image", request.k)
        if request.media in {"audio", "both"}:
            response["audio_results"] = _search_index(request.query, "audio", request.k)
        return response
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=f"Index is not available: {exc}") from exc
    except (ValueError, KeyError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Search failed") from exc


def _upload_name(filename: str) -> str:
    name = Path(filename).name
    stem, suffix = Path(name).stem, Path(name).suffix.lower()
    return f"{stem}-{uuid.uuid4().hex[:8]}{suffix}"


@app.post("/upload")
async def upload(file: UploadFile = File(...)) -> dict[str, str]:
    original_name = file.filename or ""
    extension = Path(original_name).suffix.lower()
    if extension in IMAGE_EXTENSIONS:
        media_type, destination_dir = "image", UPLOAD_ROOT / "images"
    elif extension in AUDIO_EXTENSIONS:
        media_type, destination_dir = "audio", UPLOAD_ROOT / "audio"
    else:
        raise HTTPException(status_code=400, detail="Unsupported file type. Use JPG/PNG images or WAV/MP3 audio.")

    destination = destination_dir / _upload_name(original_name)
    try:
        with destination.open("wb") as output:
            shutil.copyfileobj(file.file, output)
        record = {
            "filepath": str(destination),
            "filename": original_name,
            "sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
        }
        with _INDEX_LOCK:
            from .faiss_index import build_index, load_index, load_metadata, save_index, save_metadata

            if media_type == "image":
                embedding = embed_images([str(destination)])
                index_path, metadata_path = IMAGE_INDEX, IMAGE_METADATA
            else:
                embedding = embed_audio([str(destination)])
                index_path, metadata_path = AUDIO_INDEX, AUDIO_METADATA
            if index_path.exists() and metadata_path.exists():
                index = load_index(index_path)
                metadata = load_metadata(metadata_path)
                index.add(embedding)
                metadata.append(record)
            else:
                index = build_index(embedding)
                metadata = [record]
            save_index(index, index_path)
            save_metadata(metadata, metadata_path)
        return {"filename": original_name, "media_type": media_type, "status": "indexed"}
    except HTTPException:
        raise
    except (ValueError, OSError) as exc:
        destination.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail=f"Could not index upload: {exc}") from exc
    except Exception as exc:
        destination.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail="Upload indexing failed") from exc
