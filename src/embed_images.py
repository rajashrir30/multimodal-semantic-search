"""CLIP image and text embedding helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image, UnidentifiedImageError


_MODEL_NAME = "ViT-B-32"
_PRETRAINED = "openai"
_MODEL = None
_PREPROCESS = None
_TOKENIZER = None
_DEVICE = None


def load_image(path: str | Path) -> Image.Image:
    """Load and validate an image, returning it as an RGB PIL image."""
    image_path = Path(path)
    if not image_path.is_file():
        raise FileNotFoundError(f"Image file does not exist: {image_path}")
    try:
        with Image.open(image_path) as image:
            image.verify()
        with Image.open(image_path) as image:
            return image.convert("RGB")
    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError(f"Invalid image file: {image_path}") from exc


def _get_model():
    global _MODEL, _PREPROCESS, _TOKENIZER, _DEVICE
    if _MODEL is None:
        import open_clip
        import torch

        _DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        _MODEL, _, _PREPROCESS = open_clip.create_model_and_transforms(
            _MODEL_NAME, pretrained=_PRETRAINED, device=_DEVICE
        )
        _TOKENIZER = open_clip.get_tokenizer(_MODEL_NAME)
        _MODEL.eval()
    return _MODEL, _PREPROCESS, _TOKENIZER, _DEVICE


def _normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
    return vectors / np.maximum(norms, 1e-12)


def embed_images(paths: list[str], batch_size: int = 32) -> np.ndarray:
    """Embed image paths in batches into normalized CLIP image vectors."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    model, preprocess, _, device = _get_model()
    import torch

    images = [preprocess(load_image(path)) for path in paths]
    if not images:
        # CLIP's dimension is 512 for ViT-B-32, but derive it without a model
        # forward pass only when possible.
        return np.empty((0, 512), dtype=np.float32)

    output = []
    with torch.inference_mode():
        for start in range(0, len(images), batch_size):
            batch = torch.stack(images[start : start + batch_size]).to(device)
            output.append(model.encode_image(batch).float().cpu().numpy())
    return _normalize(np.concatenate(output, axis=0)).astype(np.float32)


def embed_text(query: str) -> np.ndarray:
    """Embed one text query with CLIP's text encoder."""
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be a non-empty string")
    model, _, tokenizer, device = _get_model()
    import torch

    with torch.inference_mode():
        vector = model.encode_text(tokenizer([query]).to(device)).float().cpu().numpy()[0]
    return _normalize(vector[None, :])[0].astype(np.float32)
