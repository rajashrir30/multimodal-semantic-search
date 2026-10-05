"""CLI for text-to-image semantic search."""

from __future__ import annotations

import argparse
from pathlib import Path

from .embed_audio import embed_text_for_audio
from .embed_images import embed_text
from .faiss_index import load_index, load_metadata, search


def _print_results(label: str, index_path: Path, metadata_path: Path, query_embedding, k: int) -> None:
    print(f"{label} RESULTS")
    metadata = load_metadata(metadata_path)
    for position, score in search(load_index(index_path), query_embedding, k):
        item = metadata[position]
        print(f"{score:.4f}\t{item['filename']}\t{item['filepath']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Search indexed images and/or audio with a text query.")
    parser.add_argument("--query", required=True)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--media", choices=("image", "audio", "both"), default="image")
    parser.add_argument("--index-path", type=Path, help="Custom image index path.")
    parser.add_argument("--metadata-path", type=Path, help="Custom image metadata path.")
    parser.add_argument("--audio-index-path", type=Path, default=Path("faiss_index/audio.index"))
    parser.add_argument("--audio-metadata-path", type=Path, default=Path("faiss_index/audio_metadata.json"))
    args = parser.parse_args()

    image_index = args.index_path or Path("faiss_index/images.index")
    image_metadata = args.metadata_path or Path("faiss_index/images_metadata.json")
    if args.index_path is None and not image_index.exists() and Path("data/index.faiss").exists():
        image_index, image_metadata = Path("data/index.faiss"), Path("data/metadata.json")
    if args.media in ("image", "both"):
        _print_results("IMAGE", image_index, image_metadata, embed_text(args.query), args.k)
    if args.media in ("audio", "both"):
        _print_results("AUDIO", args.audio_index_path, args.audio_metadata_path, embed_text_for_audio(args.query), args.k)


if __name__ == "__main__":
    main()
