"""CLI for indexing a folder of audio clips with CLAP."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

from .embed_audio import embed_audio
from .faiss_index import build_index, load_index, load_metadata, save_index, save_metadata

AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac", ".aiff", ".au"}


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def discover_audio(source: Path) -> list[Path]:
    return sorted(path for path in source.rglob("*") if path.is_file() and path.suffix.lower() in AUDIO_EXTENSIONS)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a FAISS audio-search index with CLAP.")
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--index-path", type=Path, default=Path("faiss_index/audio.index"))
    parser.add_argument("--metadata-path", type=Path, default=Path("faiss_index/audio_metadata.json"))
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--force", action="store_true", help="Re-embed every audio file.")
    args = parser.parse_args()

    paths = discover_audio(args.source)
    if not paths:
        raise SystemExit(f"No supported audio files found under {args.source}")
    records = [{"filepath": str(path), "filename": path.name, "sha256": file_hash(path)} for path in paths]
    can_append = args.index_path.is_file() and args.metadata_path.is_file() and not args.force
    old_metadata = load_metadata(args.metadata_path) if can_append else []
    old_by_path = {item["filepath"]: item for item in old_metadata}
    current_paths = {item["filepath"] for item in records}
    unchanged = all(old_by_path.get(item["filepath"], {}).get("sha256") == item["sha256"] for item in records)
    only_new = unchanged and set(old_by_path).issubset(current_paths)

    if can_append and only_new:
        new_records = [item for item in records if item["filepath"] not in old_by_path]
        if new_records:
            index = load_index(args.index_path)
            index.add(embed_audio([item["filepath"] for item in new_records], args.batch_size))
            save_index(index, args.index_path)
            save_metadata(old_metadata + new_records, args.metadata_path)
            print(f"Added {len(new_records)} new audio clip(s); kept {len(old_metadata)} existing clip(s).")
        else:
            print(f"Audio index is up to date ({len(records)} clip(s)); nothing to embed.")
        return

    embeddings = embed_audio([item["filepath"] for item in records], args.batch_size)
    save_index(build_index(embeddings), args.index_path)
    save_metadata(records, args.metadata_path)
    print(f"Indexed {len(records)} audio clip(s).")


if __name__ == "__main__":
    main()
