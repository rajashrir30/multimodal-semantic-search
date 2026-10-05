"""CLI for indexing a folder of images."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

from .embed_images import embed_images
from .faiss_index import build_index, load_index, load_metadata, save_index, save_metadata

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".tif", ".tiff"}


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def discover_images(source: Path) -> list[Path]:
    return sorted(path for path in source.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a FAISS image-search index.")
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--index-path", type=Path, default=Path("faiss_index/images.index"))
    parser.add_argument("--metadata-path", type=Path, default=Path("faiss_index/images_metadata.json"))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--force", action="store_true", help="Re-embed every image.")
    args = parser.parse_args()

    paths = discover_images(args.source)
    if not paths:
        raise SystemExit(f"No supported image files found under {args.source}")
    records = [{"filepath": str(path), "filename": path.name, "sha256": file_hash(path)} for path in paths]

    can_append = args.index_path.is_file() and args.metadata_path.is_file() and not args.force
    old_metadata = load_metadata(args.metadata_path) if can_append else []
    old_by_path = {item["filepath"]: item for item in old_metadata}
    current_paths = {item["filepath"] for item in records}
    unchanged = all(old_by_path.get(item["filepath"], {}).get("sha256") == item["sha256"] for item in records)
    # Appending is safe only when every prior record is still present. If a
    # file was deleted or changed, rebuild so FAISS positions stay aligned.
    only_new = unchanged and set(old_by_path).issubset(current_paths)

    if can_append and only_new:
        new_records = [item for item in records if item["filepath"] not in old_by_path]
        if new_records:
            index = load_index(args.index_path)
            index.add(embed_images([item["filepath"] for item in new_records], args.batch_size))
            save_index(index, args.index_path)
            save_metadata(old_metadata + new_records, args.metadata_path)
            print(f"Added {len(new_records)} new image(s); kept {len(old_metadata)} existing image(s).")
        else:
            print(f"Index is up to date ({len(records)} image(s)); nothing to embed.")
        return

    embeddings = embed_images([item["filepath"] for item in records], args.batch_size)
    save_index(build_index(embeddings), args.index_path)
    save_metadata(records, args.metadata_path)
    print(f"Indexed {len(records)} image(s).")


if __name__ == "__main__":
    main()
