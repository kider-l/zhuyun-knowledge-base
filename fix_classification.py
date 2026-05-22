"""One-time fix: update existing image chunks that were wrongly classified as SCENE_IMAGE.
Runs inside the API container via: docker exec -i $(docker ps -q -f name=api) python /tmp/fix_classification.py
"""
import json
import sys

from app.db import SessionLocal
from app.models import Chunk

SCENE_KEYWORDS = ["logo", "水印", "装饰", "背景图", "图标"]


def _is_decorative(content: str) -> bool:
    """Only truly decorative elements should remain SCENE_IMAGE."""
    if not content:
        return False
    lower = content.lower()
    return any(kw in lower for kw in SCENE_KEYWORDS)


def main() -> None:
    db = SessionLocal()
    try:
        total = 0
        fixed = 0
        # Find all image chunks that have image_class = SCENE_IMAGE
        chunks = (
            db.query(Chunk)
            .filter(Chunk.kind == "image", Chunk.chunk_metadata["image_class"].as_string() == "SCENE_IMAGE")
            .all()
        )
        total = len(chunks)
        for chunk in chunks:
            meta = dict(chunk.chunk_metadata or {})
            old_class = meta.get("image_class")
            if old_class != "SCENE_IMAGE":
                continue
            content_text = (chunk.content or "") + str(meta.get("caption") or "") + str(meta.get("region_summary_text") or "")
            if not _is_decorative(content_text):
                # Reclassify as DOC_IMAGE - these are document-content images
                meta["image_class"] = "DOC_IMAGE"
                meta["class_confidence"] = 0.7
                meta["usable_for_qa"] = True
                meta["_reclassified"] = True
                chunk.chunk_metadata = meta
                db.add(chunk)
                fixed += 1
        db.commit()
        print(f"SCENE_IMAGE chunks found: {total}, reclassified to DOC_IMAGE: {fixed}")
    finally:
        db.close()


if __name__ == "__main__":
    main()
