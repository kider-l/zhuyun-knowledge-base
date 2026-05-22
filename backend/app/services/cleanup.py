from sqlalchemy import delete, text
from sqlalchemy.orm import Session

from app.models import Asset, Chunk


def cleanup_duplicates(db: Session) -> None:
    """Remove duplicate chunks and assets, keeping only the latest row per natural key."""
    # Phase 1: remove content-duplicate chunks (same content, keep latest)
    _dedup_chunks_by_content(db)

    # Phase 2: remove asset-id-duplicate chunks (same asset_id, keep latest)
    _dedup_chunks_by_asset(db)

    # Phase 3: find and remove duplicate assets
    dup_asset_ids = _find_duplicate_asset_ids(db, "page")
    dup_asset_ids.update(_find_duplicate_asset_ids(db, "image"))

    # Phase 4: delete chunks referencing duplicate assets, then the assets
    if dup_asset_ids:
        db.execute(delete(Chunk).where(Chunk.asset_id.in_(dup_asset_ids)))
        db.execute(delete(Asset).where(Asset.id.in_(dup_asset_ids)))

    db.commit()


def _dedup_chunks_by_content(db: Session) -> None:
    """Remove chunk duplicates that share the same (document_id, page_number, kind, content)."""
    rows = db.execute(
        text("""
            SELECT c1.id FROM chunks c1
            WHERE EXISTS (
                SELECT 1 FROM chunks c2
                WHERE c2.document_id = c1.document_id
                  AND c2.page_number = c1.page_number
                  AND c2.kind = c1.kind
                  AND c2.content = c1.content
                  AND c2.created_at > c1.created_at
            )
        """)
    ).scalars().all()
    if rows:
        db.execute(delete(Chunk).where(Chunk.id.in_(rows)))


def _dedup_chunks_by_asset(db: Session) -> None:
    """Remove chunk duplicates that share the same (document_id, page_number, kind, asset_id)."""
    rows = db.execute(
        text("""
            SELECT c1.id FROM chunks c1
            WHERE c1.asset_id IS NOT NULL
              AND EXISTS (
                  SELECT 1 FROM chunks c2
                  WHERE c2.document_id = c1.document_id
                    AND c2.page_number = c1.page_number
                    AND c2.kind = c1.kind
                    AND c2.asset_id = c1.asset_id
                    AND c2.created_at > c1.created_at
              )
        """)
    ).scalars().all()
    if rows:
        db.execute(delete(Chunk).where(Chunk.id.in_(rows)))


def _find_duplicate_asset_ids(db: Session, kind: str) -> set[str]:
    """Find older duplicate assets sharing the same (document_id, page_number, region_index, region_type)."""
    rows = db.execute(
        text("""
            SELECT a1.id FROM assets a1
            WHERE a1.kind = :kind
              AND EXISTS (
                  SELECT 1 FROM assets a2
                  WHERE a2.document_id = a1.document_id
                    AND a2.page_number = a1.page_number
                    AND a2.kind = a1.kind
                    AND (a2.region_index = a1.region_index OR (a2.region_index IS NULL AND a1.region_index IS NULL))
                    AND (a2.region_type = a1.region_type OR (a2.region_type IS NULL AND a1.region_type IS NULL))
                    AND a2.created_at > a1.created_at
              )
        """),
        {"kind": kind},
    ).scalars().all()
    return set(rows)
