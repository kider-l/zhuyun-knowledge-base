"""Check document and index status."""
from app.db import SessionLocal
from sqlalchemy import text

db = SessionLocal()
doc_id = "78a4a27b-7aa4-4024-afff-a0ebe6c76c26"

doc = db.execute(
    text("SELECT id, status, parse_stats, error_message FROM documents WHERE id = :id"),
    {"id": doc_id}
).fetchone()
print(f"Document: status={doc.status}, error={doc.error_message}")
if doc.parse_stats:
    s = doc.parse_stats if isinstance(doc.parse_stats, dict) else {}
    print(f"  image_chunks={s.get('image_chunks')}, text_chunks={s.get('text_chunks')}")
    print(f"  figure_regions={s.get('figure_regions')}")
    print(f"  vision_summaries={s.get('vision_summaries')}")
    print(f"  skipped_tiny_figures={s.get('skipped_tiny_figures')}")
    print(f"  skipped_tiny_regions={s.get('skipped_tiny_regions')}")

jobs = db.execute(
    text("SELECT job_type, status, progress, message, error_message FROM jobs WHERE document_id = :id ORDER BY created_at DESC LIMIT 5"),
    {"id": doc_id}
).fetchall()
print("\nRecent jobs:")
for j in jobs:
    print(f"  {j.job_type}: {j.status} {j.progress}% err={j.error_message} msg={j.message}")

# Check chunks exist for this doc
chunks = db.execute(
    text("SELECT kind, COUNT(*) as cnt FROM chunks WHERE document_id = :id GROUP BY kind"),
    {"id": doc_id}
).fetchall()
print("\nChunks:")
for c in chunks:
    print(f"  {c.kind}: {c.cnt}")

db.close()
