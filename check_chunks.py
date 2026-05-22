"""Debug: check document image chunks."""
from app.db import SessionLocal
from app.models import Chunk, Asset, Document

db = SessionLocal()
try:
    doc_id = "69eb1071-2fd"
    doc = db.query(Document).filter(Document.id == doc_id).first()
    if doc:
        print(f"Doc: {doc.filename}")
        imgs = db.query(Chunk).filter(Chunk.document_id == doc_id, Chunk.kind == "image").order_by(Chunk.page_number).all()
        print(f"Total image chunks: {len(imgs)}")
        seen_pages = set()
        for c in imgs:
            meta = dict(c.chunk_metadata or {})
            ak = meta.get("asset_kind", "?")
            asset_disp = str(c.asset_id)[:12] if c.asset_id else "no-asset"
            page_key = f"p{c.page_number}-{ak}"
            is_new = "" if page_key in seen_pages else " <-- first"
            seen_pages.add(page_key)
            ct = ((c.content or "")[:120]).replace("\n", " ")
            print(f"  chunk={str(c.id)[:12]} page={c.page_number} kind={ak} asset={asset_disp}{is_new}")
            print(f"    text: {ct}")
        print(f"\nUnique page+kind combinations: {len(seen_pages)}")
    else:
        print("Doc not found")
finally:
    db.close()
