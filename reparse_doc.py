"""Force re-parse a document to apply new image classification."""
import sys
from app.db import SessionLocal
from app.models import Document
from app.services.pdf_parser import parse_pdf

doc_id = "69eb1071-2fd8-4203-bb38-e264bc3b32bf"

db = SessionLocal()
try:
    doc = db.query(Document).filter(Document.id == doc_id).first()
    if not doc:
        print(f"Document {doc_id} not found")
        sys.exit(1)
    print(f"Re-parsing: {doc.filename}")
    print(f"Current status: {doc.status}")
    stats = parse_pdf(db, doc)
    print(f"Parse complete. Stats:")
    for k, v in stats.items():
        if isinstance(v, (int, str)):
            print(f"  {k}: {v}")
        elif isinstance(v, list):
            print(f"  {k}: {len(v)} items")
    print("Done!")
finally:
    db.close()
