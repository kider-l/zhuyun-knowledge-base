"""Find all documents."""
from app.db import SessionLocal
from app.models import Document

db = SessionLocal()
try:
    docs = db.query(Document).all()
    print(f"Total docs: {len(docs)}")
    for d in docs:
        print(f"  id={d.id} status={d.status} filename={d.filename[:60]}")
finally:
    db.close()
