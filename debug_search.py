"""Debug: check what search returns for the accident query."""
from app.db import SessionLocal
from app.models import Chunk, Asset
from app.services.vector_store import VectorStore

db = SessionLocal()
try:
    store = VectorStore()
    query = "吊车事故现场图片 事故起因经过结果"
    # Search with mode="all", top_k=20
    results = store.search(db, query, mode="all", top_k=20)
    print(f"Main search returned {len(results)} results")
    image_results = [r for r in results if r.kind == "image" and r.asset_url]
    print(f"Image results with asset_url: {len(image_results)}")
    for r in image_results:
        meta = r.metadata
        asset_kind = meta.get("asset_kind", "?")
        print(f"  [{results.index(r)}] page={r.page_number} kind={asset_kind} score={r.score} name={r.document_name[:40]}")

    # Image-only search
    img_results = store.search(db, query, mode="all", top_k=25, kind="image")
    print(f"\nImage-only search returned {len(img_results)} results")
    embedded = [r for r in img_results if r.asset_url and r.metadata.get("asset_kind") == "embedded_image"]
    print(f"Embedded image results: {len(embedded)}")
    for r in embedded:
        print(f"  page={r.page_number} score={r.score} caption={(r.metadata.get('caption') or '')[:40]}")
finally:
    db.close()
