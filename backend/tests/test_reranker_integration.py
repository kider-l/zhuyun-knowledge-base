from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from app.api.routes_system import system_status
from app.db import Base
from app.models import Chunk, Document
from app.services.reranker import RerankerHealth, RerankOutcome
from app.services.vector_store import VectorHit, VectorStore


def make_session() -> Session:
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine, autoflush=False, autocommit=False, expire_on_commit=False)()


def add_document_with_chunks(session: Session) -> tuple[Document, Chunk, Chunk]:
    document = Document(
        filename="sample.pdf",
        stored_path=str(Path("sample.pdf")),
        status="approved",
        page_count=1,
        size_bytes=128,
        parse_stats={},
    )
    session.add(document)
    session.flush()
    first = Chunk(
        document_id=document.id,
        page_number=1,
        kind="text",
        content="inspection overview",
        approved=True,
        embedding=[0.1, 0.2],
        embedding_model="test-embedding",
        embedding_dim=2,
        chunk_metadata={},
    )
    second = Chunk(
        document_id=document.id,
        page_number=1,
        kind="text",
        content="inspection workflow detailed steps",
        approved=True,
        embedding=[0.2, 0.3],
        embedding_model="test-embedding",
        embedding_dim=2,
        chunk_metadata={},
    )
    session.add_all([first, second])
    session.commit()
    return document, first, second


def test_vector_store_prefers_rerank_scores(monkeypatch) -> None:
    session = make_session()
    _document, first, second = add_document_with_chunks(session)
    settings = VectorStore().settings
    monkeypatch.setattr(settings, "reranker_enabled", True)
    monkeypatch.setattr(settings, "reranker_model", "test-reranker")
    monkeypatch.setattr(settings, "reranker_device", "cpu")
    monkeypatch.setattr(settings, "reranker_candidate_multiplier", 5)
    monkeypatch.setattr(settings, "reranker_max_candidates", 30)
    monkeypatch.setattr(VectorStore, "search_qdrant", lambda *args, **kwargs: [])
    monkeypatch.setattr(
        VectorStore,
        "search_fallback",
        lambda self, db, query, mode, top_k, document_id=None, page_from=None, page_to=None, kind=None: [
            VectorHit(chunk_id=first.id, score=0.97),
            VectorHit(chunk_id=second.id, score=0.88),
        ],
    )
    monkeypatch.setattr("app.services.vector_store._highlight_boxes", lambda document, chunk, preview, query: ([], "none"))
    monkeypatch.setattr(
        "app.services.vector_store.rerank_candidates",
        lambda query, candidates, top_k: RerankOutcome(
            items=[
                {"chunk_id": second.id, "content": second.content, "rerank_score": 0.99, "reranked": True},
                {"chunk_id": first.id, "content": first.content, "rerank_score": 0.61, "reranked": True},
            ],
            applied=True,
            reachable=True,
            healthy=True,
            model="test-reranker",
            device="cpu",
            candidate_count=2,
        ),
    )

    store = VectorStore()
    results = store.search(session, "unmatched query", mode="text", top_k=2)

    assert [result.chunk_id for result in results] == [second.id, first.id]
    assert results[0].metadata["reranked"] is True
    assert results[0].metadata["rerank_score"] == 0.99
    assert results[0].metadata["vector_score"] == 0.88
    assert store.last_search_diagnostics["reranker_applied"] is True
    assert store.last_search_diagnostics["reranker_model"] == "test-reranker"
    assert store.last_search_diagnostics["reranker_candidate_count"] == 2


def test_vector_store_falls_back_when_reranker_unavailable(monkeypatch) -> None:
    session = make_session()
    _document, first, second = add_document_with_chunks(session)
    settings = VectorStore().settings
    monkeypatch.setattr(settings, "reranker_enabled", True)
    monkeypatch.setattr(settings, "reranker_model", "test-reranker")
    monkeypatch.setattr(settings, "reranker_device", "cpu")
    monkeypatch.setattr(VectorStore, "search_qdrant", lambda *args, **kwargs: [])
    monkeypatch.setattr(
        VectorStore,
        "search_fallback",
        lambda self, db, query, mode, top_k, document_id=None, page_from=None, page_to=None, kind=None: [
            VectorHit(chunk_id=first.id, score=0.92),
            VectorHit(chunk_id=second.id, score=0.81),
        ],
    )
    monkeypatch.setattr("app.services.vector_store._highlight_boxes", lambda document, chunk, preview, query: ([], "none"))
    monkeypatch.setattr(
        "app.services.vector_store.rerank_candidates",
        lambda query, candidates, top_k: RerankOutcome(
            applied=False,
            fallback_reason="timeout",
            reachable=False,
            healthy=False,
            model="test-reranker",
            device="cpu",
            candidate_count=2,
        ),
    )

    store = VectorStore()
    results = store.search(session, "unmatched query", mode="text", top_k=2)

    assert [result.chunk_id for result in results] == [first.id, second.id]
    assert results[0].metadata["reranked"] is False
    assert results[0].metadata["rerank_score"] is None
    assert store.last_search_diagnostics["reranker_applied"] is False
    assert store.last_search_diagnostics["reranker_fallback_reason"] == "timeout"


def test_vector_store_skips_rerank_for_image_mode(monkeypatch) -> None:
    session = make_session()
    _document, first, second = add_document_with_chunks(session)
    monkeypatch.setattr(VectorStore().settings, "reranker_enabled", True)
    monkeypatch.setattr(VectorStore, "search_qdrant", lambda *args, **kwargs: [])
    monkeypatch.setattr(
        VectorStore,
        "search_fallback",
        lambda self, db, query, mode, top_k, document_id=None, page_from=None, page_to=None, kind=None: [
            VectorHit(chunk_id=first.id, score=0.92),
            VectorHit(chunk_id=second.id, score=0.81),
        ],
    )
    monkeypatch.setattr("app.services.vector_store._highlight_boxes", lambda document, chunk, preview, query: ([], "none"))
    called = {"count": 0}

    def fake_rerank(query, candidates, top_k):
        called["count"] += 1
        return RerankOutcome(applied=False, fallback_reason="no rerank candidates", candidate_count=len(candidates))

    monkeypatch.setattr("app.services.vector_store.rerank_candidates", fake_rerank)

    results = VectorStore().search(session, "evacuation map", mode="images", top_k=2)

    assert [result.chunk_id for result in results] == [first.id, second.id]
    assert called["count"] == 1
    assert results[0].metadata["reranked"] is False


def test_vector_store_process_mode_reserves_image_results_without_reranking(monkeypatch) -> None:
    session = make_session()
    document = Document(
        filename="sample.pdf",
        stored_path=str(Path("sample.pdf")),
        status="approved",
        page_count=1,
        size_bytes=128,
        parse_stats={},
    )
    session.add(document)
    session.flush()
    text_chunk = Chunk(
        document_id=document.id,
        page_number=1,
        kind="text",
        content="inspection workflow text",
        approved=True,
        embedding=[0.1, 0.2],
        embedding_model="test-embedding",
        embedding_dim=2,
        chunk_metadata={},
    )
    image_chunk = Chunk(
        document_id=document.id,
        page_number=1,
        kind="image",
        content="A area evacuation route east safety exit",
        approved=True,
        embedding=[0.2, 0.3],
        embedding_model="test-image",
        embedding_dim=2,
        secondary_embedding=[0.3, 0.1],
        secondary_embedding_model="test-embedding",
        secondary_embedding_dim=2,
        chunk_metadata={"region_summary_text": "A area evacuation route east safety exit"},
    )
    session.add_all([text_chunk, image_chunk])
    session.commit()
    settings = VectorStore().settings
    monkeypatch.setattr(settings, "reranker_enabled", True)
    monkeypatch.setattr(settings, "process_image_reserve_ratio", 0.3)
    monkeypatch.setattr(settings, "reranker_candidate_multiplier", 5)
    monkeypatch.setattr(settings, "reranker_max_candidates", 30)
    monkeypatch.setattr(VectorStore, "search_qdrant", lambda *args, **kwargs: [])
    monkeypatch.setattr(
        VectorStore,
        "search_fallback",
        lambda self, db, query, mode, top_k, document_id=None, page_from=None, page_to=None, kind=None: [
            VectorHit(chunk_id=text_chunk.id, score=0.95),
            VectorHit(chunk_id=image_chunk.id, score=0.42),
        ],
    )
    monkeypatch.setattr("app.services.vector_store._highlight_boxes", lambda document, chunk, preview, query: ([], "none"))
    called = {"count": 0}

    def fake_rerank(query, candidates, top_k):
        called["count"] += 1
        return RerankOutcome(applied=False, fallback_reason="disabled", candidate_count=len(candidates))

    monkeypatch.setattr("app.services.vector_store.rerank_candidates", fake_rerank)

    store = VectorStore()
    results = store.search(session, "A area evacuation route", mode="process", top_k=2)

    assert len(results) == 2
    assert {result.kind for result in results} == {"text", "image"}
    assert called["count"] == 1
    assert store.last_search_diagnostics["reranker_applied"] is False
    assert store.last_search_diagnostics["fusion_strategy"] == "process_dual_channel_reserve_45pct_images"


def test_vector_store_image_mode_keeps_keyword_only_visual_regions(monkeypatch) -> None:
    session = make_session()
    document = Document(
        filename="sample.pdf",
        stored_path=str(Path("sample.pdf")),
        status="approved",
        page_count=1,
        size_bytes=128,
        parse_stats={},
    )
    session.add(document)
    session.flush()
    image_chunk = Chunk(
        document_id=document.id,
        page_number=1,
        kind="image",
        content="C4 现场疏散图 东侧出口 疏散路线",
        approved=True,
        embedding=None,
        chunk_metadata={"asset_kind": "figure_region", "region_summary_text": "C4 现场疏散图 东侧出口 疏散路线"},
    )
    session.add(image_chunk)
    session.commit()
    monkeypatch.setattr(VectorStore, "search_qdrant", lambda *args, **kwargs: [])
    monkeypatch.setattr(VectorStore, "search_fallback", lambda *args, **kwargs: [])
    monkeypatch.setattr("app.services.vector_store._highlight_boxes", lambda document, chunk, preview, query: ([], "none"))

    store = VectorStore()
    results = store.search(session, "C4现场疏散图出口路线", mode="images", top_k=5)

    assert [result.chunk_id for result in results] == [image_chunk.id]
    assert results[0].metadata["asset_kind"] == "figure_region"
    assert results[0].metadata["keyword_score"] > 0
    assert results[0].metadata["retrieval_channels"] == ["keyword"]
    assert store.last_search_diagnostics["keyword_candidate_count"] >= 1
    assert store.last_search_diagnostics["image_result_type_breakdown"]["figure_region"] >= 1


def test_system_status_reports_reranker_health(monkeypatch) -> None:
    session = make_session()
    monkeypatch.setattr(
        "app.api.routes_system.get_reranker_health",
        lambda: RerankerHealth(
            enabled=True,
            reachable=True,
            healthy=True,
            model="test-reranker",
            device="cpu",
            requested_device="cpu",
            use_fp16=False,
            error=None,
        ),
    )
    monkeypatch.setattr(
        "app.api.routes_system.get_schema_status",
        lambda: {
            "schema_version_ok": True,
            "image_schema_columns_ready": True,
            "secondary_embedding_columns_ready": True,
            "missing_columns": {},
        },
    )

    payload = system_status("admin", session)

    assert payload["services"]["reranker_enabled"] is True
    assert payload["services"]["reranker_reachable"] is True
    assert payload["services"]["reranker_healthy"] is True
    assert payload["services"]["reranker_model"] == "test-reranker"
    assert payload["services"]["schema_version_ok"] is True
    assert payload["services"]["image_schema_columns_ready"] is True
    assert payload["services"]["requires_reindex"] is False
