from time import perf_counter

from fastapi import APIRouter, Depends
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.db import get_session
from app.db import get_schema_status
from app.models import Asset, Chunk, Document
from app.schemas import AnswerRequest, AnswerResponse, SearchRequest, SearchResponse
from app.services.embeddings import get_embedding_service
from app.services.answer import AnswerService
from app.config import get_settings
from app.services.vector_store import VectorStore

router = APIRouter(prefix="/api", tags=["search"])


def search_diagnostics(db: Session, extra: dict | None = None) -> dict:
    total_docs = db.scalar(select(func.count(Document.id))) or 0
    approved_docs = db.scalar(select(func.count(Document.id)).where(Document.status == "approved")) or 0
    parsed_docs = db.scalar(select(func.count(Document.id)).where(Document.status == "parsed")) or 0
    failed_docs = db.scalar(select(func.count(Document.id)).where(Document.status == "failed")) or 0
    approved_chunks = (
        db.scalar(
            select(func.count(Chunk.id))
            .join(Document, Document.id == Chunk.document_id)
            .where(Document.status == "approved", Chunk.approved.is_(True))
        )
        or 0
    )
    embedded_chunks = (
        db.scalar(
            select(func.count(Chunk.id))
            .join(Document, Document.id == Chunk.document_id)
            .where(Document.status == "approved", Chunk.approved.is_(True), Chunk.embedding.is_not(None))
        )
        or 0
    )
    image_embedded_chunks = (
        db.scalar(
            select(func.count(Chunk.id))
            .join(Document, Document.id == Chunk.document_id)
            .where(
                Document.status == "approved",
                Chunk.approved.is_(True),
                Chunk.kind == "image",
                Chunk.embedding.is_not(None),
            )
        )
        or 0
    )
    image_secondary_embedded_chunks = (
        db.scalar(
            select(func.count(Chunk.id))
            .join(Document, Document.id == Chunk.document_id)
            .where(
                Document.status == "approved",
                Chunk.approved.is_(True),
                Chunk.kind == "image",
                Chunk.secondary_embedding.is_not(None),
            )
        )
        or 0
    )
    embedding_service = get_embedding_service()
    settings = get_settings()
    schema_status = get_schema_status()
    requires_reindex = bool(
        db.scalar(
            select(func.count(Chunk.id))
            .join(Document, Document.id == Chunk.document_id)
            .outerjoin(Asset, Asset.id == Chunk.asset_id)
            .where(
                Document.status == "approved",
                Chunk.approved.is_(True),
                Chunk.kind == "image",
                (
                    Chunk.embedding.is_(None)
                    | Chunk.secondary_embedding.is_(None)
                    | Asset.id.is_(None)
                    | Asset.region_type.is_(None)
                ),
            )
        )
    )
    return {
        "total_documents": total_docs,
        "approved_documents": approved_docs,
        "parsed_waiting_approval": parsed_docs,
        "failed_documents": failed_docs,
        "approved_chunks": approved_chunks,
        "embedded_chunks": embedded_chunks,
        "image_embedded_chunks": image_embedded_chunks,
        "image_secondary_embedded_chunks": image_secondary_embedded_chunks,
        "embedding_model": embedding_service.active_model_name,
        "image_embedding_model": embedding_service.active_image_model_name,
        "cloud_embedding_configured": embedding_service.cloud_embedding_configured,
        "image_embedding_configured": embedding_service.image_embedding_configured,
        "qdrant_available": VectorStore().available,
        "reranker_enabled": settings.reranker_enabled,
        "reranker_reachable": False,
        "reranker_healthy": False,
        "reranker_model": settings.reranker_model,
        "reranker_device": settings.reranker_device,
        "reranker_requested_device": settings.reranker_device,
        "reranker_use_fp16": settings.reranker_use_fp16,
        "schema_version_ok": schema_status["schema_version_ok"],
        "image_schema_columns_ready": schema_status["image_schema_columns_ready"],
        "requires_reindex": requires_reindex,
        "missing_schema_columns": schema_status["missing_columns"],
        **(extra or {}),
    }


@router.post("/search", response_model=SearchResponse)
def search(payload: SearchRequest, db: Session = Depends(get_session)) -> SearchResponse:
    kind = None if payload.kind == "all" else payload.kind
    store = VectorStore()
    results = store.search(
        db,
        payload.query,
        payload.mode,
        payload.top_k,
        payload.document_id,
        payload.page_from,
        payload.page_to,
        kind,
    )
    diagnostics = search_diagnostics(db, store.last_search_diagnostics)
    diagnostics["search_ms"] = diagnostics.get("total_search_ms", 0.0)
    return SearchResponse(
        query=payload.query,
        mode=payload.mode,
        results=results,
        diagnostics=diagnostics,
    )


@router.post("/answer", response_model=AnswerResponse)
def answer(payload: AnswerRequest, db: Session = Depends(get_session)) -> AnswerResponse:
    started = perf_counter()
    results = payload.results
    answer_service = AnswerService()
    usable_results = answer_service._usable_results(results)
    text = answer_service.answer(payload.query, results, payload.history)
    diagnostics = search_diagnostics(db)
    diagnostics["answer_ms"] = round((perf_counter() - started) * 1000, 2)
    return AnswerResponse(query=payload.query, answer=text, results=usable_results, diagnostics=diagnostics)
