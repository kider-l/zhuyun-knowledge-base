from fastapi import APIRouter, BackgroundTasks, Depends
from sqlalchemy import func, or_, select
from sqlalchemy.orm import Session

from app.auth import AdminUser
from app.config import get_settings
from app.db import get_schema_status, get_session
from app.models import Asset, Chunk, Document, Job
from app.schemas import JobOut
from app.services.embeddings import get_embedding_service
from app.services.ocr import get_ocr_service
from app.services.reranker import get_reranker_health
from app.services.scheduler import enqueue_index
from app.services.vector_store import VectorStore

router = APIRouter(prefix="/api", tags=["system"])


def _indexable_documents(db: Session) -> list[Document]:
    return (
        db.execute(
            select(Document)
            .where(Document.status.in_(["parsed", "approved"]))
            .order_by(Document.updated_at.desc())
        )
        .scalars()
        .all()
    )


@router.get("/system/status")
def system_status(_: AdminUser, db: Session = Depends(get_session)) -> dict:
    settings = get_settings()
    embedding_service = get_embedding_service()
    reranker = get_reranker_health()
    latest_document_stats = (
        db.execute(
            select(Document.parse_stats)
            .where(Document.parse_stats.is_not(None))
            .order_by(Document.updated_at.desc())
        )
        .scalars()
        .first()
        or {}
    )
    if not isinstance(latest_document_stats, dict):
        latest_document_stats = {}
    latest_ocr_warning = {}
    for warning in reversed(latest_document_stats.get("warnings", []) or []):
        if isinstance(warning, dict) and isinstance(warning.get("ocr"), dict):
            latest_ocr_warning = warning["ocr"]
            break
    total_documents = db.scalar(select(func.count(Document.id))) or 0
    approved_documents = db.scalar(select(func.count(Document.id)).where(Document.status == "approved")) or 0
    parsed_documents = db.scalar(select(func.count(Document.id)).where(Document.status == "parsed")) or 0
    failed_documents = db.scalar(select(func.count(Document.id)).where(Document.status == "failed")) or 0
    total_chunks = db.scalar(select(func.count(Chunk.id))) or 0
    embedded_chunks = db.scalar(select(func.count(Chunk.id)).where(Chunk.embedding.is_not(None))) or 0
    text_chunks = db.scalar(select(func.count(Chunk.id)).where(Chunk.kind == "text")) or 0
    image_chunks = db.scalar(select(func.count(Chunk.id)).where(Chunk.kind == "image")) or 0
    text_embedded_chunks = (
        db.scalar(select(func.count(Chunk.id)).where(Chunk.kind == "text", Chunk.embedding.is_not(None))) or 0
    )
    image_embedded_chunks = (
        db.scalar(select(func.count(Chunk.id)).where(Chunk.kind == "image", Chunk.embedding.is_not(None))) or 0
    )
    image_secondary_embedded_chunks = (
        db.scalar(select(func.count(Chunk.id)).where(Chunk.kind == "image", Chunk.secondary_embedding.is_not(None))) or 0
    )
    running_jobs = db.scalar(select(func.count(Job.id)).where(Job.status.in_(["queued", "running"]))) or 0
    schema_status = get_schema_status()
    ocr_service = get_ocr_service()
    ocr_diag = ocr_service.diagnostics()
    requires_reindex = bool(
        db.scalar(
            select(func.count(Chunk.id))
            .join(Document, Document.id == Chunk.document_id)
            .outerjoin(Asset, Asset.id == Chunk.asset_id)
            .where(
                Document.status == "approved",
                Chunk.kind == "image",
                or_(
                    Chunk.embedding.is_(None),
                    Chunk.secondary_embedding.is_(None),
                    Asset.id.is_(None),
                    Asset.region_type.is_(None),
                ),
            )
        )
    )
    return {
        "documents": {
            "total": total_documents,
            "approved": approved_documents,
            "parsed_waiting_approval": parsed_documents,
            "failed": failed_documents,
            "indexable": len(_indexable_documents(db)),
        },
        "chunks": {
            "total": total_chunks,
            "embedded": embedded_chunks,
            "text": text_chunks,
            "text_embedded": text_embedded_chunks,
            "image": image_chunks,
            "image_embedded": image_embedded_chunks,
            "image_secondary_embedded": image_secondary_embedded_chunks,
        },
        "models": {
            "chat_provider": settings.model_provider,
            "chat_model": settings.cloud_chat_model if settings.model_provider == "openai_compatible" else settings.chat_model,
            "text_embedding_model": embedding_service.active_model_name,
            "text_embedding_configured": embedding_service.cloud_embedding_configured,
            "image_embedding_provider": settings.image_embedding_provider,
            "image_embedding_model": embedding_service.active_image_model_name,
            "image_embedding_configured": embedding_service.image_embedding_configured,
        },
        "services": {
            "qdrant_available": VectorStore().available,
            "use_rq": settings.use_rq,
            "running_jobs": running_jobs,
            "ocr_enabled": ocr_diag["enabled"],
            "ocr_backend": ocr_diag["backend"],
            "ocr_paddle_enabled": ocr_diag["paddle_enabled"],
            "ocr_paddle_available": ocr_diag["paddle_available"],
            "ocr_cloud_enabled": ocr_diag["cloud_enabled"],
            "ocr_cloud_available": ocr_diag["cloud_available"],
            "ocr_last_error": (
                latest_ocr_warning.get("error")
                or latest_ocr_warning.get("cloud_error")
                or latest_ocr_warning.get("paddle_error")
                or latest_ocr_warning.get("quality_reason")
                or latest_ocr_warning.get("cloud_skip_reason")
            ),
            "ocr_stats": {
                "paddle_ocr_pages": latest_document_stats.get("paddle_ocr_pages", 0),
                "cloud_ocr_pages": latest_document_stats.get("cloud_ocr_pages", 0),
                "cloud_ocr_attempted_pages": latest_document_stats.get("cloud_ocr_attempted_pages", 0),
                "ocr_fallback_pages": latest_document_stats.get("ocr_fallback_pages", 0),
                "ocr_failed_pages": latest_document_stats.get("ocr_failed_pages", 0),
                "table_structured_pages": latest_document_stats.get(
                    "table_structured_pages",
                    latest_document_stats.get("structured_tables", 0),
                ),
            },
            "reranker_enabled": reranker.enabled,
            "reranker_reachable": reranker.reachable,
            "reranker_healthy": reranker.healthy,
            "reranker_model": reranker.model,
            "reranker_device": reranker.device,
            "reranker_requested_device": reranker.requested_device,
            "reranker_use_fp16": reranker.use_fp16,
            "reranker_error": reranker.error,
            "schema_version_ok": schema_status["schema_version_ok"],
            "image_schema_columns_ready": schema_status["image_schema_columns_ready"],
            "requires_reindex": requires_reindex,
            "missing_schema_columns": schema_status["missing_columns"],
        },
    }


@router.post("/index/rebuild", response_model=dict[str, list[JobOut]])
def rebuild_index(
    background_tasks: BackgroundTasks,
    _: AdminUser,
    db: Session = Depends(get_session),
) -> dict[str, list[JobOut]]:
    jobs: list[Job] = []
    for document in _indexable_documents(db):
        job = Job(document_id=document.id, job_type="index", status="queued", progress=0, message="等待重建向量索引")
        db.add(job)
        jobs.append(job)
    db.commit()
    for job in jobs:
        db.refresh(job)
        enqueue_index(background_tasks, job.document_id or "", job.id)
    return {"jobs": [JobOut.model_validate(job) for job in jobs]}
