from datetime import datetime

from sqlalchemy import select

from app.db import SessionLocal, reset_db_connections, sqlite_write_lock
from app.models import Document, Job
from app.services.pdf_parser import DocumentDeletedError, parse_pdf
from app.services.vector_store import index_document_chunks

DELETED_DOCUMENT_STATUS = "deleted"


def _document_deleted(db, document_id: str) -> bool:
    status_value = db.execute(select(Document.status).where(Document.id == document_id).limit(1)).scalar_one_or_none()
    return status_value is None or status_value == DELETED_DOCUMENT_STATUS


def run_parse_job(document_id: str, job_id: str | None = None) -> None:
    reset_db_connections()
    db = SessionLocal()
    try:
        document = db.get(Document, document_id)
        job = db.get(Job, job_id) if job_id else None
        if not document or document.status == DELETED_DOCUMENT_STATUS or _document_deleted(db, document_id):
            return
        with sqlite_write_lock:
            parse_pdf(db, document, job)
            if _document_deleted(db, document_id):
                return
            document.error_message = None
            if job:
                job.error_message = None
                job.updated_at = datetime.utcnow()
            db.commit()
    except DocumentDeletedError:
        db.rollback()
    except Exception as exc:
        db.rollback()
        document = db.get(Document, document_id)
        job = db.get(Job, job_id) if job_id else None
        if document and document.status != DELETED_DOCUMENT_STATUS:
            document.status = "failed"
            document.error_message = str(exc)
            document.updated_at = datetime.utcnow()
        if job:
            job.status = "failed"
            job.error_message = str(exc)
            job.updated_at = datetime.utcnow()
        with sqlite_write_lock:
            db.commit()
    finally:
        db.close()


def run_index_job(document_id: str, job_id: str | None = None) -> None:
    reset_db_connections()
    db = SessionLocal()
    try:
        document = db.get(Document, document_id)
        job = db.get(Job, job_id) if job_id else None
        if not document or document.status == DELETED_DOCUMENT_STATUS or _document_deleted(db, document_id):
            return
        with sqlite_write_lock:
            if job:
                job.status = "running"
                job.progress = 10
                job.message = "开始写入向量索引"
                job.updated_at = datetime.utcnow()
            for chunk in document.chunks:
                chunk.approved = True
            db.commit()
            indexed = index_document_chunks(db, document_id)
            if _document_deleted(db, document_id):
                VectorStore().delete_document(document_id)
                return
            document.status = "approved"
            document.error_message = None
            document.updated_at = datetime.utcnow()
            document.parse_stats = {**(document.parse_stats or {}), "indexed_chunks": indexed}
            if job:
                job.status = "succeeded"
                job.progress = 100
                job.message = f"已确认入库，索引 {indexed} 个块"
                job.error_message = None
                job.updated_at = datetime.utcnow()
            db.commit()
    except DocumentDeletedError:
        db.rollback()
        VectorStore().delete_document(document_id)
    except Exception as exc:
        db.rollback()
        document = db.get(Document, document_id)
        job = db.get(Job, job_id) if job_id else None
        if document and document.status != DELETED_DOCUMENT_STATUS:
            document.status = "parsed"
            document.error_message = str(exc)
            document.updated_at = datetime.utcnow()
        if job:
            job.status = "failed"
            job.error_message = str(exc)
            job.updated_at = datetime.utcnow()
        with sqlite_write_lock:
            db.commit()
    finally:
        db.close()
