from datetime import datetime, timedelta
from types import SimpleNamespace

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from app.db import Base
from app.models import Document, Job
from app.services import task_queue


def make_session() -> Session:
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine, autoflush=False, autocommit=False, expire_on_commit=False)()


def add_document_and_job(session: Session, *, job_type: str = "index", status: str = "queued") -> tuple[Document, Job]:
    document = Document(
        filename="sample.pdf",
        stored_path="sample.pdf",
        status="parsed" if job_type == "index" else "uploaded",
        page_count=1,
        size_bytes=128,
        parse_stats={},
    )
    session.add(document)
    session.flush()
    job = Job(
        document_id=document.id,
        job_type=job_type,
        status=status,
        progress=0,
        message="waiting",
        updated_at=datetime.utcnow() - timedelta(minutes=10),
    )
    session.add(job)
    session.commit()
    return document, job


def test_reconcile_marks_missing_pending_job_as_failed(monkeypatch) -> None:
    session = make_session()
    document, job = add_document_and_job(session)
    settings = SimpleNamespace(use_rq=True, rq_stalled_job_seconds=180)

    monkeypatch.setattr(task_queue, "get_settings", lambda: settings)
    monkeypatch.setattr(task_queue, "get_redis_connection", lambda *_args, **_kwargs: object())

    def fake_fetch(_job_id, connection):
        raise task_queue.NoSuchJobError

    monkeypatch.setattr(task_queue.RQJob, "fetch", fake_fetch)

    changed = task_queue.reconcile_pending_jobs(session, document_id=document.id)
    session.refresh(job)
    session.refresh(document)

    assert changed == 1
    assert job.status == "failed"
    assert "队列未被有效消费" in (job.error_message or "")
    assert document.status == "parsed"


def test_reconcile_marks_failed_rq_job_as_failed(monkeypatch) -> None:
    session = make_session()
    document, job = add_document_and_job(session, job_type="parse")
    settings = SimpleNamespace(use_rq=True, rq_stalled_job_seconds=180)

    monkeypatch.setattr(task_queue, "get_settings", lambda: settings)
    monkeypatch.setattr(task_queue, "get_redis_connection", lambda *_args, **_kwargs: object())

    fake_rq_job = SimpleNamespace(
        exc_info="Traceback (most recent call last):\nValueError: boom",
        get_status=lambda refresh=True: "failed",
    )
    monkeypatch.setattr(task_queue.RQJob, "fetch", lambda _job_id, connection: fake_rq_job)

    changed = task_queue.reconcile_pending_jobs(session, document_id=document.id)
    session.refresh(job)
    session.refresh(document)

    assert changed == 1
    assert job.status == "failed"
    assert job.error_message == "ValueError: boom"
    assert document.status == "failed"
