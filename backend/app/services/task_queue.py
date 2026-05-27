from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

from redis import Redis
from rq import Queue, Worker
from rq.exceptions import NoSuchJobError
from rq.job import Job as RQJob
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import Settings, get_settings
from app.db import SessionLocal, reset_db_connections, sqlite_write_lock
from app.models import Document, Job

DELETED_DOCUMENT_STATUS = "deleted"
RUNTIME_QUEUE_DEFAULTS = {
    "docker": "default_docker",
    "local": "default_local",
    "unknown": "default",
}
RUNTIME_LABELS = {
    "docker": "容器模式",
    "local": "本机开发模式",
    "unknown": "未知模式",
}


class RuntimeConflictError(RuntimeError):
    pass


@dataclass
class ActiveWorkerInfo:
    name: str
    hostname: str
    pid: int | None
    queue_names: list[str]
    last_heartbeat: datetime | None


def get_redis_connection(settings: Settings | None = None) -> Redis:
    current = settings or get_settings()
    return Redis.from_url(current.redis_url)


def get_queue(settings: Settings | None = None) -> Queue:
    current = settings or get_settings()
    return Queue(current.rq_queue_name, connection=get_redis_connection(current))


def summarize_exc_info(exc_info: str | None) -> str | None:
    if not exc_info:
        return None
    lines = [line.strip() for line in exc_info.splitlines() if line.strip()]
    if not lines:
        return None
    for line in reversed(lines):
        if line != "Traceback (most recent call last):":
            return line
    return lines[-1]


def stale_job_message(job_type: str) -> str:
    action = "解析" if job_type == "parse" else "入库"
    return f"{action}任务长时间未推进，队列未被有效消费。请检查是否同时运行了本机开发模式和容器模式。"


def _conflicting_queue_names(runtime_mode: str) -> set[str]:
    if runtime_mode == "docker":
        return {RUNTIME_QUEUE_DEFAULTS["local"]}
    if runtime_mode == "local":
        return {RUNTIME_QUEUE_DEFAULTS["docker"]}
    return set()


def _active_workers(connection: Redis) -> list[ActiveWorkerInfo]:
    heartbeat_cutoff = datetime.utcnow() - timedelta(seconds=90)
    workers: list[ActiveWorkerInfo] = []
    for worker in Worker.all(connection=connection):
        last_heartbeat = worker.last_heartbeat
        last_heartbeat_naive = None
        if last_heartbeat is not None:
            last_heartbeat_naive = last_heartbeat.replace(tzinfo=None)
            if last_heartbeat_naive < heartbeat_cutoff:
                continue
        workers.append(
            ActiveWorkerInfo(
                name=worker.name,
                hostname=worker.hostname,
                pid=worker.pid,
                queue_names=[queue.name for queue in worker.queues],
                last_heartbeat=last_heartbeat_naive,
            )
        )
    return workers


def describe_conflicting_workers(settings: Settings | None = None) -> list[ActiveWorkerInfo]:
    current = settings or get_settings()
    if not current.use_rq:
        return []
    conflicting_queue_names = _conflicting_queue_names(current.rq_runtime_mode)
    if not conflicting_queue_names:
        return []
    try:
        workers = _active_workers(get_redis_connection(current))
    except Exception:
        return []
    return [worker for worker in workers if conflicting_queue_names.intersection(worker.queue_names)]


def assert_runtime_exclusive(settings: Settings | None = None) -> None:
    current = settings or get_settings()
    conflicts = describe_conflicting_workers(current)
    if not conflicts:
        return

    other_mode = "本机开发模式" if current.rq_runtime_mode == "docker" else "容器模式"
    details = "; ".join(
        f"{worker.hostname} pid={worker.pid or '-'} queues={','.join(worker.queue_names)}" for worker in conflicts
    )
    raise RuntimeConflictError(
        f"检测到{other_mode}的活跃 worker，当前模式禁止同时运行。"
        f"请先停止另一种模式的 worker 后再启动。冲突 worker: {details}"
    )


def _apply_job_failure(job: Job, document: Document | None, message: str) -> None:
    if job.status == "succeeded":
        return

    job.status = "failed"
    job.error_message = message
    job.updated_at = datetime.utcnow()

    if document and document.status != DELETED_DOCUMENT_STATUS:
        if job.job_type == "parse":
            document.status = "failed"
        elif job.job_type == "index" and document.status == "approved":
            document.status = "parsed"
        document.error_message = message
        document.updated_at = datetime.utcnow()


def mark_job_failed(job_id: str, message: str) -> bool:
    reset_db_connections()
    db = SessionLocal()
    try:
        with sqlite_write_lock:
            job = db.get(Job, job_id)
            if not job:
                return False
            document = db.get(Document, job.document_id) if job.document_id else None
            _apply_job_failure(job, document, message)
            db.commit()
            return True
    finally:
        db.close()


def handle_rq_exception(job, exc_type, exc_value, traceback) -> bool:
    message = summarize_exc_info(getattr(job, "exc_info", None)) or str(exc_value) or "任务执行失败"
    mark_job_failed(job.id, message)
    return True


def reconcile_pending_jobs(db: Session, *, document_id: str | None = None) -> int:
    settings = get_settings()
    if not settings.use_rq:
        return 0

    jobs_query = select(Job).where(Job.status.in_(["queued", "running"]))
    if document_id:
        jobs_query = jobs_query.where(Job.document_id == document_id)
    pending_jobs = db.execute(jobs_query).scalars().all()
    if not pending_jobs:
        return 0

    try:
        connection = get_redis_connection(settings)
    except Exception:
        return 0

    cutoff = datetime.utcnow() - timedelta(seconds=settings.rq_stalled_job_seconds)
    changed = 0

    with sqlite_write_lock:
        for job in pending_jobs:
            document = db.get(Document, job.document_id) if job.document_id else None
            try:
                rq_job = RQJob.fetch(job.id, connection=connection)
                rq_status = rq_job.get_status(refresh=True)
            except NoSuchJobError:
                if job.updated_at <= cutoff:
                    _apply_job_failure(job, document, stale_job_message(job.job_type))
                    changed += 1
                continue
            except Exception:
                continue

            if rq_status == "failed":
                message = summarize_exc_info(rq_job.exc_info) or stale_job_message(job.job_type)
                _apply_job_failure(job, document, message)
                changed += 1
            elif rq_status == "started" and job.status == "queued":
                job.status = "running"
                job.progress = max(job.progress, 10)
                if not job.message or "等待" in job.message:
                    job.message = "任务已开始执行"
                job.updated_at = datetime.utcnow()
                changed += 1

        if changed:
            db.commit()
    return changed
