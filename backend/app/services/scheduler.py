from fastapi import BackgroundTasks

from app.config import get_settings
from app.services.ingestion import run_index_job, run_parse_job
from app.services.task_queue import get_queue


class SchedulerEnqueueError(RuntimeError):
    pass


def _enqueue_kwargs(job_id: str) -> dict[str, int | str]:
    settings = get_settings()
    kwargs: dict[str, int | str] = {"job_id": job_id}
    if settings.rq_runtime_mode != "local":
        kwargs["job_timeout"] = settings.rq_job_timeout_seconds
    return kwargs


def enqueue_parse(background_tasks: BackgroundTasks, document_id: str, job_id: str) -> None:
    settings = get_settings()
    if settings.use_rq:
        try:
            queue = get_queue(settings)
            queue.enqueue("app.services.ingestion.run_parse_job", document_id, job_id, **_enqueue_kwargs(job_id))
            return
        except Exception as exc:
            raise SchedulerEnqueueError(f"解析任务入队失败: {exc}") from exc
    background_tasks.add_task(run_parse_job, document_id, job_id)


def enqueue_index(background_tasks: BackgroundTasks, document_id: str, job_id: str) -> None:
    settings = get_settings()
    if settings.use_rq:
        try:
            queue = get_queue(settings)
            queue.enqueue("app.services.ingestion.run_index_job", document_id, job_id, **_enqueue_kwargs(job_id))
            return
        except Exception as exc:
            raise SchedulerEnqueueError(f"入库任务入队失败: {exc}") from exc
    background_tasks.add_task(run_index_job, document_id, job_id)
