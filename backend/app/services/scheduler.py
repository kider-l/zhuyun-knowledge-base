from fastapi import BackgroundTasks

from app.config import get_settings
from app.services.ingestion import run_index_job, run_parse_job


class SchedulerEnqueueError(RuntimeError):
    pass


def enqueue_parse(background_tasks: BackgroundTasks, document_id: str, job_id: str) -> None:
    settings = get_settings()
    if settings.use_rq:
        try:
            from redis import Redis
            from rq import Queue

            queue = Queue("default", connection=Redis.from_url(settings.redis_url))
            queue.enqueue("app.services.ingestion.run_parse_job", document_id, job_id, job_timeout=60 * 60 * 6)
            return
        except Exception as exc:
            raise SchedulerEnqueueError(f"解析任务入队失败: {exc}") from exc
    background_tasks.add_task(run_parse_job, document_id, job_id)


def enqueue_index(background_tasks: BackgroundTasks, document_id: str, job_id: str) -> None:
    settings = get_settings()
    if settings.use_rq:
        try:
            from redis import Redis
            from rq import Queue

            queue = Queue("default", connection=Redis.from_url(settings.redis_url))
            queue.enqueue("app.services.ingestion.run_index_job", document_id, job_id, job_timeout=60 * 60 * 6)
            return
        except Exception as exc:
            raise SchedulerEnqueueError(f"入库任务入队失败: {exc}") from exc
    background_tasks.add_task(run_index_job, document_id, job_id)
