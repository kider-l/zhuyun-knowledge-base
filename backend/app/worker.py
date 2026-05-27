import os

from redis import Redis
from rq import SimpleWorker, Worker

from app.config import get_settings
from app.db import init_db
from app.services.task_queue import assert_runtime_exclusive, handle_rq_exception


def main() -> None:
    settings = get_settings()
    init_db()
    assert_runtime_exclusive(settings)
    worker_class = SimpleWorker if os.name == "nt" else Worker
    worker = worker_class([settings.rq_queue_name], connection=Redis.from_url(settings.redis_url))
    worker.push_exc_handler(handle_rq_exception)
    worker.work()


if __name__ == "__main__":
    main()
