import os

from redis import Redis
from rq import SimpleWorker, Worker

from app.config import get_settings
from app.db import init_db


def main() -> None:
    settings = get_settings()
    init_db()
    worker_class = SimpleWorker if os.name == "nt" else Worker
    worker = worker_class(["default"], connection=Redis.from_url(settings.redis_url))
    worker.work()


if __name__ == "__main__":
    main()
