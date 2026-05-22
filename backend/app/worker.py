from redis import Redis
from rq import Worker

from app.config import get_settings
from app.db import init_db


def main() -> None:
    settings = get_settings()
    init_db()
    worker = Worker(["default"], connection=Redis.from_url(settings.redis_url))
    worker.work()


if __name__ == "__main__":
    main()

