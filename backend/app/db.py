from collections.abc import Generator
from pathlib import Path
import threading

from sqlalchemy import create_engine, event, inspect, text
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker
from sqlalchemy.pool import NullPool

from app.config import get_settings


settings = get_settings()
sqlite_write_lock = threading.RLock()

connect_args = {}
engine_kwargs = {"pool_pre_ping": True}
if settings.database_url.startswith("sqlite"):
    connect_args["check_same_thread"] = False
    connect_args["timeout"] = 30
    sqlite_path = settings.database_url.replace("sqlite:///", "", 1)
    Path(sqlite_path).parent.mkdir(parents=True, exist_ok=True)
    engine_kwargs["poolclass"] = NullPool
else:
    connect_args["prepare_threshold"] = None
    engine_kwargs["poolclass"] = NullPool

engine = create_engine(settings.database_url, connect_args=connect_args, **engine_kwargs)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, expire_on_commit=False)


if settings.database_url.startswith("sqlite"):
    @event.listens_for(engine, "connect")
    def _configure_sqlite(dbapi_connection, _connection_record) -> None:
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA busy_timeout=30000")
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()


class Base(DeclarativeBase):
    pass


def init_db() -> None:
    from app import models  # noqa: F401

    Base.metadata.create_all(bind=engine)
    _migrate_schema()


REQUIRED_SCHEMA_COLUMNS = {
    "assets": {"parent_asset_id", "region_index", "region_type", "region_summary"},
    "chunks": {
        "embedding",
        "embedding_model",
        "embedding_dim",
        "secondary_embedding",
        "secondary_embedding_model",
        "secondary_embedding_dim",
    },
}


def _migrate_schema() -> None:
    inspector = inspect(engine)
    table_names = set(inspector.get_table_names())
    chunk_columns = {column["name"] for column in inspector.get_columns("chunks")} if "chunks" in table_names else set()
    asset_columns = {column["name"] for column in inspector.get_columns("assets")} if "assets" in table_names else set()
    chunk_migrations = {
        "embedding": "ALTER TABLE chunks ADD COLUMN embedding JSON",
        "embedding_model": "ALTER TABLE chunks ADD COLUMN embedding_model VARCHAR(128)",
        "embedding_dim": "ALTER TABLE chunks ADD COLUMN embedding_dim INTEGER",
        "secondary_embedding": "ALTER TABLE chunks ADD COLUMN secondary_embedding JSON",
        "secondary_embedding_model": "ALTER TABLE chunks ADD COLUMN secondary_embedding_model VARCHAR(128)",
        "secondary_embedding_dim": "ALTER TABLE chunks ADD COLUMN secondary_embedding_dim INTEGER",
    }
    asset_migrations = {
        "parent_asset_id": "ALTER TABLE assets ADD COLUMN parent_asset_id VARCHAR(36)",
        "region_index": "ALTER TABLE assets ADD COLUMN region_index INTEGER",
        "region_type": "ALTER TABLE assets ADD COLUMN region_type VARCHAR(64)",
        "region_summary": "ALTER TABLE assets ADD COLUMN region_summary TEXT",
    }
    with engine.begin() as connection:
        for column, statement in chunk_migrations.items():
            if "chunks" in table_names and column not in chunk_columns:
                connection.execute(text(statement))
        for column, statement in asset_migrations.items():
            if "assets" in table_names and column not in asset_columns:
                connection.execute(text(statement))


def get_schema_status() -> dict[str, object]:
    inspector = inspect(engine)
    table_names = set(inspector.get_table_names())
    missing_columns: dict[str, list[str]] = {}
    for table_name, required_columns in REQUIRED_SCHEMA_COLUMNS.items():
        if table_name not in table_names:
            missing_columns[table_name] = sorted(required_columns)
            continue
        existing_columns = {column["name"] for column in inspector.get_columns(table_name)}
        missing = sorted(required_columns - existing_columns)
        if missing:
            missing_columns[table_name] = missing
    image_schema_columns_ready = not missing_columns.get("assets")
    secondary_embedding_columns_ready = not missing_columns.get("chunks")
    return {
        "schema_version_ok": not missing_columns,
        "image_schema_columns_ready": image_schema_columns_ready,
        "secondary_embedding_columns_ready": secondary_embedding_columns_ready,
        "missing_columns": missing_columns,
    }


def get_session() -> Generator[Session, None, None]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def reset_db_connections() -> None:
    engine.dispose()
