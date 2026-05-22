from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


BACKEND_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=(PROJECT_ROOT / ".env", BACKEND_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = "GraphSearch PDF Multimodal Retrieval"
    secret_key: str = "dev-only-change-me"
    admin_username: str = "admin"
    admin_password: str = "admin123"

    database_url: str = "sqlite:///./data/app.db"
    redis_url: str = "redis://localhost:6379/0"
    qdrant_url: str | None = None
    use_rq: bool = False
    storage_dir: Path = Path("./storage")

    model_provider: Literal["ollama", "openai_compatible", "none"] = "ollama"
    ollama_base_url: str = "http://localhost:11434"
    embedding_model: str = "nomic-embed-text"
    embedding_dim: int = 768
    chat_model: str = "qwen2.5:7b-instruct"
    cloud_api_base_url: str | None = None
    cloud_api_key: str | None = None
    cloud_embedding_model: str = "text-embedding-3-small"
    cloud_chat_model: str = "gpt-4o-mini"
    cloud_timeout_seconds: int = 90
    embedding_api_base_url: str | None = None
    embedding_api_key: str | None = None
    embedding_api_model: str | None = None
    embedding_dimensions: int | None = 1024
    image_embedding_api_base_url: str | None = None
    image_embedding_api_key: str | None = None
    image_embedding_api_model: str = "tongyi-embedding-vision-flash-2026-03-06"
    image_embedding_dimensions: int | None = 1024
    image_embedding_provider: Literal["jina", "dashscope"] = "dashscope"
    vision_summary_api_base_url: str | None = None
    vision_summary_api_key: str | None = None
    vision_summary_model: str = "qwen-vl-plus"
    table_summary_enabled: bool = True
    layout_detection_backend: Literal["auto", "doclayout_yolo", "paddle", "heuristic", "none"] = "heuristic"
    table_structure_backend: Literal["auto", "paddle", "none"] = "auto"
    figure_region_vision_enabled: bool = True
    figure_region_ocr_enabled: bool = True
    process_image_reserve_ratio: float = 0.3

    reranker_enabled: bool = False
    reranker_url: str = "http://reranker:8001"
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    reranker_device: Literal["auto", "cpu", "cuda"] = "cpu"
    reranker_use_fp16: bool = False
    reranker_timeout_seconds: int = 8
    reranker_candidate_multiplier: int = 5
    reranker_max_candidates: int = 30

    ocr_backend: str = "none"
    cloud_ocr_enabled: bool = True
    cloud_ocr_model: str = "qwen-vl-plus"
    cloud_ocr_max_pages: int = 5
    ocr_lang: str = "chi_sim+eng"
    max_upload_mb: int = 2048
    cors_origins: list[str] = Field(default_factory=lambda: ["http://localhost:5173", "http://127.0.0.1:5173"])

    @field_validator("cors_origins", mode="before")
    @classmethod
    def parse_cors_origins(cls, value: str | list[str]) -> list[str]:
        if isinstance(value, str):
            return [item.strip() for item in value.split(",") if item.strip()]
        return value


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    settings.storage_dir.mkdir(parents=True, exist_ok=True)
    return settings
