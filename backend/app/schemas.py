from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_serializer


class UTCModel(BaseModel):
    """基类：JSON 序列化时给无时区的 datetime 自动加上 Z（UTC）标记。"""
    @field_serializer("*", when_used="json")
    @classmethod
    def _add_utc_flag(cls, value: object) -> object:
        if isinstance(value, datetime) and value.tzinfo is None:
            return value.isoformat() + "Z"
        return value


class LoginRequest(BaseModel):
    username: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


class DocumentOut(UTCModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    filename: str
    original_path: str | None
    size_bytes: int
    page_count: int
    status: str
    parse_stats: dict[str, Any]
    error_message: str | None
    created_at: datetime
    updated_at: datetime


class JobOut(UTCModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    document_id: str | None
    job_type: str
    status: str
    progress: int
    message: str | None
    error_message: str | None
    created_at: datetime
    updated_at: datetime


class AssetOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    document_id: str
    page_number: int
    kind: str
    width: int | None
    height: int | None
    bbox: dict[str, Any] | None
    parent_asset_id: str | None = None
    region_index: int | None = None
    region_type: str | None = None
    region_summary: str | None = None
    caption: str | None
    ocr_text: str | None
    url: str | None = None


class ChunkOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    document_id: str
    asset_id: str | None
    page_number: int
    kind: str
    content: str
    title_path: str | None
    bbox: dict[str, Any] | None
    chunk_metadata: dict[str, Any]
    embedding_model: str | None = None
    embedding_dim: int | None = None
    secondary_embedding_model: str | None = None
    secondary_embedding_dim: int | None = None
    approved: bool
    indexed: bool


class DocumentReview(BaseModel):
    document: DocumentOut
    jobs: list[JobOut]
    stats: dict[str, Any]
    sample_chunks: list[ChunkOut]
    chunks: list[ChunkOut] = Field(default_factory=list)
    page_assets: list[AssetOut]
    image_assets: list[AssetOut]


class LocalImportRequest(BaseModel):
    path: str
    confirm_duplicates: bool = False


class SearchRequest(BaseModel):
    query: str = Field(min_length=1)
    mode: Literal["all", "images", "process", "text"] = "all"
    top_k: int = Field(default=6, ge=1, le=30)
    document_id: str | None = None
    page_from: int | None = Field(default=None, ge=1)
    page_to: int | None = Field(default=None, ge=1)
    kind: Literal["all", "text", "image"] = "all"


class SearchResult(BaseModel):
    chunk_id: str
    document_id: str
    document_name: str
    page_number: int
    kind: str
    score: float
    snippet: str
    title_path: str | None = None
    asset_id: str | None = None
    asset_url: str | None = None
    match_reason: str | None = None
    highlight_boxes: list[dict[str, float]] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class SearchResponse(BaseModel):
    query: str
    mode: str
    results: list[SearchResult]
    diagnostics: dict[str, Any] = Field(default_factory=dict)


class AnswerResponse(BaseModel):
    query: str
    answer: str
    results: list[SearchResult]
    diagnostics: dict[str, Any] = Field(default_factory=dict)


class AnswerHistoryTurn(BaseModel):
    query: str = Field(min_length=1)
    answer: str = Field(min_length=1)


class AnswerRequest(BaseModel):
    query: str = Field(min_length=1)
    results: list[SearchResult] = Field(default_factory=list)
    history: list[AnswerHistoryTurn] = Field(default_factory=list)


class ChunkUpdateRequest(BaseModel):
    content: str | None = Field(default=None, min_length=1)
    approved: bool | None = None


class DuplicateDocumentInfo(UTCModel):
    document_id: str
    filename: str
    sha256: str
    status: str
    uploaded_at: datetime


class UploadBatchItem(BaseModel):
    filename: str
    status: str
    message: str | None = None
    document: DocumentOut | None = None
    job: JobOut | None = None
    duplicate: DuplicateDocumentInfo | None = None


class UploadBatchResponse(BaseModel):
    items: list[UploadBatchItem]


class UploadLogOut(UTCModel):
    id: str
    filename: str
    sha256: str
    source: str
    uploaded_by: str | None
    status: str
    message: str | None = None
    created_at: datetime
    updated_at: datetime
    document_id: str | None = None
    duplicate_of_document_id: str | None = None
    document_status: str | None = None
