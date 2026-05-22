import logging
import os
import threading
from dataclasses import dataclass
from typing import Any

import uvicorn
from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel

app = FastAPI(title="Reranker Service", version="1.1.0")

logging.basicConfig(level=os.getenv("RERANKER_LOG_LEVEL", "INFO").upper())
logger = logging.getLogger("reranker")


def _env_flag(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _requested_device() -> str:
    return os.getenv("RERANKER_DEVICE", "cpu").strip().lower() or "cpu"


def _resolve_device() -> str:
    requested = _requested_device()
    if requested not in {"auto", "cpu", "cuda"}:
        raise RuntimeError(f"unsupported reranker device: {requested}")

    try:
        import torch
    except Exception:
        if requested == "cuda":
            raise RuntimeError("CUDA requested but torch is unavailable")
        return "cpu"

    cuda_available = bool(torch.cuda.is_available())
    if requested == "auto":
        return "cuda" if cuda_available else "cpu"
    if requested == "cuda" and not cuda_available:
        raise RuntimeError("CUDA requested but no GPU is available")
    return requested


def _batch_size() -> int:
    raw = os.getenv("RERANKER_BATCH_SIZE", "8").strip()
    try:
        return max(1, int(raw))
    except ValueError:
        return 8


@dataclass
class RuntimeState:
    model_name: str
    requested_device: str
    device: str = "cpu"
    use_fp16: bool = False
    loading: bool = False
    loaded: bool = False
    error: str | None = None
    model: Any | None = None


runtime = RuntimeState(
    model_name=os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3"),
    requested_device=_requested_device(),
)


class RerankRequest(BaseModel):
    query: str
    candidates: list[dict[str, Any]]


class RerankCandidate(BaseModel):
    chunk_id: str
    score: float


class RerankResponse(BaseModel):
    results: list[RerankCandidate]
    model: str
    device: str


def load_model() -> Any:
    runtime.device = _resolve_device()
    runtime.use_fp16 = _env_flag("RERANKER_USE_FP16", runtime.device == "cuda")
    runtime.error = None
    runtime.loading = True
    runtime.loaded = False
    runtime.model = None

    if runtime.device == "cpu":
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

    try:
        from FlagEmbedding import FlagReranker
    except Exception as exc:
        raise RuntimeError(f"failed to import FlagEmbedding: {exc}") from exc

    kwargs: dict[str, Any] = {
        "use_fp16": runtime.use_fp16,
        "devices": runtime.device,
        "batch_size": _batch_size(),
    }
    logger.info(
        "Loading reranker model %s on %s (fp16=%s)",
        runtime.model_name,
        runtime.device,
        runtime.use_fp16,
    )
    try:
        model = FlagReranker(runtime.model_name, **kwargs)
    except Exception as exc:
        raise RuntimeError(f"failed to load reranker model {runtime.model_name}: {exc}") from exc

    runtime.model = model
    runtime.loading = False
    runtime.loaded = True
    logger.info("Reranker model loaded successfully")
    return model


def ensure_model_loaded() -> Any:
    if runtime.loading:
        raise RuntimeError("reranker model is still loading")
    if runtime.model is not None and runtime.loaded:
        return runtime.model
    try:
        return load_model()
    except Exception as exc:
        runtime.error = str(exc)
        runtime.loaded = False
        runtime.model = None
        logger.exception("Failed to initialize reranker model")
        raise


@app.on_event("startup")
def startup() -> None:
    def _load_in_background() -> None:
        try:
            load_model()
        except Exception as exc:
            runtime.error = str(exc)
            runtime.loading = False
            runtime.loaded = False
            runtime.model = None
            logger.exception("Failed to initialize reranker model")

    thread = threading.Thread(target=_load_in_background, daemon=True)
    thread.start()


@app.get("/health")
def health(response: Response) -> dict[str, Any]:
    healthy = runtime.loaded and runtime.model is not None and runtime.error is None
    response.status_code = 200 if healthy else 503
    return {
        "status": "ok" if healthy else ("loading" if runtime.loading else "error"),
        "loaded": healthy,
        "loading": runtime.loading,
        "model": runtime.model_name,
        "requested_device": runtime.requested_device,
        "device": runtime.device,
        "use_fp16": runtime.use_fp16,
        "error": runtime.error,
    }


@app.post("/rerank", response_model=RerankResponse)
def rerank(req: RerankRequest) -> RerankResponse:
    if not req.query or not req.candidates:
        return RerankResponse(results=[], model=runtime.model_name, device=runtime.device)

    try:
        model = ensure_model_loaded()
    except Exception:
        raise HTTPException(status_code=503, detail=runtime.error or "Reranker model not loaded")

    pairs = [(req.query, str(candidate.get("content", ""))) for candidate in req.candidates]

    try:
        raw_scores = model.compute_score(pairs)
    except Exception as exc:
        logger.exception("Reranker compute_score failed")
        raise HTTPException(status_code=500, detail=f"Reranker scoring failed: {exc}") from exc

    if isinstance(raw_scores, (int, float)):
        score_values = [float(raw_scores)]
    else:
        score_values = [float(score) for score in raw_scores]

    results: list[RerankCandidate] = []
    for index, candidate in enumerate(req.candidates):
        chunk_id = str(candidate.get("chunk_id", ""))
        if not chunk_id:
            continue
        score = score_values[index] if index < len(score_values) else 0.0
        results.append(RerankCandidate(chunk_id=chunk_id, score=score))

    results.sort(key=lambda item: item.score, reverse=True)
    return RerankResponse(results=results, model=runtime.model_name, device=runtime.device)


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8001)
