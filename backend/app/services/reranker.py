import logging
from dataclasses import dataclass, field
from typing import Any

import httpx

from app.config import get_settings

logger = logging.getLogger("reranker_client")


@dataclass
class RerankerHealth:
    enabled: bool
    reachable: bool
    healthy: bool
    model: str
    device: str
    requested_device: str
    use_fp16: bool
    error: str | None = None


@dataclass
class RerankOutcome:
    items: list[dict[str, Any]] = field(default_factory=list)
    applied: bool = False
    fallback_reason: str | None = None
    reachable: bool = False
    healthy: bool = False
    model: str | None = None
    device: str | None = None
    candidate_count: int = 0
    elapsed_ms: float = 0.0


def _base_health(settings) -> RerankerHealth:
    return RerankerHealth(
        enabled=settings.reranker_enabled,
        reachable=False,
        healthy=False,
        model=settings.reranker_model,
        device="n/a" if not settings.reranker_enabled else settings.reranker_device,
        requested_device=settings.reranker_device,
        use_fp16=settings.reranker_use_fp16,
        error=None if settings.reranker_enabled else "reranker disabled",
    )


def get_reranker_health() -> RerankerHealth:
    settings = get_settings()
    health = _base_health(settings)
    if not settings.reranker_enabled:
        return health

    url = f"{settings.reranker_url.rstrip('/')}/health"
    try:
        with httpx.Client(timeout=min(settings.reranker_timeout_seconds, 10.0)) as client:
            response = client.get(url)
            payload = response.json()
    except Exception as exc:
        health.error = str(exc)
        return health

    health.reachable = True
    health.healthy = response.status_code < 400 and bool(payload.get("loaded"))
    health.model = str(payload.get("model") or settings.reranker_model)
    health.device = str(payload.get("device") or settings.reranker_device)
    health.requested_device = str(payload.get("requested_device") or settings.reranker_device)
    health.use_fp16 = bool(payload.get("use_fp16", settings.reranker_use_fp16))
    health.error = payload.get("error")
    return health


def rerank_candidates(
    query: str,
    candidates: list[dict[str, Any]],
    top_k: int,
) -> RerankOutcome:
    settings = get_settings()
    outcome = RerankOutcome(
        applied=False,
        fallback_reason=None if settings.reranker_enabled else "disabled",
        reachable=False,
        healthy=False,
        model=settings.reranker_model,
        device=settings.reranker_device,
        candidate_count=len(candidates),
    )
    if not settings.reranker_enabled:
        return outcome
    if not candidates or top_k <= 0:
        outcome.fallback_reason = "no rerank candidates"
        return outcome

    url = f"{settings.reranker_url.rstrip('/')}/rerank"
    payload = {
        "query": query,
        "candidates": [
            {
                "chunk_id": candidate.get("chunk_id", ""),
                "content": candidate.get("content", ""),
            }
            for candidate in candidates
        ],
    }

    started = __import__("time").perf_counter()
    try:
        with httpx.Client(timeout=settings.reranker_timeout_seconds) as client:
            response = client.post(url, json=payload)
            response.raise_for_status()
            data = response.json()
    except Exception as exc:
        outcome.elapsed_ms = round((__import__("time").perf_counter() - started) * 1000, 2)
        logger.warning("Reranker call failed, falling back: %s", exc)
        outcome.fallback_reason = str(exc)
        return outcome

    results = data.get("results", [])
    if not results:
        outcome.elapsed_ms = round((__import__("time").perf_counter() - started) * 1000, 2)
        outcome.reachable = True
        outcome.healthy = True
        outcome.model = str(data.get("model") or settings.reranker_model)
        outcome.device = str(data.get("device") or settings.reranker_device)
        outcome.fallback_reason = "empty reranker response"
        return outcome

    outcome.reachable = True
    outcome.healthy = True
    outcome.model = str(data.get("model") or settings.reranker_model)
    outcome.device = str(data.get("device") or settings.reranker_device)
    outcome.elapsed_ms = round((__import__("time").perf_counter() - started) * 1000, 2)

    reranked_ids = [item["chunk_id"] for item in results if item.get("chunk_id")]
    id_order = {chunk_id: index for index, chunk_id in enumerate(reranked_ids)}
    scored_candidates = [(candidate, id_order.get(candidate.get("chunk_id", ""), len(reranked_ids))) for candidate in candidates]
    scored_candidates.sort(key=lambda item: item[1])

    rerank_score_map = {item["chunk_id"]: item["score"] for item in results if item.get("chunk_id")}
    sorted_candidates: list[dict[str, Any]] = []
    for candidate, _ in scored_candidates[:top_k]:
        chunk_id = candidate.get("chunk_id", "")
        enriched = dict(candidate)
        if chunk_id in rerank_score_map:
            enriched["rerank_score"] = float(rerank_score_map[chunk_id])
            enriched["reranked"] = True
        sorted_candidates.append(enriched)

    outcome.items = sorted_candidates
    outcome.applied = True
    return outcome
