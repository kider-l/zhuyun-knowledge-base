import math
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
from uuid import uuid4

import fitz
from sqlalchemy import Select, select
from sqlalchemy.orm import Session

from app.config import get_settings
from app.services.reranker import rerank_candidates
from app.models import Asset, Chunk, Document
from app.schemas import SearchResult
from app.services.chunking import make_snippet
from app.services.embeddings import get_embedding_service


TEXT_COLLECTION = "text_chunks"
IMAGE_COLLECTION = "image_chunks"
IMAGE_TEXT_COLLECTION = "image_text_chunks"
DELETED_DOCUMENT_STATUS = "deleted"

IMAGE_INTENT_TERMS = ["图", "图纸", "疏散", "路线", "平面", "示意", "地图", "现场", "出口", "区域", "线路", "配电", "接线", "电路", "电气", "物资", "存放点"]

# 纸张资料关键词 — 用于识别笔录、登记表、扫描件等非场景照片
PAPER_RECORD_KEYWORDS = [
    # 笔录/询问
    "笔录", "记录人", "询问人", "被询问人", "被询", "问人", "谈话记录",
    # 签名/盖章
    "签名", "签字", "盖章", "签章", "印章",
    # 扫描/复印
    "扫描件", "复印件", "影印件",
    # 表格/登记
    "登记表", "申请表", "审批表", "检查表", "记录表",
    "巡查记录", "值班记录", "交接记录", "维修记录", "保养记录",
    # 档案
    "备案", "存档", "归档", "档案",
    # 文书
    "通知书", "告知书", "确认书",
]


def _is_paper_record(content: str) -> bool:
    """检测chunk内容是否为纸张资料（笔录、表格、扫描件等），而非实景照片。"""
    if not content:
        return False
    compact = _compact_text(content)
    return any(_compact_text(kw) in compact for kw in PAPER_RECORD_KEYWORDS)


@dataclass
class VectorHit:
    chunk_id: str
    score: float
    channel: str = "vector_text"


def cosine(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b)) / ((math.sqrt(sum(x * x for x in a)) or 1.0) * (math.sqrt(sum(y * y for y in b)) or 1.0))


def query_terms(query: str) -> list[str]:
    terms = re.findall(r"[A-Za-z0-9_#.+-]+|[\u4e00-\u9fff]{2,}", query)
    expanded: list[str] = []
    for term in terms:
        if term not in expanded:
            expanded.append(term)
        if re.fullmatch(r"[\u4e00-\u9fff]{4,}", term):
            for idx in range(len(term) - 1):
                piece = term[idx : idx + 2]
                if piece not in expanded:
                    expanded.append(piece)
    return expanded


def keyword_score(query: str, content: str, mode: str = "all") -> float:
    content_lower = content.lower()
    score = 0.0
    for term in query_terms(query):
        if term.lower() in content_lower:
            score += 1.0 if len(term) <= 2 else 1.8
    if query and query.lower() in content_lower:
        score += 3.0
    if mode == "images" and any(word in content for word in ["图", "图纸", "平面图", "剖面", "节点", "示意"]):
        score += 1.5
    if mode == "process" and any(word in content for word in ["流程", "步骤", "工序", "制度", "方案", "运维", "管理"]):
        score += 1.5
    return min(score / 8.0, 1.0)


def _is_image_intent(query: str) -> bool:
    return any(term in query for term in IMAGE_INTENT_TERMS)


def _image_keyword_boost(content: str, asset_kind: str, mode: str, image_intent: bool) -> float:
    boost = 0.0
    if mode in {"images", "process"} or image_intent:
        if asset_kind in {"figure_region", "whole_figure_fallback"}:
            boost += 0.12
        elif asset_kind == "figure":
            boost += 0.08
        elif asset_kind == "embedded_image":
            boost += 0.06
        elif asset_kind == "page":
            boost += 0.03
    if image_intent and any(term in content for term in ["疏散", "路线", "平面", "出口", "现场", "图", "图纸", "区域", "线路", "配电", "接线", "电路", "电气", "物资", "存放点"]):
        boost += 0.08
    return boost


def _candidate_limit(top_k: int, base_multiplier: int, hard_limit: int) -> int:
    return min(max(top_k, top_k * base_multiplier), hard_limit)


def _process_image_ratio(query: str, default_ratio: float) -> float:
    if _is_image_intent(query):
        return 0.88
    return max(default_ratio, 0.45)


def match_reason(query: str, content: str, semantic_score: float | None = None) -> str:
    terms = [term for term in query_terms(query) if term.lower() in content.lower()]
    parts = []
    if semantic_score is not None:
        parts.append(f"语义相似度 {semantic_score:.2f}")
    if terms:
        parts.append("命中关键词：" + "、".join(terms[:6]))
    if not parts:
        parts.append("与查询存在弱相关，建议换更具体的关键词")
    return "；".join(parts)


def _normalize_box(rect: fitz.Rect, page_rect: fitz.Rect) -> dict[str, float] | None:
    if not page_rect.width or not page_rect.height or rect.width <= 0 or rect.height <= 0:
        return None
    return {
        "x": max(0.0, min(1.0, rect.x0 / page_rect.width)),
        "y": max(0.0, min(1.0, rect.y0 / page_rect.height)),
        "width": max(0.0, min(1.0, rect.width / page_rect.width)),
        "height": max(0.0, min(1.0, rect.height / page_rect.height)),
    }


def _bbox_to_rect(bbox: dict | None) -> fitz.Rect | None:
    if not bbox:
        return None
    try:
        return fitz.Rect(float(bbox["x0"]), float(bbox["y0"]), float(bbox["x1"]), float(bbox["y1"]))
    except (KeyError, TypeError, ValueError):
        return None


def _compact_text(value: str) -> str:
    return re.sub(r"\s+", "", value).lower()


def _document_deleted(db: Session, document_id: str) -> bool:
    status_value = db.execute(select(Document.status).where(Document.id == document_id).limit(1)).scalar_one_or_none()
    return status_value is None or status_value == DELETED_DOCUMENT_STATUS


def _content_terms(*values: str) -> list[str]:
    terms: list[str] = []
    for value in values:
        for term in query_terms(value):
            cleaned = term.strip()
            if len(cleaned) >= 2 and cleaned not in terms:
                terms.append(cleaned)
    return terms


def _line_rects_from_words(page: fitz.Page) -> list[tuple[str, fitz.Rect]]:
    words = page.get_text("words") or []
    rows: dict[tuple[int, int, int], list[tuple[float, float, float, float, str]]] = {}
    for item in words:
        if len(item) < 8:
            continue
        x0, y0, x1, y1, text, block_no, line_no, _word_no = item[:8]
        key = (int(block_no), int(line_no), round(float(y0)))
        rows.setdefault(key, []).append((float(x0), float(y0), float(x1), float(y1), str(text)))

    lines: list[tuple[str, fitz.Rect]] = []
    for row in rows.values():
        ordered = sorted(row, key=lambda word: word[0])
        text = "".join(word[4] for word in ordered)
        rect = fitz.Rect(
            min(word[0] for word in ordered),
            min(word[1] for word in ordered),
            max(word[2] for word in ordered),
            max(word[3] for word in ordered),
        )
        if text.strip() and rect.width > 0 and rect.height > 0:
            lines.append((text, rect))
    return sorted(lines, key=lambda item: (item[1].y0, item[1].x0))


def _text_line_highlights(page: fitz.Page, page_rect: fitz.Rect, chunk: Chunk, query: str) -> list[dict[str, float]]:
    lines = _line_rects_from_words(page)
    if not lines:
        return []

    snippet = make_snippet(chunk.content, query, limit=180)
    compact_snippet = _compact_text(snippet)
    content_terms = _content_terms(query, snippet, chunk.content[:500])
    scored: list[tuple[float, fitz.Rect]] = []
    for line_text, rect in lines:
        compact_line = _compact_text(line_text)
        if not compact_line:
            continue
        score = 0.0
        if compact_line and compact_line in compact_snippet:
            score += min(len(compact_line) / 20, 4.0)
        if compact_snippet and compact_snippet in compact_line:
            score += 5.0
        for term in content_terms[:16]:
            if term.lower() in compact_line:
                score += 1.0 if len(term) <= 2 else 1.6
        if score > 0:
            scored.append((score, rect))

    if not scored:
        return []

    scored.sort(key=lambda item: item[0], reverse=True)
    selected = sorted([rect for _score, rect in scored[:5]], key=lambda rect: (rect.y0, rect.x0))
    boxes = [_normalize_box(rect + (-1, -1, 1, 1), page_rect) for rect in selected]
    return [box for box in boxes if box]


def _metadata_ocr_highlights(chunk: Chunk, query: str) -> list[dict[str, float]]:
    lines = (chunk.chunk_metadata or {}).get("ocr_lines")
    if not isinstance(lines, list):
        return []
    snippet = make_snippet(chunk.content, query, limit=180)
    compact_snippet = _compact_text(snippet)
    terms = _content_terms(query, snippet, chunk.content[:500])
    scored: list[tuple[float, int, dict]] = []
    for index, line in enumerate(lines):
        if not isinstance(line, dict):
            continue
        text = str(line.get("text") or "")
        compact_line = _compact_text(text)
        if not compact_line:
            continue
        score = 0.0
        if compact_line in compact_snippet:
            score += min(len(compact_line) / 10, 5.0)
        for term in terms[:16]:
            if term.lower() in compact_line:
                score += 1.0 if len(term) <= 2 else 1.6
        if score > 0:
            scored.append((score, index, line))
    if not scored and lines:
        scored = [(1.0, index, line) for index, line in enumerate(lines[:5]) if isinstance(line, dict)]
    scored.sort(key=lambda item: item[0], reverse=True)
    selected = sorted(scored[:8], key=lambda item: item[1])
    boxes: list[dict[str, float]] = []
    for _score, _index, line in selected:
        try:
            x = float(line.get("x", 0))
            y = float(line.get("y", 0))
            width = float(line.get("width", 0))
            height = float(line.get("height", 0))
        except (TypeError, ValueError):
            continue
        if width > 0 and height > 0:
            boxes.append({"x": x, "y": y, "width": width, "height": height})
    return boxes


def _dedupe_boxes(boxes: list[dict[str, float]], limit: int = 8) -> list[dict[str, float]]:
    deduped: list[dict[str, float]] = []
    seen: set[tuple[float, float, float, float]] = set()
    for box in boxes:
        key = (
            round(float(box.get("x", 0.0)), 3),
            round(float(box.get("y", 0.0)), 3),
            round(float(box.get("width", 0.0)), 3),
            round(float(box.get("height", 0.0)), 3),
        )
        if key in seen:
            continue
        if key[2] <= 0 or key[3] <= 0:
            continue
        seen.add(key)
        deduped.append(box)
        if len(deduped) >= limit:
            break
    return deduped


def _approximate_ocr_text_bands(preview_asset: Asset | None, chunk: Chunk, query: str) -> list[dict[str, float]]:
    page_ocr_text = str(getattr(preview_asset, "ocr_text", None) or "").strip()
    if not page_ocr_text:
        return []

    compact_page = _compact_text(page_ocr_text)
    if not compact_page:
        return []

    snippet = make_snippet(chunk.content, query, limit=220)
    candidates: list[str] = []
    for value in [snippet, chunk.content[:600], query]:
        compact_value = _compact_text(value)
        if len(compact_value) >= 6 and compact_value not in candidates:
            candidates.append(compact_value)
        for term in _content_terms(value):
            compact_term = _compact_text(term)
            if len(compact_term) >= 4 and compact_term not in candidates:
                candidates.append(compact_term)

    best_offset: int | None = None
    best_length = 0
    for candidate in candidates:
        offset = compact_page.find(candidate)
        if offset >= 0 and len(candidate) > best_length:
            best_offset = offset
            best_length = len(candidate)

    if best_offset is None:
        terms = [_compact_text(term) for term in _content_terms(query, snippet, chunk.content[:600]) if len(_compact_text(term)) >= 2]
        matches = [compact_page.find(term) for term in terms[:12] if compact_page.find(term) >= 0]
        if matches:
            best_offset = min(matches)
            best_length = max(12, min(48, len(compact_page) // 10))

    if best_offset is None:
        return []

    page_length = max(len(compact_page), 1)
    start_ratio = best_offset / page_length
    span_ratio = min(0.22, max(0.08, best_length / page_length * 2.4))
    top = min(0.84, max(0.08, 0.08 + start_ratio * 0.78))
    bottom = min(0.96, top + span_ratio)
    estimated_lines = max(1, min(4, math.ceil(best_length / 28)))
    line_height = max(0.035, (bottom - top) / estimated_lines)

    boxes: list[dict[str, float]] = []
    for index in range(estimated_lines):
        y = min(0.96 - line_height, top + index * line_height)
        boxes.append({
            "x": 0.06,
            "y": y,
            "width": 0.88,
            "height": max(0.028, line_height - 0.006),
        })
    return _dedupe_boxes(boxes)


def _highlight_boxes(document: Document, chunk: Chunk, preview_asset: Asset | None, query: str) -> tuple[list[dict[str, float]], str]:
    if not preview_asset:
        return [], "none"
    if preview_asset.kind != "page" and preview_asset.id == chunk.asset_id:
        return [{"x": 0.0, "y": 0.0, "width": 1.0, "height": 1.0}], "asset"

    path = Path(document.stored_path)
    if not path.exists() or chunk.page_number < 1:
        return [], "none"

    try:
        with fitz.open(path) as pdf:
            if chunk.page_number > pdf.page_count:
                return [], "none"
            page = pdf.load_page(chunk.page_number - 1)
            page_rect = page.rect
            rect = _bbox_to_rect(chunk.bbox)
            if rect:
                box = _normalize_box(rect, page_rect)
                return ([box] if box else []), ("bbox" if box else "none")

            ocr_boxes = _metadata_ocr_highlights(chunk, query)
            if ocr_boxes:
                return _dedupe_boxes(ocr_boxes), "ocr_lines"

            line_boxes = _text_line_highlights(page, page_rect, chunk, query)
            if line_boxes:
                return _dedupe_boxes(line_boxes), "text_lines"

            boxes: list[dict[str, float]] = []
            candidates = _content_terms(query, chunk.content[:500])
            for term in candidates[:8]:
                for hit in page.search_for(term)[:3]:
                    box = _normalize_box(hit, page_rect)
                    if box:
                        boxes.append(box)
                if len(boxes) >= 8:
                    break
            if boxes:
                return _dedupe_boxes(boxes), "page_search"

            approximate_boxes = _approximate_ocr_text_bands(preview_asset, chunk, query)
            if approximate_boxes:
                return approximate_boxes, "approximate"
            return [], "none"
    except Exception:
        approximate_boxes = _approximate_ocr_text_bands(preview_asset, chunk, query)
        return (approximate_boxes, "approximate") if approximate_boxes else ([], "none")


class VectorStore:
    def __init__(self) -> None:
        self.settings = get_settings()
        self.embedding_service = get_embedding_service()
        self.client = None
        self.last_search_diagnostics: dict[str, object] = {}
        if self.settings.qdrant_url:
            try:
                from qdrant_client import QdrantClient

                self.client = QdrantClient(url=self.settings.qdrant_url, timeout=5)
                self.client.get_collections()
            except Exception:
                self.client = None

    @property
    def available(self) -> bool:
        return self.client is not None

    def ensure_collection(self, name: str, vector_size: int) -> None:
        if not self.client:
            return
        from qdrant_client import models

        collections = {item.name for item in self.client.get_collections().collections}
        if name not in collections:
            self.client.create_collection(
                collection_name=name,
                vectors_config=models.VectorParams(size=vector_size, distance=models.Distance.COSINE),
            )

    def index_chunks(self, db: Session, chunks: Iterable[Chunk]) -> int:
        chunk_list = list(chunks)
        if not chunk_list:
            return 0
        indexed = 0
        for chunk in chunk_list:
            if _document_deleted(db, chunk.document_id):
                return indexed
            asset = db.get(Asset, chunk.asset_id) if chunk.kind == "image" and chunk.asset_id else None
            secondary_vector: list[float] | None = None
            secondary_model_name: str | None = None
            if asset and asset.path:
                vector = self.embedding_service.embed_image_file(asset.path, chunk.content)
                model_name = self.embedding_service.active_image_model_name
                secondary_vector = self.embedding_service.embed(chunk.content)
                secondary_model_name = self.embedding_service.active_model_name
            elif chunk.kind == "image":
                vector = self.embedding_service.embed_image_query(chunk.content)
                model_name = self.embedding_service.active_image_model_name
                secondary_vector = self.embedding_service.embed(chunk.content)
                secondary_model_name = self.embedding_service.active_model_name
            else:
                vector = self.embedding_service.embed(chunk.content)
                model_name = self.embedding_service.active_model_name
            chunk.embedding = vector
            chunk.embedding_model = model_name
            chunk.embedding_dim = len(vector)
            if secondary_vector is not None:
                chunk.secondary_embedding = secondary_vector
                chunk.secondary_embedding_model = secondary_model_name
                chunk.secondary_embedding_dim = len(secondary_vector)
            else:
                chunk.secondary_embedding = None
                chunk.secondary_embedding_model = None
                chunk.secondary_embedding_dim = None
            chunk.indexed = True
            chunk.qdrant_point_id = chunk.qdrant_point_id or str(uuid4())
            if not self.client:
                indexed += 1
                continue

            from qdrant_client import models

            collection = IMAGE_COLLECTION if chunk.kind == "image" else TEXT_COLLECTION
            self.ensure_collection(collection, len(vector))
            payload = {
                "chunk_id": chunk.id,
                "document_id": chunk.document_id,
                "asset_id": chunk.asset_id,
                "page_number": chunk.page_number,
                "kind": chunk.kind,
                "content": chunk.content[:1200],
                "title_path": chunk.title_path,
            }
            self.client.upsert(
                collection_name=collection,
                points=[models.PointStruct(id=chunk.qdrant_point_id, vector=vector, payload=payload)],
            )
            if chunk.kind == "image" and secondary_vector:
                self.ensure_collection(IMAGE_TEXT_COLLECTION, len(secondary_vector))
                self.client.upsert(
                    collection_name=IMAGE_TEXT_COLLECTION,
                    points=[models.PointStruct(id=chunk.qdrant_point_id, vector=secondary_vector, payload=payload)],
                )
            indexed += 1
        db.commit()
        return indexed

    def delete_document(self, document_id: str) -> None:
        if not self.client:
            return
        from qdrant_client import models

        selector = models.FilterSelector(
            filter=models.Filter(
                must=[models.FieldCondition(key="document_id", match=models.MatchValue(value=document_id))]
            )
        )
        for collection in [TEXT_COLLECTION, IMAGE_COLLECTION, IMAGE_TEXT_COLLECTION]:
            try:
                self.client.delete(collection_name=collection, points_selector=selector)
            except Exception:
                continue

    def _qdrant_filter(
        self,
        *,
        document_id: str | None = None,
        page_from: int | None = None,
        page_to: int | None = None,
        kind: str | None = None,
    ):
        if not self.client:
            return None
        from qdrant_client import models

        conditions = []
        if document_id:
            conditions.append(models.FieldCondition(key="document_id", match=models.MatchValue(value=document_id)))
        if kind in {"text", "image"}:
            conditions.append(models.FieldCondition(key="kind", match=models.MatchValue(value=kind)))
        if page_from is not None or page_to is not None:
            range_kwargs = {}
            if page_from is not None:
                range_kwargs["gte"] = page_from
            if page_to is not None:
                range_kwargs["lte"] = page_to
            conditions.append(models.FieldCondition(key="page_number", range=models.Range(**range_kwargs)))
        return models.Filter(must=conditions) if conditions else None

    def search_qdrant(
        self,
        query: str,
        mode: str,
        top_k: int,
        document_id: str | None = None,
        page_from: int | None = None,
        page_to: int | None = None,
        kind: str | None = None,
    ) -> list[VectorHit]:
        if not self.client:
            return []
        from qdrant_client import models

        collections = []
        if mode in {"all", "text", "process"} and kind != "image":
            collections.append(TEXT_COLLECTION)
        if mode in {"all", "images", "process"} and kind != "text":
            collections.append(IMAGE_COLLECTION)
            collections.append(IMAGE_TEXT_COLLECTION)
        hits: list[VectorHit] = []
        query_filter = self._qdrant_filter(document_id=document_id, page_from=page_from, page_to=page_to, kind=kind)
        for collection in collections:
            vector = self.embedding_service.embed_image_query(query) if collection == IMAGE_COLLECTION else self.embedding_service.embed(query)
            try:
                self.ensure_collection(collection, len(vector))
                rerank_allowed = self.settings.reranker_enabled and mode == "text"
                base_multiplier = self.settings.reranker_candidate_multiplier if rerank_allowed else (8 if mode == "images" else 5)
                candidate_limit = _candidate_limit(
                    top_k,
                    base_multiplier,
                    self.settings.reranker_max_candidates,
                )
                results = self.client.search(
                    collection_name=collection,
                    query_vector=vector,
                    query_filter=query_filter,
                    limit=max(top_k, candidate_limit),
                )
            except Exception:
                continue
            for item in results:
                payload = item.payload or {}
                chunk_id = str(payload.get("chunk_id", ""))
                if chunk_id:
                    channel = "vector_image" if collection == IMAGE_COLLECTION else "vector_text"
                    hits.append(VectorHit(chunk_id=chunk_id, score=float(item.score), channel=channel))
        return hits

    def search_fallback(
        self,
        db: Session,
        query: str,
        mode: str,
        top_k: int,
        document_id: str | None = None,
        page_from: int | None = None,
        page_to: int | None = None,
        kind: str | None = None,
    ) -> list[VectorHit]:
        text_query_vector: list[float] | None = None
        image_query_vector: list[float] | None = None
        stmt: Select[tuple[Chunk]] = (
            select(Chunk)
            .join(Document, Document.id == Chunk.document_id)
            .where(Document.status == "approved", Chunk.approved.is_(True))
        )
        if document_id:
            stmt = stmt.where(Chunk.document_id == document_id)
        if page_from is not None:
            stmt = stmt.where(Chunk.page_number >= page_from)
        if page_to is not None:
            stmt = stmt.where(Chunk.page_number <= page_to)
        if kind in {"text", "image"}:
            stmt = stmt.where(Chunk.kind == kind)
        if mode == "images":
            stmt = stmt.where(Chunk.kind == "image")
        elif mode == "text":
            stmt = stmt.where(Chunk.kind == "text")
        chunks = db.execute(stmt).scalars().all()
        hits: list[VectorHit] = []
        for chunk in chunks:
            if chunk.kind == "image":
                if image_query_vector is None:
                    image_query_vector = self.embedding_service.embed_image_query(query)
                primary_vector = image_query_vector
                if text_query_vector is None:
                    text_query_vector = self.embedding_service.embed(query)
            else:
                if text_query_vector is None:
                    text_query_vector = self.embedding_service.embed(query)
                primary_vector = text_query_vector
            content_vector = chunk.embedding if chunk.embedding and len(chunk.embedding) == len(primary_vector) else None
            semantic = cosine(primary_vector, content_vector) if content_vector else 0.0
            secondary_semantic = 0.0
            if chunk.kind == "image" and text_query_vector is not None:
                secondary_vector = (
                    chunk.secondary_embedding
                    if chunk.secondary_embedding and len(chunk.secondary_embedding) == len(text_query_vector)
                    else None
                )
                secondary_semantic = cosine(text_query_vector, secondary_vector) if secondary_vector else 0.0
            semantic_score = max(semantic, secondary_semantic)
            if semantic_score <= 0:
                continue
            channel = "vector_image" if chunk.kind == "image" and semantic >= secondary_semantic else "vector_text"
            hits.append(VectorHit(chunk_id=chunk.id, score=semantic_score, channel=channel))
        rerank_allowed = self.settings.reranker_enabled and mode == "text"
        base_multiplier = self.settings.reranker_candidate_multiplier if rerank_allowed else (8 if mode == "images" else 5)
        candidate_limit = _candidate_limit(top_k, base_multiplier, self.settings.reranker_max_candidates)
        return sorted(hits, key=lambda item: item.score, reverse=True)[:candidate_limit]

    def search_keyword(
        self,
        db: Session,
        query: str,
        mode: str,
        top_k: int,
        document_id: str | None = None,
        page_from: int | None = None,
        page_to: int | None = None,
        kind: str | None = None,
    ) -> list[VectorHit]:
        stmt: Select[tuple[Chunk]] = (
            select(Chunk)
            .join(Document, Document.id == Chunk.document_id)
            .where(Document.status == "approved", Chunk.approved.is_(True))
        )
        if document_id:
            stmt = stmt.where(Chunk.document_id == document_id)
        if page_from is not None:
            stmt = stmt.where(Chunk.page_number >= page_from)
        if page_to is not None:
            stmt = stmt.where(Chunk.page_number <= page_to)
        if kind in {"text", "image"}:
            stmt = stmt.where(Chunk.kind == kind)
        if mode == "images":
            stmt = stmt.where(Chunk.kind == "image")
        elif mode == "text":
            stmt = stmt.where(Chunk.kind == "text")
        chunks = db.execute(stmt).scalars().all()
        image_intent = _is_image_intent(query)
        hits: list[VectorHit] = []
        for chunk in chunks:
            lexical = keyword_score(query, chunk.content, mode)
            if lexical <= 0:
                continue
            asset_kind = str((chunk.chunk_metadata or {}).get("asset_kind") or chunk.kind)
            score = min(1.0, lexical + _image_keyword_boost(chunk.content, asset_kind, mode, image_intent))
            hits.append(VectorHit(chunk_id=chunk.id, score=score, channel="keyword"))
        candidate_limit = _candidate_limit(top_k, 10 if mode == "images" else 6, max(self.settings.reranker_max_candidates, top_k * 6))
        return sorted(hits, key=lambda item: item.score, reverse=True)[:candidate_limit]

    def search(
        self,
        db: Session,
        query: str,
        mode: str = "all",
        top_k: int = 8,
        document_id: str | None = None,
        page_from: int | None = None,
        page_to: int | None = None,
        kind: str | None = None,
    ) -> list[SearchResult]:
        started = time.perf_counter()
        image_intent = _is_image_intent(query)
        self.last_search_diagnostics = {
            "reranker_enabled": self.settings.reranker_enabled,
            "reranker_applied": False,
            "reranker_reachable": False,
            "reranker_healthy": False,
            "reranker_model": self.settings.reranker_model,
            "reranker_device": self.settings.reranker_device,
            "reranker_requested_device": self.settings.reranker_device,
            "reranker_fallback_reason": None,
            "reranker_candidate_count": 0,
            "candidate_count_before_rerank": 0,
            "text_hits_count": 0,
            "image_hits_count": 0,
            "vector_candidate_count": 0,
            "keyword_candidate_count": 0,
            "image_result_type_breakdown": {},
            "fusion_strategy": "hybrid_candidate_fusion",
            "vector_search_ms": 0.0,
            "rerank_ms": 0.0,
            "result_build_ms": 0.0,
            "total_search_ms": 0.0,
        }
        search_started = time.perf_counter()
        vector_hits = self.search_qdrant(query, mode, top_k, document_id, page_from, page_to, kind) if self.client else []
        if not vector_hits:
            vector_hits = self.search_fallback(db, query, mode, top_k, document_id, page_from, page_to, kind)
        keyword_hits = self.search_keyword(db, query, mode, top_k, document_id, page_from, page_to, kind)
        self.last_search_diagnostics["vector_search_ms"] = round((time.perf_counter() - search_started) * 1000, 2)
        self.last_search_diagnostics["vector_candidate_count"] = len(vector_hits)
        self.last_search_diagnostics["keyword_candidate_count"] = len(keyword_hits)

        all_hits = vector_hits + keyword_hits
        if not all_hits:
            return []
        merged: dict[str, float] = {}
        vector_score_map: dict[str, float] = {}
        keyword_score_map: dict[str, float] = {}
        retrieval_channels: dict[str, set[str]] = {}
        for hit in all_hits:
            retrieval_channels.setdefault(hit.chunk_id, set()).add(hit.channel)
            merged[hit.chunk_id] = max(merged.get(hit.chunk_id, 0.0), hit.score)
            if hit.channel == "keyword":
                keyword_score_map[hit.chunk_id] = max(keyword_score_map.get(hit.chunk_id, 0.0), hit.score)
            else:
                vector_score_map[hit.chunk_id] = max(vector_score_map.get(hit.chunk_id, 0.0), hit.score)

        ordered_chunk_ids = [chunk_id for chunk_id, _score in sorted(merged.items(), key=lambda item: item[1], reverse=True)]
        chunks_by_id = {
            chunk.id: chunk
            for chunk in db.execute(select(Chunk).where(Chunk.id.in_(ordered_chunk_ids))).scalars().all()
        }
        chunks = [chunks_by_id[chunk_id] for chunk_id in ordered_chunk_ids if chunk_id in chunks_by_id]
        text_chunks = [chunk for chunk in chunks if chunk.kind == "text"]
        image_chunks = [chunk for chunk in chunks if chunk.kind == "image"]
        self.last_search_diagnostics["candidate_count_before_rerank"] = len(chunks)
        self.last_search_diagnostics["text_hits_count"] = len(text_chunks)
        self.last_search_diagnostics["image_hits_count"] = len(image_chunks)
        documents = {doc.id: doc for doc in db.execute(select(Document).where(Document.id.in_({c.document_id for c in chunks}))).scalars()}
        assets = {
            asset.id: asset
            for asset in db.execute(select(Asset).where(Asset.id.in_({c.asset_id for c in chunks if c.asset_id}))).scalars()
        }
        page_assets = {
            (asset.document_id, asset.page_number): asset
            for asset in db.execute(
                select(Asset).where(
                    Asset.kind == "page",
                    Asset.document_id.in_({c.document_id for c in chunks}),
                    Asset.page_number.in_({c.page_number for c in chunks}),
                )
            ).scalars()
        }

        rerank_score_map: dict[str, float] = {}
        reranked_chunk_ids: set[str] = set()
        lexical_score_map = {chunk.id: keyword_score(query, chunk.content, mode) for chunk in chunks}
        rerank_allowed = self.settings.reranker_enabled and mode == "text"
        rerank_candidate_chunks: list[Chunk] = []
        if rerank_allowed:
            rerank_candidate_chunks = text_chunks[: _candidate_limit(top_k, 5, self.settings.reranker_max_candidates)]

        if rerank_candidate_chunks:
            rerank_started = time.perf_counter()
            rerank_outcome = rerank_candidates(
                query,
                [{"chunk_id": chunk.id, "content": chunk.content} for chunk in rerank_candidate_chunks],
                min(top_k * 5, len(rerank_candidate_chunks), self.settings.reranker_max_candidates),
            )
            rerank_elapsed_ms = rerank_outcome.elapsed_ms or round((time.perf_counter() - rerank_started) * 1000, 2)
        else:
            rerank_outcome = rerank_candidates(query, [], 0)
            rerank_elapsed_ms = 0.0
        self.last_search_diagnostics.update(
            {
                "reranker_applied": rerank_outcome.applied,
                "reranker_reachable": rerank_outcome.reachable,
                "reranker_healthy": rerank_outcome.healthy,
                "reranker_model": rerank_outcome.model or self.settings.reranker_model,
                "reranker_device": rerank_outcome.device or self.settings.reranker_device,
                "reranker_fallback_reason": rerank_outcome.fallback_reason,
                "reranker_candidate_count": rerank_outcome.candidate_count,
                "rerank_ms": rerank_elapsed_ms,
            }
        )
        if rerank_outcome.applied:
            reranked_ids = [item["chunk_id"] for item in rerank_outcome.items if item.get("chunk_id")]
            reranked_chunk_ids = set(reranked_ids)
            rerank_score_map = {
                item["chunk_id"]: float(item.get("rerank_score", 0.0))
                for item in rerank_outcome.items
                if item.get("chunk_id")
            }
            reranked_chunks = [chunks_by_id[chunk_id] for chunk_id in reranked_ids if chunk_id in chunks_by_id]
            remaining_chunks = [chunk for chunk in chunks if chunk.id not in reranked_chunk_ids]
            chunks = reranked_chunks + remaining_chunks
        elif mode == "text":
            chunks = chunks[: _candidate_limit(top_k, 4, self.settings.reranker_max_candidates)]

        if mode == "process" or (mode == "all" and image_intent):
            image_ratio = _process_image_ratio(query, self.settings.process_image_reserve_ratio)
            image_slots = min(len(image_chunks), max(1, int(math.ceil(top_k * image_ratio))))
            text_slots = max(0, top_k - image_slots)
            ordered_text_chunks = [chunk for chunk in chunks if chunk.kind == "text"]
            ordered_image_chunks = [chunk for chunk in chunks if chunk.kind == "image"]
            # 优先保证精准切图（figure_region等）入选，再以全页截图补位
            PRECISE_PRIORITY = {"figure_region": 0, "embedded_image": 1, "figure": 2, "table": 3, "whole_figure_fallback": 4, "page": 5}
            ordered_image_chunks.sort(
                key=lambda c: (
                    PRECISE_PRIORITY.get(str((c.chunk_metadata or {}).get("asset_kind", "")), 9),
                    -(vector_score_map.get(c.id, 0.0) + keyword_score_map.get(c.id, 0.0)),
                ),
            )
            selected = ordered_text_chunks[:text_slots] + ordered_image_chunks[:image_slots]
            if len(selected) < top_k:
                selected_ids = {chunk.id for chunk in selected}
                for chunk in chunks:
                    if chunk.id in selected_ids:
                        continue
                    selected.append(chunk)
                    if len(selected) >= top_k:
                        break
            chunks = selected
            strategy_label = "image_intent_dual_channel" if mode == "all" else "process_dual_channel"
            self.last_search_diagnostics["fusion_strategy"] = f"{strategy_label}_reserve_{int(image_ratio * 100)}pct_images"
        elif mode == "images":
            chunks = chunks[: _candidate_limit(top_k, 8, max(self.settings.reranker_max_candidates, top_k * 8))]
            self.last_search_diagnostics["fusion_strategy"] = "image_hybrid_fusion"
        elif mode == "text":
            self.last_search_diagnostics["fusion_strategy"] = "text_rerank_only"

        result_build_started = time.perf_counter()
        results: list[SearchResult] = []
        for chunk in chunks:
            doc = documents.get(chunk.document_id)
            if not doc:
                continue
            if document_id and chunk.document_id != document_id:
                continue
            if page_from is not None and chunk.page_number < page_from:
                continue
            if page_to is not None and chunk.page_number > page_to:
                continue
            if kind in {"text", "image"} and chunk.kind != kind:
                continue
            # SCENE_IMAGE 与 DOC_IMAGE 均保留，前端按 image_class 分组展示
            vector_score = vector_score_map.get(chunk.id, 0.0)
            lexical = lexical_score_map.get(chunk.id, 0.0)
            keyword_score_value = keyword_score_map.get(chunk.id, 0.0)
            rerank_score = rerank_score_map.get(chunk.id)
            asset_kind = str((chunk.chunk_metadata or {}).get("asset_kind") or chunk.kind)
            asset_priority = _image_keyword_boost(chunk.content, asset_kind, mode, image_intent) if chunk.kind == "image" else 0.0
            if rerank_score is not None:
                score = rerank_score + keyword_score_value * 0.12 + asset_priority * 0.2
            elif mode == "images":
                score = vector_score * 0.55 + keyword_score_value * 0.45 + asset_priority
            elif mode == "process":
                score = vector_score * 0.58 + keyword_score_value * 0.42 + asset_priority * 0.6
            else:
                image_boost_coeff = 0.45 if image_intent else 0.25
                score = vector_score * 0.72 + keyword_score_value * 0.28 + asset_priority * image_boost_coeff
            asset = assets.get(chunk.asset_id) if chunk.asset_id else None
            # 切图文件缺失时回退到页面级截图
            if asset and not Path(asset.path).exists():
                asset = None
            page_asset = page_assets.get((chunk.document_id, chunk.page_number))
            preview_asset = asset or page_asset
            boxes, highlight_precision = _highlight_boxes(doc, chunk, preview_asset, query)
            match_score = rerank_score if rerank_score is not None else max(vector_score, keyword_score_value)
            results.append(
                SearchResult(
                    chunk_id=chunk.id,
                    document_id=chunk.document_id,
                    document_name=doc.filename,
                    page_number=chunk.page_number,
                    kind=chunk.kind,
                    score=round(score, 4),
                    snippet=make_snippet(chunk.content, query),
                    title_path=chunk.title_path,
                    asset_id=preview_asset.id if preview_asset else chunk.asset_id,
                    asset_url=f"/api/assets/{preview_asset.id}" if preview_asset else None,
                    match_reason=match_reason(query, chunk.content, match_score),
                    highlight_boxes=boxes,
                    metadata={
                        **(chunk.chunk_metadata or {}),
                        "match_reason": match_reason(query, chunk.content, match_score),
                        "embedding_model": chunk.embedding_model,
                        "has_embedding": bool(chunk.embedding),
                        "highlight_available": bool(boxes),
                        "highlight_precision": highlight_precision,
                        "vector_score": round(vector_score, 4),
                        "keyword_score": round(keyword_score_value, 4),
                        "rerank_score": round(rerank_score, 4) if rerank_score is not None else None,
                        "final_score": round(score, 4),
                        "reranked": chunk.id in reranked_chunk_ids,
                        "asset_kind": asset_kind,
                        "layout_source": (chunk.chunk_metadata or {}).get("layout_source"),
                        "retrieval_channels": sorted(retrieval_channels.get(chunk.id, set())),
                        "doc_sha256": doc.sha256,
                        "paper_record": _is_paper_record(chunk.content),
                    },
                )
            )
        ranked_results = sorted(results, key=lambda item: item.score, reverse=True)
        # 图片去重策略
        # - figure_region / embedded_image：精准切图，同一页不同区域各自保留
        # - page / whole_figure_fallback：全页级别，同一页只保留最佳一个
        # - 查询含图片意图时，优先展示切图，过滤纯文字页截图
        ASSET_PRIORITY = {"page": 0, "whole_figure_fallback": 1, "figure": 2, "table": 3, "embedded_image": 4, "figure_region": 5}
        PRECISE_CROP_TYPES = {"figure_region", "embedded_image", "figure", "table"}
        deduped: list[SearchResult] = []
        best_per_page: dict[tuple[str, int], SearchResult] = {}
        kept_regions: set[tuple[str, int, str]] = set()
        for r in ranked_results:
            if r.kind == "image":
                asset_kind = str(r.metadata.get("asset_kind", ""))
                # 精准切图：按 (文档, 页码, 资产ID) 去重，同一页多张图各自保留
                if asset_kind in PRECISE_CROP_TYPES:
                    region_key = (r.document_id, r.page_number, r.asset_id or "")
                    if region_key not in kept_regions:
                        kept_regions.add(region_key)
                        deduped.append(r)
                else:
                    # 全页级图片：同一页只保留最佳
                    key = (r.document_id, r.page_number)
                    existing = best_per_page.get(key)
                    if existing:
                        existing_pri = ASSET_PRIORITY.get(str(existing.metadata.get("asset_kind", "")), 0)
                        current_pri = ASSET_PRIORITY.get(asset_kind, 0)
                        if current_pri > existing_pri or (current_pri == existing_pri and r.score > existing.score):
                            best_per_page[key] = r
                    else:
                        best_per_page[key] = r
            else:
                deduped.append(r)
        # 图片意图查询：如果已有足够精准切图，去掉全页级截图避免干扰
        if image_intent:
            precise_count = sum(1 for r in deduped if str(r.metadata.get("asset_kind", "")) in PRECISE_CROP_TYPES)
            if precise_count >= 2:
                best_per_page = {
                    k: v for k, v in best_per_page.items()
                    if v.metadata.get("image_class") != "SCENE_IMAGE"
                }
            deduped = deduped + sorted(best_per_page.values(), key=lambda r: r.score, reverse=True)
        else:
            deduped = deduped + sorted(best_per_page.values(), key=lambda r: r.score, reverse=True)
        ranked_results = deduped
        image_result_type_breakdown: dict[str, int] = {}
        for result in ranked_results[:top_k]:
            asset_kind = str(result.metadata.get("asset_kind") or result.kind)
            image_result_type_breakdown[asset_kind] = image_result_type_breakdown.get(asset_kind, 0) + 1
        self.last_search_diagnostics["image_result_type_breakdown"] = image_result_type_breakdown
        self.last_search_diagnostics["result_build_ms"] = round((time.perf_counter() - result_build_started) * 1000, 2)
        self.last_search_diagnostics["total_search_ms"] = round((time.perf_counter() - started) * 1000, 2)
        return ranked_results[:top_k]


def index_document_chunks(db: Session, document_id: str) -> int:
    store = VectorStore()
    chunks = db.execute(select(Chunk).where(Chunk.document_id == document_id, Chunk.approved.is_(True))).scalars().all()
    return store.index_chunks(db, chunks)
