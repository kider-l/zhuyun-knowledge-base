from __future__ import annotations

from datetime import datetime
from pathlib import Path
import re

from sqlalchemy import delete
from sqlalchemy.orm import Session

from app.models import Asset, Chunk, Document, Job
from app.services.chunking import extract_captions, infer_title_path, normalize_text, split_text
from app.services.layout_analysis import LayoutRegion, get_layout_analysis_service
from app.services.ocr import get_ocr_service
from app.services.region_crop import compose_bbox, crop_normalized_bbox
from app.services.storage import asset_dir
from app.services.vision_summary import get_vision_summary_service


def _update_job(db: Session, job: Job | None, progress: int, message: str) -> None:
    if not job:
        return
    job.progress = progress
    job.message = message
    job.updated_at = datetime.utcnow()
    db.commit()


def _clear_previous_parse(db: Session, document: Document) -> None:
    db.execute(delete(Chunk).where(Chunk.document_id == document.id))
    db.execute(delete(Asset).where(Asset.document_id == document.id))
    db.commit()
    db.refresh(document)


def _page_keywords(text: str) -> bool:
    keywords = ["图", "图纸", "平面图", "疏散", "路线", "流程", "示意", "机房", "BIM", "表"]
    return any(word in text for word in keywords)


def _extract_keywords_for_image(caption: str, context: str, max_count: int = 8) -> list[str]:
    """Extract key terms from caption + context text for image annotation."""
    combined = f"{caption} {context}"[:1200]
    tokens = [t.strip() for t in combined.replace("\n", " ").split() if len(t.strip()) >= 2]
    seen: set[str] = set()
    result: list[str] = []
    for t in tokens:
        if t not in seen:
            seen.add(t)
            result.append(t)
    return result[:max_count]


def _table_text(rows: list[list[object]]) -> str:
    lines: list[str] = []
    for row in rows:
        cells = [str(cell).strip() for cell in row if cell is not None and str(cell).strip()]
        if cells:
            lines.append(" | ".join(cells))
    return "\n".join(lines)


def _compact_text(value: str) -> str:
    return re.sub(r"\s+", "", value).lower()


def _ocr_lines_for_chunk(chunk_text: str, ocr_lines: list[dict], limit: int = 8) -> list[dict]:
    if not ocr_lines:
        return []
    compact_chunk = _compact_text(chunk_text)
    scored: list[tuple[float, int, dict]] = []
    for index, line in enumerate(ocr_lines):
        text = str(line.get("text") or "").strip()
        compact_line = _compact_text(text)
        if not compact_line:
            continue
        score = 0.0
        if compact_line in compact_chunk:
            score += min(len(compact_line) / 10, 5.0)
        else:
            common = sum(1 for char in set(compact_line) if char in compact_chunk)
            score += common / max(len(set(compact_line)), 1)
        if score > 0.35:
            scored.append((score, index, line))
    if not scored:
        return []
    scored.sort(key=lambda item: item[0], reverse=True)
    selected = sorted(scored[:limit], key=lambda item: item[1])
    return [
        {
            "text": str(line.get("text") or ""),
            "x": float(line.get("x") or 0),
            "y": float(line.get("y") or 0),
            "width": float(line.get("width") or 0),
            "height": float(line.get("height") or 0),
        }
        for _score, _index, line in selected
    ]


def _normalized_bbox_to_page_dict(page_rect, bbox: tuple[float, float, float, float]) -> dict[str, float]:
    x0, y0, x1, y1 = bbox
    return {
        "x0": page_rect.x0 + page_rect.width * x0,
        "y0": page_rect.y0 + page_rect.height * y0,
        "x1": page_rect.x0 + page_rect.width * x1,
        "y1": page_rect.y0 + page_rect.height * y1,
    }


def _layout_region_to_absolute_bbox(page_rect, region: LayoutRegion) -> dict[str, float]:
    return _normalized_bbox_to_page_dict(page_rect, region.bbox)


def _bbox_overlap(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    inter_w = max(0.0, min(ax1, bx1) - max(ax0, bx0))
    inter_h = max(0.0, min(ay1, by1) - max(ay0, by0))
    inter = inter_w * inter_h
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0)
    area_b = max(0.0, bx1 - bx0) * max(0.0, by1 - by0)
    return inter / max(min(area_a, area_b), 1e-6)


def _dedupe_regions(regions: list[LayoutRegion]) -> list[LayoutRegion]:
    deduped: list[LayoutRegion] = []
    for region in sorted(regions, key=lambda item: item.score, reverse=True):
        if any(_bbox_overlap(region.bbox, existing.bbox) >= 0.72 for existing in deduped):
            continue
        deduped.append(region)
    return deduped


def _build_image_chunk_content(
    *,
    filename: str,
    page_number: int,
    asset_kind: str,
    caption: str = "",
    ocr_text: str = "",
    region_summary_text: str = "",
    page_context: str = "",
) -> str:
    parts = [
        f"文件：{filename}",
        f"页码：{page_number}",
        f"资源类型：{asset_kind}",
    ]
    if caption:
        parts.append(f"图题：{caption}")
    if region_summary_text:
        parts.append(f"区域描述：{region_summary_text}")
    if ocr_text:
        parts.append(f"区域文字：{ocr_text[:1800]}")
    if page_context:
        parts.append(f"页面上下文：{page_context[:1200]}")
    return normalize_text("\n".join(parts))


def _create_text_chunks(
    *,
    db: Session,
    document: Document,
    page_number: int,
    title_path: str | None,
    full_text: str,
    source: str,
    ocr_lines: list[dict],
    stats: dict,
) -> None:
    for chunk_text in split_text(full_text):
        chunk_ocr_lines = _ocr_lines_for_chunk(chunk_text, ocr_lines) if source in {"ocr", "mixed"} else []
        db.add(
            Chunk(
                document_id=document.id,
                page_number=page_number,
                kind="text",
                content=chunk_text,
                title_path=title_path,
                chunk_metadata={
                    "source": source,
                    "needs_ocr": False,
                    "ocr_lines": chunk_ocr_lines,
                },
            )
        )
        stats["text_chunks"] += 1


def _create_legacy_page_chunk(
    *,
    db: Session,
    document: Document,
    page_asset: Asset,
    page_number: int,
    title_path: str | None,
    full_text: str,
    captions: list[str],
    ocr_meta: dict,
    ocr_lines: list[dict],
    stats: dict,
) -> None:
    content_parts = [
        f"文件：{document.filename}",
        f"第 {page_number} 页页面截图",
        "关键词：图纸 图片 页面 机房 运维 平面图 流程 疏散 路线",
    ]
    if captions:
        content_parts.append("图表标题：" + "；".join(captions[:5]))
    if full_text:
        content_parts.append(full_text[:1400])
    elif ocr_meta.get("error"):
        content_parts.append(f"OCR 未完成：{ocr_meta.get('error')}")
    else:
        content_parts.append("该页为图片型或扫描型页面，建议结合页面预览查看。")
    db.add(
        Chunk(
            document_id=document.id,
            asset_id=page_asset.id,
            page_number=page_number,
            kind="image",
            content=normalize_text("\n".join(content_parts)),
            title_path=title_path,
            chunk_metadata={
                "asset_kind": "page",
                "legacy_fallback": True,
                "needs_ocr": not bool(full_text),
                "ocr": ocr_meta,
                "ocr_lines": _ocr_lines_for_chunk(full_text, ocr_lines, limit=12) if ocr_lines else [],
            },
        )
    )
    stats["image_chunks"] += 1
    stats["legacy_page_chunks"] += 1


def _create_legacy_embedded_image_chunk(
    *,
    db: Session,
    document: Document,
    page_number: int,
    title_path: str | None,
    asset: Asset,
    caption: str | None,
    full_text: str,
    stats: dict,
    image_classification: dict[str, object] | None = None,
) -> None:
    meta: dict[str, object] = {
        "asset_kind": "embedded_image",
        "legacy_fallback": True,
        "needs_ocr": not bool(full_text),
    }
    if image_classification:
        meta["image_class"] = image_classification.get("image_class", "DOC_IMAGE")
        meta["class_confidence"] = image_classification.get("class_confidence", 0.5)
        meta["core_topic"] = image_classification.get("core_topic", "")
        meta["image_keywords"] = image_classification.get("image_keywords", [])
        meta["usable_for_qa"] = image_classification.get("usable_for_qa", True)
    content = normalize_text(
        "\n".join(
            [
                f"文件：{document.filename}",
                f"第 {page_number} 页嵌入图片",
                f"图片标题：{caption or '未识别标题'}",
                full_text[:1000] or "图片上下文暂缺，可能需要 OCR。",
            ]
        )
    )
    db.add(
        Chunk(
            document_id=document.id,
            asset_id=asset.id,
            page_number=page_number,
            kind="image",
            content=content,
            title_path=title_path,
            bbox=asset.bbox,
            chunk_metadata=meta,
        )
    )
    stats["image_chunks"] += 1
    stats["legacy_embedded_chunks"] += 1


def parse_pdf(db: Session, document: Document, job: Job | None = None) -> dict:
    try:
        import fitz
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("PyMuPDF is required to parse PDF files") from exc

    document.status = "parsing"
    document.error_message = None
    document.updated_at = datetime.utcnow()
    if job:
        job.status = "running"
        job.progress = 1
        job.message = "Starting PDF parsing"
        job.updated_at = datetime.utcnow()
    db.commit()

    _clear_previous_parse(db, document)

    pdf_path = Path(document.stored_path)
    assets_path = asset_dir(document.id)
    ocr = get_ocr_service()
    layout = get_layout_analysis_service()
    vision_summary = get_vision_summary_service()
    stats = {
        "pages": 0,
        "text_pages": 0,
        "ocr_pages": 0,
        "needs_ocr_pages": 0,
        "text_chunks": 0,
        "image_chunks": 0,
        "page_assets": 0,
        "embedded_images": 0,
        "table_assets": 0,
        "table_chunks": 0,
        "table_summaries": 0,
        "structured_tables": 0,
        "legacy_page_chunks": 0,
        "legacy_embedded_chunks": 0,
        "figure_regions": 0,
        "split_regions": 0,
        "vision_summaries": 0,
        "fallback_whole_figure_count": 0,
        "figure_region_diagnostics": [],
        "text_chars": 0,
        "ocr_backend": "cloud_vl" if ocr.backend in {"", "none", "disabled"} and ocr.settings.cloud_ocr_enabled else ocr.backend,
        "warnings": [],
    }

    with fitz.open(pdf_path) as pdf:
        document.page_count = pdf.page_count
        stats["pages"] = pdf.page_count
        for page_index in range(pdf.page_count):
            page = pdf.load_page(page_index)
            page_number = page_index + 1
            page_text = normalize_text(page.get_text("text") or "")
            captions = extract_captions(page_text)
            title_path = infer_title_path(page_text)

            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
            page_asset_path = assets_path / f"page_{page_number:04d}.png"
            pix.save(str(page_asset_path))
            page_asset = Asset(
                document_id=document.id,
                page_number=page_number,
                kind="page",
                path=str(page_asset_path),
                mime_type="image/png",
                width=pix.width,
                height=pix.height,
                caption=" / ".join(captions[:3]) if captions else None,
                ocr_text=None,
            )
            db.add(page_asset)
            db.flush()
            stats["page_assets"] += 1

            ocr_text = ""
            ocr_meta: dict = {}
            ocr_lines: list[dict] = []
            needs_page_ocr = len(page_text) < 40
            if needs_page_ocr:
                ocr_result = ocr.recognize(page_asset_path)
                ocr_text = ocr_result["text"]
                ocr_meta = ocr_result["meta"]
                ocr_lines = ocr_result["lines"]
                page_asset.ocr_text = ocr_text or None
                if ocr_text:
                    stats["ocr_pages"] += 1
                else:
                    stats["needs_ocr_pages"] += 1
                    if ocr_meta.get("error"):
                        stats["warnings"].append({"page": page_number, "ocr": ocr_meta})

            full_text = normalize_text("\n\n".join(part for part in [page_text, ocr_text] if part))
            if page_text:
                stats["text_pages"] += 1
                stats["text_chars"] += len(page_text)
            elif ocr_text:
                stats["text_chars"] += len(ocr_text)

            if full_text:
                source = "mixed" if page_text and ocr_text else ("pdf_text" if page_text else "ocr")
                _create_text_chunks(
                    db=db,
                    document=document,
                    page_number=page_number,
                    title_path=title_path,
                    full_text=full_text,
                    source=source,
                    ocr_lines=ocr_lines,
                    stats=stats,
                )

            try:
                image_blocks = [block for block in page.get_text("dict").get("blocks", []) if block.get("type") == 1]
            except Exception:
                image_blocks = []

            page_should_be_image_chunk = bool(image_blocks or captions or _page_keywords(full_text) or len(full_text) < 40)
            if page_should_be_image_chunk:
                _create_legacy_page_chunk(
                    db=db,
                    document=document,
                    page_asset=page_asset,
                    page_number=page_number,
                    title_path=title_path,
                    full_text=full_text,
                    captions=captions,
                    ocr_meta=ocr_meta,
                    ocr_lines=ocr_lines,
                    stats=stats,
                )

            table_candidates: list[tuple[dict[str, float], str]] = []
            try:
                detected_tables = list(getattr(page.find_tables(), "tables", []))
            except Exception:
                detected_tables = []
            for table in detected_tables:
                bbox = getattr(table, "bbox", None)
                if not bbox:
                    continue
                rect = fitz.Rect(bbox)
                if rect.width < 40 or rect.height < 30:
                    continue
                table_candidates.append(
                    (
                        {"x0": rect.x0, "y0": rect.y0, "x1": rect.x1, "y1": rect.y1},
                        "pymupdf",
                    )
                )

            if not table_candidates:
                for table_region in layout.detect_table_regions(page_asset_path):
                    table_candidates.append((_layout_region_to_absolute_bbox(page.rect, table_region), table_region.source))

            for table_index, (bbox, source_name) in enumerate(table_candidates):
                rect = fitz.Rect(bbox["x0"], bbox["y0"], bbox["x1"], bbox["y1"])
                if rect.width < 40 or rect.height < 30:
                    continue
                table_path = assets_path / f"page_{page_number:04d}_table_{table_index:02d}.png"
                table_pix = page.get_pixmap(matrix=fitz.Matrix(2, 2), clip=rect, alpha=False)
                table_pix.save(str(table_path))
                extracted_table_text = ""
                if source_name == "pymupdf":
                    try:
                        table_rows = detected_tables[table_index].extract() if table_index < len(detected_tables) else []
                    except Exception:
                        table_rows = []
                    extracted_table_text = normalize_text(_table_text(table_rows))
                structured = ocr.extract_table_structure(table_path)
                if structured["meta"].get("structured"):
                    stats["structured_tables"] += 1
                extracted_table_text = normalize_text(
                    "\n".join(part for part in [extracted_table_text, str(structured.get("text") or "")] if part)
                )
                summary = vision_summary.summarize_table_image(table_path, extracted_table_text)
                if summary:
                    stats["table_summaries"] += 1
                asset = Asset(
                    document_id=document.id,
                    page_number=page_number,
                    kind="table",
                    path=str(table_path),
                    mime_type="image/png",
                    width=table_pix.width,
                    height=table_pix.height,
                    bbox=bbox,
                    region_type="table",
                    region_summary=summary[:1000] if summary else None,
                    caption=summary[:500] if summary else None,
                    ocr_text=extracted_table_text or None,
                )
                db.add(asset)
                db.flush()
                stats["table_assets"] += 1
                content = normalize_text(
                    "\n".join(
                        part
                        for part in [
                            f"文件：{document.filename}",
                            f"页码：{page_number}",
                            f"表格概述：{summary}",
                            f"表格结构文本：{extracted_table_text[:1800]}",
                            full_text[:700],
                        ]
                        if part
                    )
                )
                db.add(
                    Chunk(
                        document_id=document.id,
                        asset_id=asset.id,
                        page_number=page_number,
                        kind="image",
                        content=content,
                        title_path=title_path,
                        bbox=asset.bbox,
                        chunk_metadata={
                            "asset_kind": "table",
                            "image_class": "DOC_IMAGE",
                            "class_confidence": 1.0,
                            "core_topic": (summary or "表格")[:120],
                            "image_keywords": _extract_keywords_for_image(summary or "", extracted_table_text),
                            "usable_for_qa": True,
                            "table_index": table_index,
                            "summary": summary,
                            "table_structured_text": extracted_table_text[:3000],
                            "table_structure_meta": structured["meta"],
                            "has_structured_text": bool(extracted_table_text),
                        },
                    )
                )
                stats["table_chunks"] += 1
                stats["image_chunks"] += 1

            figure_regions = [region for region in layout.detect_regions(page_asset_path) if region.label == "figure"]
            image_count_on_page = 0
            for block_index, block in enumerate(image_blocks):
                bbox = block.get("bbox")
                if not bbox or len(bbox) < 4:
                    continue
                x0, y0, x1, y1 = [float(value) for value in bbox[:4]]
                normalized = (
                    x0 / max(page.rect.width, 1),
                    y0 / max(page.rect.height, 1),
                    x1 / max(page.rect.width, 1),
                    y1 / max(page.rect.height, 1),
                )
                figure_regions.append(LayoutRegion(label="figure", bbox=normalized, score=0.5, source="embedded_image"))
                width = int(block.get("width") or 0)
                height = int(block.get("height") or 0)
                if width < 80 or height < 80 or "image" not in block:
                    continue
                ext = str(block.get("ext") or "png").lower()
                if ext not in {"png", "jpg", "jpeg"}:
                    ext = "png"
                image_path = assets_path / f"page_{page_number:04d}_image_{block_index:02d}.{ext}"
                image_path.write_bytes(block["image"])
                caption = captions[min(image_count_on_page, len(captions) - 1)] if captions else None
                asset = Asset(
                    document_id=document.id,
                    page_number=page_number,
                    kind="image",
                    path=str(image_path),
                    mime_type=f"image/{'jpeg' if ext in {'jpg', 'jpeg'} else ext}",
                    width=width,
                    height=height,
                    bbox={"x0": x0, "y0": y0, "x1": x1, "y1": y1},
                    region_type="embedded_image",
                    region_index=image_count_on_page,
                    caption=caption,
                )
                db.add(asset)
                db.flush()
                stats["embedded_images"] += 1
                image_count_on_page += 1
                img_class = vision_summary.classify_image(
                    image_path,
                    asset_kind="embedded_image",
                    caption=caption or "",
                    page_text=full_text,
                    chapter_info=title_path or "",
                )
                _create_legacy_embedded_image_chunk(
                    db=db,
                    document=document,
                    page_number=page_number,
                    title_path=title_path,
                    asset=asset,
                    caption=caption,
                    full_text=full_text,
                    stats=stats,
                    image_classification=img_class,
                )
            figure_regions = _dedupe_regions(figure_regions)

            if not figure_regions and (image_blocks or captions or _page_keywords(full_text) or len(full_text) < 40):
                figure_regions = [LayoutRegion(label="figure", bbox=(0.02, 0.02, 0.98, 0.98), score=0.1, source="page_fallback")]

            page_figure_diagnostics = {
                "page_number": page_number,
                "figure_count": len(figure_regions),
                "figure_sources": [region.source for region in figure_regions],
                "split_region_counts": [],
                "whole_figure_fallbacks": 0,
            }
            for figure_index, figure_region in enumerate(figure_regions):
                figure_path = assets_path / f"page_{page_number:04d}_figure_{figure_index:02d}.png"
                crop_normalized_bbox(page_asset_path, figure_region.bbox, figure_path)
                figure_bbox = _layout_region_to_absolute_bbox(page.rect, figure_region)
                figure_asset = Asset(
                    document_id=document.id,
                    page_number=page_number,
                    kind="image",
                    path=str(figure_path),
                    mime_type="image/png",
                    bbox=figure_bbox,
                    region_type="figure",
                    region_index=figure_index,
                    caption=captions[min(figure_index, len(captions) - 1)] if captions else None,
                )
                db.add(figure_asset)
                db.flush()
                stats["figure_regions"] += 1

                split_regions = layout.split_figure_regions(figure_path)
                if len(split_regions) <= 1:
                    split_regions = [LayoutRegion(label="figure_region", bbox=(0.0, 0.0, 1.0, 1.0), score=1.0, source="whole_figure")]
                    stats["fallback_whole_figure_count"] += 1
                    page_figure_diagnostics["whole_figure_fallbacks"] += 1
                else:
                    stats["split_regions"] += len(split_regions)
                page_figure_diagnostics["split_region_counts"].append(len(split_regions))

                for region_index, split_region in enumerate(split_regions):
                    region_path = assets_path / f"page_{page_number:04d}_figure_{figure_index:02d}_region_{region_index:02d}.png"
                    crop_normalized_bbox(figure_path, split_region.bbox, region_path)
                    region_bbox_normalized = compose_bbox(figure_region.bbox, split_region.bbox)
                    region_bbox = _normalized_bbox_to_page_dict(page.rect, region_bbox_normalized)
                    region_ocr = ocr.recognize_region(region_path)
                    region_ocr_text = region_ocr["text"].strip()
                    figure_summary = vision_summary.summarize_figure_region(
                        region_path,
                        extracted_text=region_ocr_text,
                        context_text=full_text,
                    )
                    region_summary_text = vision_summary.region_summary_text(figure_summary)
                    if region_summary_text:
                        stats["vision_summaries"] += 1
                    region_asset = figure_asset
                    if split_region.source != "whole_figure" or region_index > 0:
                        region_asset = Asset(
                            document_id=document.id,
                            page_number=page_number,
                            kind="image",
                            path=str(region_path),
                            mime_type="image/png",
                            bbox=region_bbox,
                            parent_asset_id=figure_asset.id,
                            region_index=region_index,
                            region_type="figure_region",
                            region_summary=region_summary_text[:1000] if region_summary_text else None,
                            caption=figure_asset.caption,
                            ocr_text=region_ocr_text or None,
                        )
                        db.add(region_asset)
                        db.flush()
                    else:
                        region_asset.region_type = "whole_figure_fallback"
                        region_asset.region_summary = region_summary_text[:1000] if region_summary_text else None
                        region_asset.ocr_text = region_ocr_text or None

                    region_img_class = vision_summary.classify_image(
                        region_path,
                        asset_kind=region_asset.region_type or "image",
                        caption=region_asset.caption or "",
                        page_text=full_text,
                        chapter_info=title_path or "",
                    )
                    db.add(
                        Chunk(
                            document_id=document.id,
                            asset_id=region_asset.id,
                            page_number=page_number,
                            kind="image",
                            content=_build_image_chunk_content(
                                filename=document.filename,
                                page_number=page_number,
                                asset_kind=region_asset.region_type or "image",
                                caption=region_asset.caption or "",
                                ocr_text=region_ocr_text,
                                region_summary_text=region_summary_text,
                                page_context=full_text,
                            ),
                            title_path=title_path,
                            bbox=region_bbox,
                            chunk_metadata={
                                "asset_kind": region_asset.region_type or "image",
                                "image_class": region_img_class.get("image_class", "DOC_IMAGE"),
                                "class_confidence": region_img_class.get("class_confidence", 0.5),
                                "core_topic": region_img_class.get("core_topic", ""),
                                "image_keywords": region_img_class.get("image_keywords", []),
                                "usable_for_qa": region_img_class.get("usable_for_qa", True),
                                "layout_source": figure_region.source,
                                "parent_asset_id": figure_asset.id if region_asset.id != figure_asset.id else None,
                                "region_index": region_index,
                                "caption": region_asset.caption,
                                "needs_ocr": not bool(region_ocr_text),
                                "ocr": region_ocr["meta"],
                                "ocr_lines": _ocr_lines_for_chunk(region_ocr_text, region_ocr["lines"], limit=12) if region_ocr["lines"] else [],
                                "region_summary": figure_summary,
                                "region_summary_text": region_summary_text,
                                "visible_labels": figure_summary.get("visible_labels", []),
                                "key_symbols": figure_summary.get("key_symbols", []),
                                "exits_or_destinations": figure_summary.get("exits_or_destinations", []),
                            },
                        )
                    )
                    stats["image_chunks"] += 1
            if figure_regions:
                stats["figure_region_diagnostics"].append(page_figure_diagnostics)

            if page_index % 3 == 0 or page_index == pdf.page_count - 1:
                db.commit()
                progress = 5 + int((page_index + 1) / max(pdf.page_count, 1) * 85)
                _update_job(db, job, progress, f"Parsed page {page_number}/{pdf.page_count}")

    document.status = "parsed"
    document.parse_stats = stats
    document.updated_at = datetime.utcnow()
    if job:
        job.status = "succeeded"
        job.progress = 100
        job.message = "PDF parsing completed, awaiting approval"
        job.updated_at = datetime.utcnow()
    db.commit()
    return stats
