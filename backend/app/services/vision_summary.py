import base64
import json
import mimetypes
from pathlib import Path

import httpx

from app.config import get_settings


class VisionSummaryService:
    def __init__(self) -> None:
        self.settings = get_settings()

    @property
    def configured(self) -> bool:
        base_url = self.settings.vision_summary_api_base_url or self.settings.cloud_api_base_url
        api_key = self.settings.vision_summary_api_key or self.settings.image_embedding_api_key or self.settings.cloud_api_key
        return bool(base_url and api_key and self.settings.vision_summary_model)

    @property
    def table_summary_configured(self) -> bool:
        base_url = self.settings.vision_summary_api_base_url or self.settings.cloud_api_base_url
        api_key = self.settings.vision_summary_api_key or self.settings.image_embedding_api_key or self.settings.cloud_api_key
        return bool(self.settings.table_summary_enabled and base_url and api_key and self.settings.vision_summary_model)

    def summarize_table_image(self, image_path: str | Path, extracted_text: str = "") -> str:
        fallback = self._fallback_summary(extracted_text)
        if not self.table_summary_configured:
            return fallback
        path = Path(image_path)
        prompt = (
            "请阅读这张 PDF 表格截图，输出用于资料检索的中文概述。"
            "要求：1. 说明表格主题；2. 提取关键字段、对象、数值或约束；"
            "3. 如果图像不清晰，结合已抽取文本说明；4. 控制在 180 字以内。\n\n"
            f"已抽取文本：{extracted_text[:2000] or '无'}"
        )
        try:
            summary = self._chat_with_image(path, prompt, self.settings.vision_summary_model, temperature=0.1)
            return summary or fallback
        except Exception:
            return fallback

    def extract_page_text(self, image_path: str | Path) -> str:
        if not self.configured:
            return ""
        path = Path(image_path)
        prompt = (
            "请对这张 PDF 页面截图进行 OCR。"
            "只输出页面中可读的中文/英文正文、标题、表格文字和编号。"
            "尽量保持原有顺序；不要编造看不清的内容；如果几乎不可读，请输出空字符串。"
        )
        return self._chat_with_image(path, prompt, self.settings.cloud_ocr_model or self.settings.vision_summary_model, temperature=0)

    def summarize_figure_region(
        self,
        image_path: str | Path,
        *,
        extracted_text: str = "",
        context_text: str = "",
    ) -> dict[str, object]:
        fallback = self._fallback_region_summary(extracted_text=extracted_text, context_text=context_text)
        if not self.configured or not self.settings.figure_region_vision_enabled:
            return fallback
        path = Path(image_path)
        prompt = (
            "请阅读这张工程图纸、现场疏散图、流程图或平面示意图，并输出一个 JSON 对象。"
            "只输出 JSON，不要额外解释。"
            '字段必须包含: "scene_type", "area_name", "floor_or_building", "visible_labels", '
            '"key_symbols", "route_description", "exits_or_destinations", "legend_summary", "confidence"。'
            "其中 visible_labels、key_symbols、exits_or_destinations 必须是字符串数组，confidence 为 0 到 1 的数字。"
            "重点识别区域名、楼层/分区、箭头、出口、编号、图例、路线和可见文字。"
            f"\n\n已识别文本：{extracted_text[:1200] or '无'}"
            f"\n\n页面上下文：{context_text[:1200] or '无'}"
        )
        try:
            raw = self._chat_with_image(path, prompt, self.settings.vision_summary_model, temperature=0.1)
            parsed = self._parse_json_object(raw)
            if not parsed:
                return fallback
            return self._normalize_region_summary(parsed, fallback)
        except Exception:
            return fallback

    def region_summary_text(self, summary: dict[str, object]) -> str:
        values: list[str] = []
        for key in [
            "scene_type",
            "area_name",
            "floor_or_building",
            "route_description",
            "legend_summary",
        ]:
            value = str(summary.get(key) or "").strip()
            if value:
                values.append(value)
        for key in ["visible_labels", "key_symbols", "exits_or_destinations"]:
            raw = summary.get(key) or []
            if isinstance(raw, list):
                items = [str(item).strip() for item in raw if str(item).strip()]
                if items:
                    values.append(" ".join(items))
        return " ".join(values).strip()

    def _chat_with_image(self, path: Path, prompt: str, model: str, temperature: float) -> str:
        mime_type = mimetypes.guess_type(path.name)[0] or "image/png"
        data_uri = f"data:{mime_type};base64,{base64.b64encode(path.read_bytes()).decode('ascii')}"
        base_url = (self.settings.vision_summary_api_base_url or self.settings.cloud_api_base_url or "").rstrip("/")
        api_key = self.settings.vision_summary_api_key or self.settings.image_embedding_api_key or self.settings.cloud_api_key
        if not base_url or not api_key or not model:
            raise RuntimeError("vision model API is not configured")
        with httpx.Client(timeout=self.settings.cloud_timeout_seconds) as client:
            response = client.post(
                f"{base_url}/chat/completions",
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json={
                    "model": model,
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "image_url", "image_url": {"url": data_uri}},
                                {"type": "text", "text": prompt},
                            ],
                        }
                    ],
                    "temperature": temperature,
                },
            )
            response.raise_for_status()
            choices = response.json().get("choices") or []
            return choices[0].get("message", {}).get("content", "").strip() if choices else ""

    def _parse_json_object(self, raw: str) -> dict[str, object] | None:
        raw = raw.strip()
        if not raw:
            return None
        try:
            parsed = json.loads(raw)
            return parsed if isinstance(parsed, dict) else None
        except Exception:
            pass
        start = raw.find("{")
        end = raw.rfind("}")
        if start < 0 or end <= start:
            return None
        try:
            parsed = json.loads(raw[start : end + 1])
            return parsed if isinstance(parsed, dict) else None
        except Exception:
            return None

    def _normalize_region_summary(
        self,
        parsed: dict[str, object],
        fallback: dict[str, object],
    ) -> dict[str, object]:
        normalized: dict[str, object] = {}
        for key in [
            "scene_type",
            "area_name",
            "floor_or_building",
            "route_description",
            "legend_summary",
        ]:
            value = str(parsed.get(key) or fallback.get(key) or "").strip()
            normalized[key] = value
        for key in ["visible_labels", "key_symbols", "exits_or_destinations"]:
            raw = parsed.get(key)
            if isinstance(raw, list):
                normalized[key] = [str(item).strip() for item in raw if str(item).strip()]
            elif raw:
                normalized[key] = [str(raw).strip()]
            else:
                normalized[key] = list(fallback.get(key) or [])
        try:
            confidence = float(parsed.get("confidence", fallback.get("confidence", 0.35)) or 0.35)
        except (TypeError, ValueError):
            confidence = 0.35
        normalized["confidence"] = max(0.0, min(1.0, confidence))
        return normalized

    def _fallback_region_summary(self, *, extracted_text: str = "", context_text: str = "") -> dict[str, object]:
        joined = " ".join(part for part in [extracted_text.strip(), context_text.strip()] if part)
        labels = [item for item in joined.replace("\n", " ").split(" ") if item][:10]
        return {
            "scene_type": "figure_region",
            "area_name": labels[0] if labels else "",
            "floor_or_building": "",
            "visible_labels": labels[:8],
            "key_symbols": [],
            "route_description": joined[:240],
            "exits_or_destinations": [],
            "legend_summary": joined[:240],
            "confidence": 0.2,
        }

    def _fallback_summary(self, extracted_text: str) -> str:
        text = " ".join(extracted_text.split())
        if text:
            return f"表格内容摘要：{text[:300]}"
        return "表格截图，未能抽取到结构化文字，建议结合页面预览查看。"

    def classify_image(
        self,
        image_path: str | Path,
        *,
        asset_kind: str = "",
        caption: str = "",
        page_text: str = "",
        chapter_info: str = "",
    ) -> dict[str, object]:
        """Classify an image as DOC_IMAGE or SCENE_IMAGE.

        Returns structured annotation dict with fields:
          image_class, class_confidence, core_topic, image_keywords,
          chapter, usable_for_qa, summary.
        """
        # Tables are always DOC_IMAGE — skip VLM call
        if asset_kind == "table":
            topic = (caption or "表格").strip()
            return {
                "image_class": "DOC_IMAGE",
                "class_confidence": 1.0,
                "core_topic": topic[:120],
                "image_keywords": self._extract_keywords(topic, page_text, 5),
                "chapter": chapter_info or "",
                "usable_for_qa": True,
                "summary": f"表格：{topic[:200]}" if topic else "表格截图",
            }
        if not self.configured:
            return self._fallback_classification(caption, page_text, chapter_info)
        path = Path(image_path)
        if not path.exists() or path.stat().st_size < 256:
            return self._fallback_classification(caption, page_text, chapter_info)
        prompt = (
            "你是一个工程文档图片分析器。请判断这张图片是否属于以下类型：\n\n"
            "【SCENE_IMAGE — 仅适用于以下情况】\n"
            "- 纯装饰性元素：logo、水印、背景花纹、装饰性边框\n"
            "- 无关占位图：clip art、无关图标、广告图\n"
            "注意：照片、现场图、人物图、风景图如果出现在工程文档中，属于文档内容一部分，不是SCENE_IMAGE。\n\n"
            "除此以外，所有图片均属于【DOC_IMAGE — 文档资料图】。\n\n"
            "只输出 JSON，不要额外解释。字段：\n"
            '{"image_class": "DOC_IMAGE" 或 "SCENE_IMAGE", '
            '"class_confidence": 0-1的小数, '
            '"core_topic": "图片核心内容主题（15字内）", '
            '"keywords": ["关键词1", ..., "关键词N"],  /* 5-10个 */ '
            '"usable_for_qa": true 或 false, '
            '"summary": "一句话概括图片内容（30字内）"}'
        )
        context_parts = []
        if caption:
            context_parts.append(f"图题/描述：{caption}")
        if chapter_info:
            context_parts.append(f"所属章节：{chapter_info}")
        if page_text:
            context_parts.append(f"页面上下文：{page_text[:500]}")
        if context_parts:
            prompt += "\n\n附加上下文：\n" + "\n".join(context_parts)
        try:
            raw = self._chat_with_image(path, prompt, self.settings.vision_summary_model, temperature=0.1)
            parsed = self._parse_json_object(raw)
            if not parsed or not parsed.get("image_class") in {"DOC_IMAGE", "SCENE_IMAGE"}:
                return self._fallback_classification(caption, page_text, chapter_info)
            return {
                "image_class": str(parsed["image_class"]),
                "class_confidence": max(0.0, min(1.0, float(parsed.get("class_confidence", 0.6)))),
                "core_topic": str(parsed.get("core_topic") or caption or "")[:120],
                "image_keywords": self._normalize_keywords(parsed.get("keywords")),
                "chapter": chapter_info or "",
                "usable_for_qa": bool(parsed.get("usable_for_qa", True)),
                "summary": str(parsed.get("summary") or "")[:200],
            }
        except Exception:
            return self._fallback_classification(caption, page_text, chapter_info)

    def _fallback_classification(self, caption: str = "", page_text: str = "", chapter_info: str = "") -> dict[str, object]:
        """Fallback when VLM is unavailable."""
        combined = " ".join(part for part in [caption, page_text] if part)
        # 仅检测纯装饰性元素
        decorative_keywords = ["logo", "水印", "装饰", "背景图", "图标"]
        is_decorative = any(kw in combined.lower() for kw in decorative_keywords)
        if is_decorative:
            return {
                "image_class": "SCENE_IMAGE",
                "class_confidence": 0.6,
                "core_topic": (caption or "装饰元素")[:120],
                "image_keywords": [],
                "chapter": chapter_info or "",
                "usable_for_qa": False,
                "summary": "装饰性元素",
            }
        # 默认所有插入图片为DOC_IMAGE（工程文档中的照片、图纸等均有业务价值）
        return {
            "image_class": "DOC_IMAGE",
            "class_confidence": 0.7,
            "core_topic": (caption or "文档内图片")[:120],
            "image_keywords": self._extract_keywords(caption, page_text, 5),
            "chapter": chapter_info or "",
            "usable_for_qa": True,
            "summary": caption or "",
        }

    @staticmethod
    def _extract_keywords(caption: str, page_text: str, count: int = 5) -> list[str]:
        combined = f"{caption} {page_text}"[:800]
        tokens = [t.strip() for t in combined.replace("\n", " ").split() if len(t.strip()) >= 2]
        seen: set[str] = set()
        result: list[str] = []
        for t in tokens:
            if t not in seen:
                seen.add(t)
                result.append(t)
        return result[:count]

    @staticmethod
    def _normalize_keywords(raw: object) -> list[str]:
        if isinstance(raw, list):
            return [str(item).strip() for item in raw if str(item).strip()][:10]
        if isinstance(raw, str):
            return [raw.strip()] if raw.strip() else []
        return []


def get_vision_summary_service() -> VisionSummaryService:
    return VisionSummaryService()
