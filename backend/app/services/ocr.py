from functools import lru_cache
from pathlib import Path
from typing import TypedDict

from app.config import get_settings
from app.services.vision_summary import get_vision_summary_service


class OCRLine(TypedDict):
    text: str
    x: float
    y: float
    width: float
    height: float


class OCRResult(TypedDict):
    text: str
    meta: dict
    lines: list[OCRLine]


def _empty_result(meta: dict) -> OCRResult:
    return {"text": "", "meta": meta, "lines": []}


def _line(text: str, x: float, y: float, width: float, height: float) -> OCRLine:
    return {
        "text": text.strip(),
        "x": max(0.0, min(1.0, x)),
        "y": max(0.0, min(1.0, y)),
        "width": max(0.0, min(1.0, width)),
        "height": max(0.0, min(1.0, height)),
    }


class OCRService:
    def __init__(self) -> None:
        self.settings = get_settings()
        self.backend = self.settings.ocr_backend.lower().strip()
        self._paddle = None

    @property
    def enabled(self) -> bool:
        return self.backend not in {"", "none", "disabled"} or self.settings.cloud_ocr_enabled

    def recognize(self, image_path: Path) -> OCRResult:
        if self.backend in {"", "none", "disabled"} and self.settings.cloud_ocr_enabled:
            return self._recognize_cloud(image_path)
        if not self.enabled:
            return _empty_result({"backend": "none", "available": False})
        if self.backend == "tesseract":
            return self._recognize_tesseract(image_path)
        if self.backend == "paddle":
            return self._recognize_paddle(image_path)
        return _empty_result({"backend": self.backend, "available": False, "error": "unsupported OCR_BACKEND"})

    def recognize_region(self, image_path: Path) -> OCRResult:
        if self.settings.figure_region_ocr_enabled and self.backend == "paddle":
            result = self._recognize_paddle(image_path)
            if result["text"].strip():
                return result
        return self.recognize(image_path)

    def extract_table_structure(self, image_path: Path) -> dict:
        if self.settings.table_structure_backend in {"auto", "paddle"}:
            structured = self._extract_table_paddle(image_path)
            if structured["text"].strip():
                return structured
        ocr_result = self.recognize(image_path)
        return {
            "text": ocr_result["text"].strip(),
            "html": None,
            "rows": [],
            "meta": {
                **ocr_result["meta"],
                "structured": False,
            },
        }

    def _recognize_cloud(self, image_path: Path) -> OCRResult:
        try:
            text = get_vision_summary_service().extract_page_text(image_path)
            return {
                "text": text.strip(),
                "lines": [],
                "meta": {
                "backend": "cloud_vl",
                "model": self.settings.cloud_ocr_model,
                "available": bool(text.strip()),
                    "has_line_boxes": False,
                },
            }
        except Exception as exc:
            return _empty_result({"backend": "cloud_vl", "available": False, "error": str(exc), "has_line_boxes": False})

    def _recognize_tesseract(self, image_path: Path) -> OCRResult:
        try:
            import pytesseract
            from PIL import Image

            image = Image.open(image_path)
            data = pytesseract.image_to_data(image, lang=self.settings.ocr_lang, output_type=pytesseract.Output.DICT)
            grouped: dict[tuple[int, int, int], list[int]] = {}
            for index, raw_text in enumerate(data.get("text", [])):
                text = str(raw_text).strip()
                if not text:
                    continue
                key = (
                    int(data["block_num"][index]),
                    int(data["par_num"][index]),
                    int(data["line_num"][index]),
                )
                grouped.setdefault(key, []).append(index)

            lines: list[OCRLine] = []
            width, height = image.size
            for indexes in grouped.values():
                text = "".join(str(data["text"][idx]).strip() for idx in indexes if str(data["text"][idx]).strip())
                if not text:
                    continue
                left = min(float(data["left"][idx]) for idx in indexes)
                top = min(float(data["top"][idx]) for idx in indexes)
                right = max(float(data["left"][idx]) + float(data["width"][idx]) for idx in indexes)
                bottom = max(float(data["top"][idx]) + float(data["height"][idx]) for idx in indexes)
                lines.append(_line(text, left / width, top / height, (right - left) / width, (bottom - top) / height))

            plain_text = "\n".join(item["text"] for item in lines)
            return {"text": plain_text.strip(), "lines": lines, "meta": {"backend": "tesseract", "available": True, "has_line_boxes": bool(lines)}}
        except Exception as exc:
            return _empty_result({"backend": "tesseract", "available": False, "error": str(exc), "has_line_boxes": False})

    def _recognize_paddle(self, image_path: Path) -> OCRResult:
        try:
            if self._paddle is None:
                from paddleocr import PaddleOCR
                from PIL import Image

                self._paddle = PaddleOCR(use_angle_cls=True, lang="ch", show_log=False)
            else:
                from PIL import Image

            image_width, image_height = Image.open(image_path).size
            result = self._paddle.ocr(str(image_path), cls=True)
            lines: list[OCRLine] = []
            for page in result or []:
                for item in page or []:
                    if len(item) < 2 or not item[1]:
                        continue
                    text = str(item[1][0]).strip()
                    points = item[0] or []
                    if not text or len(points) < 4:
                        continue
                    xs = [float(point[0]) for point in points]
                    ys = [float(point[1]) for point in points]
                    x0, y0, x1, y1 = min(xs), min(ys), max(xs), max(ys)
                    lines.append(_line(text, x0 / image_width, y0 / image_height, (x1 - x0) / image_width, (y1 - y0) / image_height))
            plain_text = "\n".join(item["text"] for item in lines)
            return {"text": plain_text.strip(), "lines": lines, "meta": {"backend": "paddle", "available": True, "has_line_boxes": bool(lines)}}
        except Exception as exc:
            return _empty_result({"backend": "paddle", "available": False, "error": str(exc), "has_line_boxes": False})

    def _extract_table_paddle(self, image_path: Path) -> dict:
        try:
            from paddleocr import PPStructure  # type: ignore
        except Exception as exc:
            return {
                "text": "",
                "html": None,
                "rows": [],
                "meta": {"backend": "paddle_table", "available": False, "error": str(exc), "structured": False},
            }
        try:
            engine = PPStructure(show_log=False, layout=False, table=True, ocr=True)
            result = engine(str(image_path))
        except Exception as exc:
            return {
                "text": "",
                "html": None,
                "rows": [],
                "meta": {"backend": "paddle_table", "available": False, "error": str(exc), "structured": False},
            }
        texts: list[str] = []
        html: str | None = None
        rows: list = []
        for item in result or []:
            res = item.get("res") if isinstance(item, dict) else None
            if isinstance(res, dict):
                html = html or res.get("html")
                if res.get("html"):
                    texts.append(str(res.get("html")))
                table_cells = res.get("cell_bbox")
                if table_cells:
                    rows.append(table_cells)
            elif isinstance(res, list):
                rows.extend(res)
        if html:
            texts.append(html)
        return {
            "text": "\n".join(texts).strip(),
            "html": html,
            "rows": rows,
            "meta": {
                "backend": "paddle_table",
                "available": bool(texts or html or rows),
                "structured": bool(html or rows),
            },
        }


@lru_cache
def get_ocr_service() -> OCRService:
    return OCRService()
