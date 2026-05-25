from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from PIL import Image, ImageChops, ImageOps

from app.config import get_settings


@dataclass
class LayoutRegion:
    label: str
    bbox: tuple[float, float, float, float]
    score: float = 0.0
    source: str = "heuristic"


def _clamp_bbox(bbox: tuple[float, float, float, float]) -> tuple[float, float, float, float] | None:
    x0, y0, x1, y1 = bbox
    x0 = max(0.0, min(1.0, x0))
    y0 = max(0.0, min(1.0, y0))
    x1 = max(0.0, min(1.0, x1))
    y1 = max(0.0, min(1.0, y1))
    if x1 - x0 < 0.03 or y1 - y0 < 0.03:
        return None
    return (x0, y0, x1, y1)


def _expand_bbox(
    bbox: tuple[float, float, float, float],
    *,
    x_margin: float = 0.01,
    y_margin: float = 0.01,
) -> tuple[float, float, float, float] | None:
    x0, y0, x1, y1 = bbox
    return _clamp_bbox((x0 - x_margin, y0 - y_margin, x1 + x_margin, y1 + y_margin))


def _overlap_ratio(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    inter_w = max(0.0, min(ax1, bx1) - max(ax0, bx0))
    inter_h = max(0.0, min(ay1, by1) - max(ay0, by0))
    inter = inter_w * inter_h
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0)
    area_b = max(0.0, bx1 - bx0) * max(0.0, by1 - by0)
    denom = min(area_a, area_b) or 1.0
    return inter / denom


def _merge_regions(regions: Iterable[LayoutRegion], overlap_threshold: float = 0.75) -> list[LayoutRegion]:
    merged: list[LayoutRegion] = []
    for region in regions:
        bbox = _clamp_bbox(region.bbox)
        if not bbox:
            continue
        region = LayoutRegion(label=region.label, bbox=bbox, score=region.score, source=region.source)
        replaced = False
        for index, current in enumerate(merged):
            if current.label != region.label:
                continue
            if _overlap_ratio(current.bbox, region.bbox) >= overlap_threshold:
                if region.score >= current.score:
                    merged[index] = region
                replaced = True
                break
        if not replaced:
            merged.append(region)
    return merged


def _mask_from_image(image: Image.Image) -> tuple[list[list[int]], int, int]:
    grayscale = ImageOps.grayscale(image)
    bounded = ImageChops.invert(ImageOps.autocontrast(grayscale))
    width, height = bounded.size
    scale = max(1, int(max(width, height) / 1200))
    if scale > 1:
        bounded = bounded.resize((max(1, width // scale), max(1, height // scale)))
    width, height = bounded.size
    pixels = bounded.load()
    mask = [[1 if pixels[x, y] > 28 else 0 for x in range(width)] for y in range(height)]
    return mask, width, height


def _connected_components(mask: list[list[int]], width: int, height: int) -> list[tuple[int, int, int, int, int]]:
    visited = [[False for _ in range(width)] for _ in range(height)]
    components: list[tuple[int, int, int, int, int]] = []
    for y in range(height):
        for x in range(width):
            if not mask[y][x] or visited[y][x]:
                continue
            stack = [(x, y)]
            visited[y][x] = True
            min_x = max_x = x
            min_y = max_y = y
            area = 0
            while stack:
                cx, cy = stack.pop()
                area += 1
                min_x = min(min_x, cx)
                min_y = min(min_y, cy)
                max_x = max(max_x, cx)
                max_y = max(max_y, cy)
                for nx, ny in ((cx - 1, cy), (cx + 1, cy), (cx, cy - 1), (cx, cy + 1)):
                    if 0 <= nx < width and 0 <= ny < height and mask[ny][nx] and not visited[ny][nx]:
                        visited[ny][nx] = True
                        stack.append((nx, ny))
            components.append((min_x, min_y, max_x + 1, max_y + 1, area))
    return components


def _component_regions(
    mask: list[list[int]],
    width: int,
    height: int,
    *,
    min_area_ratio: float = 0.008,
    min_width_ratio: float = 0.12,
    min_height_ratio: float = 0.06,
    min_density: float = 0.04,
) -> list[LayoutRegion]:
    image_area = max(width * height, 1)
    regions: list[LayoutRegion] = []
    for x0, y0, x1, y1, area in _connected_components(mask, width, height):
        box_w = x1 - x0
        box_h = y1 - y0
        area_ratio = area / image_area
        width_ratio = box_w / max(width, 1)
        height_ratio = box_h / max(height, 1)
        density = area / max(box_w * box_h, 1)
        if area_ratio < min_area_ratio:
            continue
        if width_ratio < min_width_ratio or height_ratio < min_height_ratio:
            continue
        if density < min_density:
            continue
        bbox = _expand_bbox((x0 / width, y0 / height, x1 / width, y1 / height), x_margin=0.012, y_margin=0.015)
        if bbox:
            regions.append(LayoutRegion(label="figure", bbox=bbox, score=area_ratio + density * 0.1))
    return _merge_regions(regions, overlap_threshold=0.68)


def _projection_segments(values: list[float], min_gap: int) -> list[tuple[int, int]]:
    segments: list[tuple[int, int]] = []
    start: int | None = None
    for index, value in enumerate(values):
        if value <= 0.05:
            if start is None:
                start = index
        elif start is not None:
            if index - start >= min_gap:
                segments.append((start, index))
            start = None
    if start is not None and len(values) - start >= min_gap:
        segments.append((start, len(values)))
    return segments


def _split_by_projection(mask: list[list[int]], width: int, height: int) -> list[tuple[float, float, float, float]]:
    row_density = [sum(row) / max(width, 1) for row in mask]
    col_density = [sum(mask[row][col] for row in range(height)) / max(height, 1) for col in range(width)]

    gaps: list[tuple[str, int, int, int]] = []
    for start, end in _projection_segments(row_density, max(10, height // 40)):
        gaps.append(("y", start, end, end - start))
    for start, end in _projection_segments(col_density, max(10, width // 40)):
        gaps.append(("x", start, end, end - start))
    if not gaps:
        return [(0.0, 0.0, 1.0, 1.0)]
    gaps.sort(key=lambda item: item[3], reverse=True)

    # 递归切割：每次选最大间隙，在子区域上继续切割
    regions = [(0.0, 0.0, 1.0, 1.0)]
    for axis, start, end, _span in gaps:
        next_regions: list[tuple[float, float, float, float]] = []
        for rx0, ry0, rx1, ry1 in regions:
            rw = max(rx1 - rx0, 1e-6)
            rh = max(ry1 - ry0, 1e-6)
            if axis == "y":
                top_rel = start / max(height, 1)
                bottom_rel = end / max(height, 1)
                top = (top_rel - ry0) / rh
                bottom = (bottom_rel - ry0) / rh
                if top > 0.12 and bottom < 0.88:
                    next_regions.append((rx0, ry0, rx1, ry0 + rh * top))
                    next_regions.append((rx0, ry0 + rh * bottom, rx1, ry1))
                else:
                    next_regions.append((rx0, ry0, rx1, ry1))
            else:
                left_rel = start / max(width, 1)
                right_rel = end / max(width, 1)
                left = (left_rel - rx0) / rw
                right = (right_rel - rx0) / rw
                if left > 0.12 and right < 0.88:
                    next_regions.append((rx0, ry0, rx0 + rw * left, ry1))
                    next_regions.append((rx0 + rw * right, ry0, rx1, ry1))
                else:
                    next_regions.append((rx0, ry0, rx1, ry1))
        regions = next_regions
        if len(regions) >= 8:
            break
    return regions


class LayoutAnalysisService:
    def __init__(self) -> None:
        self.settings = get_settings()

    def detect_regions(self, image_path: str | Path) -> list[LayoutRegion]:
        backend = self.settings.layout_detection_backend
        path = Path(image_path)
        if backend in {"doclayout_yolo", "auto"}:
            regions = self._detect_doclayout(path)
            if regions:
                return regions
        if backend in {"paddle", "auto"}:
            regions = self._detect_paddle(path)
            if regions:
                return regions
        if backend == "none":
            return []
        return self._detect_heuristic(path)

    def split_figure_regions(self, image_path: str | Path) -> list[LayoutRegion]:
        path = Path(image_path)
        image = Image.open(path)
        mask, width, height = _mask_from_image(image)
        segments = _split_by_projection(mask, width, height)
        regions: list[LayoutRegion] = []
        if len(segments) > 1:
            for index, segment in enumerate(segments):
                bbox = _clamp_bbox(segment)
                if bbox:
                    regions.append(LayoutRegion(label="figure_region", bbox=bbox, score=1.0 - index * 0.01))
        if len(regions) <= 1:
            component_regions = _component_regions(
                mask,
                width,
                height,
                min_area_ratio=0.006,
                min_width_ratio=0.08,
                min_height_ratio=0.05,
                min_density=0.006,
            )
            regions = [
                LayoutRegion(label="figure_region", bbox=item.bbox, score=item.score, source=item.source)
                for item in component_regions
            ]
        filtered: list[LayoutRegion] = []
        for region in _merge_regions(regions, overlap_threshold=0.6):
            x0, y0, x1, y1 = region.bbox
            if (x1 - x0) * (y1 - y0) < 0.05:
                continue
            filtered.append(region)
        return filtered[:8]

    def detect_table_regions(self, image_path: str | Path) -> list[LayoutRegion]:
        return [region for region in self.detect_regions(image_path) if region.label == "table"]

    def _detect_doclayout(self, image_path: Path) -> list[LayoutRegion]:
        try:
            from doclayout_yolo import YOLOv10  # type: ignore
        except Exception:
            return []
        try:
            model = YOLOv10("doclayout_yolo_docstructbench_imgsz1024.pt")
            result = model.predict(str(image_path), imgsz=1024, conf=0.2, verbose=False)[0]
        except Exception:
            return []
        regions: list[LayoutRegion] = []
        width = max(float(getattr(result, "orig_shape", [1, 1])[1]), 1.0)
        height = max(float(getattr(result, "orig_shape", [1, 1])[0]), 1.0)
        names = getattr(result, "names", {})
        for box in getattr(result, "boxes", []) or []:
            cls_id = int(getattr(box, "cls", [0])[0])
            label = str(names.get(cls_id, "figure")).lower()
            xyxy = getattr(box, "xyxy", None)
            if xyxy is None:
                continue
            x0, y0, x1, y1 = [float(value) for value in xyxy[0].tolist()]
            bbox = _clamp_bbox((x0 / width, y0 / height, x1 / width, y1 / height))
            if not bbox:
                continue
            if "table" in label:
                normalized_label = "table"
            elif any(word in label for word in ("figure", "image", "chart")):
                normalized_label = "figure"
            else:
                continue
            regions.append(LayoutRegion(label=normalized_label, bbox=bbox, score=float(getattr(box, "conf", [0])[0]), source="doclayout_yolo"))
        return _merge_regions(regions)

    def _detect_paddle(self, image_path: Path) -> list[LayoutRegion]:
        try:
            from paddleocr import PPStructure  # type: ignore
        except Exception:
            return []
        try:
            engine = PPStructure(show_log=False, layout=True, ocr=False, table=False)
            result = engine(str(image_path))
        except Exception:
            return []
        image = Image.open(image_path)
        width, height = image.size
        regions: list[LayoutRegion] = []
        for item in result or []:
            item_type = str(item.get("type") or "").lower()
            bbox_values = item.get("bbox") or item.get("box")
            if not bbox_values or len(bbox_values) < 4:
                continue
            x0, y0, x1, y1 = [float(value) for value in bbox_values[:4]]
            bbox = _clamp_bbox((x0 / max(width, 1), y0 / max(height, 1), x1 / max(width, 1), y1 / max(height, 1)))
            if not bbox:
                continue
            if item_type in {"table"}:
                label = "table"
            elif item_type in {"figure", "image", "chart"}:
                label = "figure"
            else:
                continue
            regions.append(LayoutRegion(label=label, bbox=bbox, score=0.5, source="paddle"))
        return _merge_regions(regions)

    def _detect_heuristic(self, image_path: Path) -> list[LayoutRegion]:
        image = Image.open(image_path)
        mask, width, height = _mask_from_image(image)
        return _component_regions(mask, width, height)


def get_layout_analysis_service() -> LayoutAnalysisService:
    return LayoutAnalysisService()
