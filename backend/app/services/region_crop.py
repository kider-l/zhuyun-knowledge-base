from __future__ import annotations

from pathlib import Path

from PIL import Image


def crop_normalized_bbox(
    image_path: str | Path,
    bbox: tuple[float, float, float, float],
    target_path: str | Path,
) -> tuple[int, int]:
    image = Image.open(image_path)
    width, height = image.size
    x0, y0, x1, y1 = bbox
    left = max(0, min(width - 1, int(round(x0 * width))))
    top = max(0, min(height - 1, int(round(y0 * height))))
    right = max(left + 1, min(width, int(round(x1 * width))))
    bottom = max(top + 1, min(height, int(round(y1 * height))))
    cropped = image.crop((left, top, right, bottom))
    target = Path(target_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    cropped.save(target)
    return cropped.size


def compose_bbox(
    parent_bbox: tuple[float, float, float, float],
    child_bbox: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    px0, py0, px1, py1 = parent_bbox
    cx0, cy0, cx1, cy1 = child_bbox
    pw = max(px1 - px0, 1e-6)
    ph = max(py1 - py0, 1e-6)
    return (
        px0 + pw * cx0,
        py0 + ph * cy0,
        px0 + pw * cx1,
        py0 + ph * cy1,
    )
