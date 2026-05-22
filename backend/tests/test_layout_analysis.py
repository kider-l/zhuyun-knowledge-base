from pathlib import Path

from PIL import Image, ImageDraw

from app.services.layout_analysis import get_layout_analysis_service


def test_split_figure_regions_separates_two_panels(tmp_path: Path) -> None:
    image_path = tmp_path / "figure.png"
    image = Image.new("RGB", (1200, 800), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((80, 80, 520, 720), outline="black", width=6)
    draw.rectangle((680, 80, 1120, 720), outline="black", width=6)
    draw.text((180, 360), "A evacuation route", fill="black")
    draw.text((760, 360), "B evacuation route", fill="black")
    image.save(image_path)

    regions = get_layout_analysis_service().split_figure_regions(image_path)

    assert len(regions) >= 2
    assert any(region.bbox[0] < 0.2 and region.bbox[2] < 0.55 for region in regions)
    assert any(region.bbox[0] > 0.45 and region.bbox[2] > 0.8 for region in regions)
