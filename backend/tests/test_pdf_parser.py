from pathlib import Path

import pytest


def test_pymupdf_can_create_and_read_basic_pdf(tmp_path: Path) -> None:
    fitz = pytest.importorskip("fitz")
    pdf_path = tmp_path / "sample.pdf"
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "机房楼层高分析图 BIM 运维管理")
    doc.save(pdf_path)
    doc.close()

    with fitz.open(pdf_path) as parsed:
        assert parsed.page_count == 1
        assert "BIM" in parsed[0].get_text("text")

