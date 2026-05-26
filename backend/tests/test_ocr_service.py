from pathlib import Path

from app.services.ocr import OCRService


def _result(text: str, backend: str = "paddle", error: str | None = None) -> dict:
    meta = {"backend": backend, "available": bool(text), "has_line_boxes": backend == "paddle"}
    if error:
        meta["error"] = error
    return {"text": text, "lines": [], "meta": meta}


def _service(monkeypatch) -> OCRService:
    service = OCRService()
    service.backend = "paddle"
    monkeypatch.setattr(service.settings, "cloud_ocr_enabled", True)
    return service


def test_recognize_uses_high_quality_paddle_without_cloud(monkeypatch, tmp_path: Path) -> None:
    service = _service(monkeypatch)
    called = {"cloud": 0}
    paddle_text = "机房楼层高分布图 BIM 运维管理 疏散路线 东侧出口 消防通道 配电柜编号"

    monkeypatch.setattr(service, "_recognize_paddle", lambda path: _result(paddle_text, "paddle"))

    def fake_cloud(path):
        called["cloud"] += 1
        return _result("云端不应该被调用", "cloud_vl")

    monkeypatch.setattr(service, "_recognize_cloud", fake_cloud)

    result = service.recognize(tmp_path / "page.png")

    assert result["text"] == paddle_text
    assert result["meta"]["backend"] == "paddle"
    assert result["meta"]["fallback_used"] is False
    assert result["meta"]["quality_ok"] is True
    assert called["cloud"] == 0


def test_recognize_falls_back_to_cloud_for_low_quality_paddle(monkeypatch, tmp_path: Path) -> None:
    service = _service(monkeypatch)
    cloud_text = "云端识别出的机房楼层平面图 设备编号 A1 B2 疏散路线和消防出口"

    monkeypatch.setattr(service, "_recognize_paddle", lambda path: _result("aaa", "paddle"))
    monkeypatch.setattr(service, "_recognize_cloud", lambda path: _result(cloud_text, "cloud_vl"))

    result = service.recognize(tmp_path / "page.png")

    assert result["text"] == cloud_text
    assert result["meta"]["backend"] == "cloud_vl"
    assert result["meta"]["fallback_used"] is True
    assert result["meta"]["fallback_from"] == "paddle"
    assert result["meta"]["previous_quality_reason"] == "too_short"
    assert result["meta"]["attempted"] == ["paddle", "cloud_vl"]


def test_recognize_returns_explicit_failure_when_paddle_and_cloud_fail(monkeypatch, tmp_path: Path) -> None:
    service = _service(monkeypatch)

    monkeypatch.setattr(service, "_recognize_paddle", lambda path: _result("", "paddle", error="paddle missing"))
    monkeypatch.setattr(service, "_recognize_cloud", lambda path: _result("", "cloud_vl", error="cloud timeout"))

    result = service.recognize(tmp_path / "page.png")

    assert result["text"] == ""
    assert result["meta"]["backend"] == "hybrid"
    assert result["meta"]["fallback_used"] is True
    assert result["meta"]["quality_ok"] is False
    assert result["meta"]["paddle_error"] == "paddle missing"
    assert result["meta"]["cloud_error"] == "cloud timeout"
    assert result["meta"]["attempted"] == ["paddle", "cloud_vl"]


def test_recognize_respects_cloud_page_limit(monkeypatch, tmp_path: Path) -> None:
    service = _service(monkeypatch)
    called = {"cloud": 0}

    monkeypatch.setattr(service, "_recognize_paddle", lambda path: _result("", "paddle"))

    def fake_cloud(path):
        called["cloud"] += 1
        return _result("云端不应该被调用", "cloud_vl")

    monkeypatch.setattr(service, "_recognize_cloud", fake_cloud)

    result = service.recognize(tmp_path / "page.png", allow_cloud=False)

    assert result["text"] == ""
    assert result["meta"]["backend"] == "paddle"
    assert result["meta"]["cloud_skipped"] is True
    assert result["meta"]["cloud_skip_reason"] == "cloud_ocr_page_limit"
    assert result["meta"]["attempted"] == ["paddle"]
    assert called["cloud"] == 0


def test_extract_table_structure_keeps_paddle_structure_primary(monkeypatch, tmp_path: Path) -> None:
    service = _service(monkeypatch)
    called = {"ocr": 0}
    structured = {
        "text": "<table><tr><td>设备编号</td></tr></table>",
        "html": "<table><tr><td>设备编号</td></tr></table>",
        "rows": [[0, 0, 1, 1]],
        "meta": {"backend": "paddle_table", "available": True, "structured": True},
    }

    monkeypatch.setattr(service, "_extract_table_paddle", lambda path: structured)

    def fake_recognize(path, *, allow_cloud=True):
        called["ocr"] += 1
        return _result("fallback", "cloud_vl")

    monkeypatch.setattr(service, "recognize", fake_recognize)

    result = service.extract_table_structure(tmp_path / "table.png")

    assert result == structured
    assert called["ocr"] == 0
