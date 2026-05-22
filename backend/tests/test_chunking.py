from app.services.chunking import extract_captions, make_snippet, split_text


def test_extract_captions_from_chinese_pdf_text() -> None:
    text = "1.4 建筑方案\n图4-1 机房楼层高分析图\n表 2 单体信息表"
    assert "图4-1 机房楼层高分析图" in extract_captions(text)
    assert "表 2 单体信息表" in extract_captions(text)


def test_split_text_keeps_overlap() -> None:
    text = "\n\n".join([f"段落{i} " + "机房运维资料" * 20 for i in range(8)])
    chunks = split_text(text, max_chars=180, overlap=20)
    assert len(chunks) > 1
    assert all(chunk.strip() for chunk in chunks)


def test_snippet_centers_query() -> None:
    snippet = make_snippet("前置内容" * 20 + "园区总平面图" + "后置内容" * 20, "园区总平面图", limit=80)
    assert "园区总平面图" in snippet

