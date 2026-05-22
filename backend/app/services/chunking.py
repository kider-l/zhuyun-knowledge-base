import re


HEADING_RE = re.compile(r"^\s*((?:第[一二三四五六七八九十百千万]+[章节篇])|(?:\d+(?:\.\d+){0,4})[、.\s]+)(.{2,80})$")
CAPTION_RE = re.compile(r"(图|表)\s*\d+(?:[-—－]\d+)*\s*[^\n]{0,80}")


def normalize_text(text: str) -> str:
    text = text.replace("\u3000", " ").replace("\xa0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def split_text(text: str, max_chars: int = 900, overlap: int = 140) -> list[str]:
    text = normalize_text(text)
    if not text:
        return []
    paragraphs = [item.strip() for item in re.split(r"\n\s*\n", text) if item.strip()]
    chunks: list[str] = []
    current = ""
    for paragraph in paragraphs:
        if len(paragraph) > max_chars:
            if current:
                chunks.append(current.strip())
                current = ""
            start = 0
            while start < len(paragraph):
                end = min(start + max_chars, len(paragraph))
                chunks.append(paragraph[start:end].strip())
                if end == len(paragraph):
                    break
                start = max(0, end - overlap)
            continue
        if not current:
            current = paragraph
        elif len(current) + len(paragraph) + 2 <= max_chars:
            current = f"{current}\n\n{paragraph}"
        else:
            chunks.append(current.strip())
            tail = current[-overlap:] if overlap and len(current) > overlap else ""
            current = f"{tail}\n\n{paragraph}".strip()
    if current:
        chunks.append(current.strip())
    return [chunk for chunk in chunks if len(chunk) >= 20 or len(chunks) == 1]


def extract_captions(text: str) -> list[str]:
    normalized = normalize_text(text)
    captions: list[str] = []
    for line in normalized.splitlines():
        for match in CAPTION_RE.finditer(line):
            caption = match.group(0).strip()
            if caption and caption not in captions:
                captions.append(caption)
    return captions


def extract_title_candidates(text: str) -> list[str]:
    titles: list[str] = []
    for raw_line in normalize_text(text).splitlines():
        line = raw_line.strip()
        if not line or len(line) > 100:
            continue
        match = HEADING_RE.match(line)
        if match:
            title = line.strip()
            if title not in titles:
                titles.append(title)
    return titles


def infer_title_path(page_text: str, fallback: str | None = None) -> str | None:
    titles = extract_title_candidates(page_text)
    if not titles:
        return fallback
    return " / ".join(titles[-3:])


def make_snippet(text: str, query: str = "", limit: int = 240) -> str:
    text = normalize_text(text)
    if not text:
        return ""
    if query and query in text:
        pos = text.find(query)
        start = max(0, pos - limit // 3)
        snippet = text[start : start + limit]
    else:
        snippet = text[:limit]
    return snippet + ("..." if len(snippet) < len(text) else "")

