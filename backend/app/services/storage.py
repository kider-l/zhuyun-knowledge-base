import hashlib
import re
import shutil
from pathlib import Path
from typing import BinaryIO

from app.config import get_settings


SAFE_NAME_RE = re.compile(r"[^\w\u4e00-\u9fff.()（）【】\[\]、-]+")


def sanitize_filename(filename: str) -> str:
    cleaned = SAFE_NAME_RE.sub("_", filename).strip("._")
    return cleaned or "document.pdf"


def probe_upload_stream(stream: BinaryIO) -> tuple[str, int]:
    position = stream.tell() if hasattr(stream, "tell") else None
    digest = hashlib.sha256()
    size = 0
    while True:
        chunk = stream.read(1024 * 1024)
        if not chunk:
            break
        size += len(chunk)
        digest.update(chunk)
    if position is not None and hasattr(stream, "seek"):
        stream.seek(position)
    return digest.hexdigest(), size


def probe_file(source: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with source.open("rb") as stream:
        while True:
            chunk = stream.read(1024 * 1024)
            if not chunk:
                break
            size += len(chunk)
            digest.update(chunk)
    return digest.hexdigest(), size


def document_dir(document_id: str) -> Path:
    path = get_settings().storage_dir / "documents" / document_id
    path.mkdir(parents=True, exist_ok=True)
    (path / "assets").mkdir(exist_ok=True)
    return path


def asset_dir(document_id: str) -> Path:
    path = document_dir(document_id) / "assets"
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_upload(document_id: str, filename: str, stream: BinaryIO) -> tuple[Path, str, int]:
    target = document_dir(document_id) / sanitize_filename(filename)
    digest = hashlib.sha256()
    size = 0
    with target.open("wb") as out:
        while True:
            chunk = stream.read(1024 * 1024)
            if not chunk:
                break
            size += len(chunk)
            digest.update(chunk)
            out.write(chunk)
    return target, digest.hexdigest(), size


def import_local_file(document_id: str, source: Path) -> tuple[Path, str, int]:
    target = document_dir(document_id) / sanitize_filename(source.name)
    digest = hashlib.sha256()
    size = 0
    with source.open("rb") as src, target.open("wb") as out:
        while True:
            chunk = src.read(1024 * 1024)
            if not chunk:
                break
            size += len(chunk)
            digest.update(chunk)
            out.write(chunk)
    return target, digest.hexdigest(), size


def remove_document_storage(document_id: str) -> None:
    shutil.rmtree(document_dir(document_id), ignore_errors=True)
