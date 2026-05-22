from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, Depends, File, Form, HTTPException, UploadFile, status
from fastapi.responses import FileResponse
from sqlalchemy import desc, select
from sqlalchemy.orm import Session

from app.auth import AdminUser
from app.config import get_settings
from app.db import SessionLocal, get_session, sqlite_write_lock
from app.models import Asset, Chunk, Document, Job, UploadLog
from app.schemas import (
    AssetOut,
    ChunkOut,
    ChunkUpdateRequest,
    DocumentOut,
    DocumentReview,
    DuplicateDocumentInfo,
    JobOut,
    LocalImportRequest,
    UploadBatchItem,
    UploadBatchResponse,
    UploadLogOut,
)
from app.services.scheduler import SchedulerEnqueueError, enqueue_index, enqueue_parse
from app.services.storage import import_local_file, probe_file, probe_upload_stream, remove_document_storage, save_upload
from app.services.vector_store import VectorStore

router = APIRouter(prefix="/api", tags=["documents"])


def _asset_out(asset: Asset) -> AssetOut:
    item = AssetOut.model_validate(asset)
    item.url = f"/api/assets/{asset.id}"
    return item


def _create_parse_job(db: Session, document: Document) -> Job:
    job = Job(document_id=document.id, job_type="parse", status="queued", progress=0, message="等待解析")
    db.add(job)
    db.commit()
    db.refresh(job)
    return job


def _create_upload_log(
    db: Session,
    *,
    filename: str,
    sha256: str,
    source: str,
    uploaded_by: str,
    status_value: str,
    document_id: str | None = None,
    duplicate_of_document_id: str | None = None,
    message: str | None = None,
) -> UploadLog:
    item = UploadLog(
        filename=filename,
        sha256=sha256,
        source=source,
        uploaded_by=uploaded_by,
        status=status_value,
        message=message,
        document_id=document_id,
        duplicate_of_document_id=duplicate_of_document_id,
        updated_at=datetime.utcnow(),
    )
    db.add(item)
    db.flush()
    return item


def _duplicate_info(document: Document) -> DuplicateDocumentInfo:
    return DuplicateDocumentInfo(
        document_id=document.id,
        filename=document.filename,
        sha256=document.sha256,
        status=document.status,
        uploaded_at=document.created_at,
    )


def _duplicate_response(duplicates: list[dict]) -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_409_CONFLICT,
        detail={
            "code": "duplicate_document",
            "message": "检测到重复文档，请确认是否继续上传。",
            "duplicates": duplicates,
        },
    )


def _active_job(db: Session, *, document_id: str, job_type: str) -> Job | None:
    return (
        db.execute(
            select(Job)
            .where(Job.document_id == document_id, Job.job_type == job_type, Job.status.in_(["queued", "running"]))
            .order_by(desc(Job.created_at))
            .limit(1)
        )
        .scalar_one_or_none()
    )


def _active_jobs(db: Session, *, document_id: str, job_types: tuple[str, ...]) -> list[Job]:
    return (
        db.execute(
            select(Job)
            .where(Job.document_id == document_id, Job.job_type.in_(job_types), Job.status.in_(["queued", "running"]))
            .order_by(desc(Job.created_at))
        )
        .scalars()
        .all()
    )


def _mark_job_failed(job: Job, message: str) -> None:
    job.status = "failed"
    job.error_message = message
    job.updated_at = datetime.utcnow()


def _enqueue_or_fail(
    *,
    action: str,
    db: Session,
    document: Document,
    job: Job,
    enqueue_call,
) -> None:
    try:
        enqueue_call()
    except SchedulerEnqueueError as exc:
        with sqlite_write_lock:
            if action == "parse" and document.status == "uploaded":
                document.status = "failed"
            elif action == "index" and document.status == "approved":
                document.status = "parsed"
            document.error_message = str(exc)
            document.updated_at = datetime.utcnow()
            _mark_job_failed(job, str(exc))
            db.commit()
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)) from exc


@router.post("/documents/upload", response_model=dict)
def upload_document(
    background_tasks: BackgroundTasks,
    username: AdminUser,
    file: UploadFile = File(...),
    confirm_duplicates: bool = Form(False),
    db: Session = Depends(get_session),
) -> dict:
    result = upload_documents(
        background_tasks=background_tasks,
        username=username,
        files=[file],
        confirm_duplicates=confirm_duplicates,
        db=db,
    )
    if not result.items:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="未收到上传文件")
    item = result.items[0]
    if item.document is None or item.job is None:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=item.message or "上传失败")
    return {"document": item.document, "job": item.job}


@router.post("/documents/upload-batch", response_model=UploadBatchResponse)
def upload_documents(
    background_tasks: BackgroundTasks,
    username: AdminUser,
    files: list[UploadFile] = File(...),
    confirm_duplicates: bool = Form(False),
    db: Session = Depends(get_session),
) -> UploadBatchResponse:
    if not files:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="未收到上传文件")

    max_bytes = get_settings().max_upload_mb * 1024 * 1024
    specs: list[dict] = []
    duplicates: list[dict] = []
    seen_hashes: dict[str, str] = {}
    for file in files:
        if not file.filename or not file.filename.lower().endswith(".pdf"):
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"{file.filename or '未命名文件'} 不是 PDF")
        sha256, size = probe_upload_stream(file.file)
        if size > max_bytes:
            raise HTTPException(status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, detail=f"{file.filename} 超过上传大小限制")
        duplicate_doc = db.execute(select(Document).where(Document.sha256 == sha256).limit(1)).scalar_one_or_none()
        duplicate_batch_filename = seen_hashes.get(sha256)
        if duplicate_doc is not None:
            duplicates.append({"incoming_filename": file.filename, "existing": _duplicate_info(duplicate_doc).model_dump(mode="json")})
        elif duplicate_batch_filename:
            duplicates.append(
                {
                    "incoming_filename": file.filename,
                    "existing": {
                        "document_id": "",
                        "filename": duplicate_batch_filename,
                        "sha256": sha256,
                        "status": "batch_duplicate",
                        "uploaded_at": datetime.utcnow().isoformat(),
                    },
                }
            )
        else:
            seen_hashes[sha256] = file.filename
        specs.append({"file": file, "sha256": sha256, "size": size})

    if duplicates and not confirm_duplicates:
        with sqlite_write_lock:
            for duplicate in duplicates:
                existing = duplicate["existing"]
                _create_upload_log(
                    db,
                    filename=duplicate["incoming_filename"],
                    sha256=existing["sha256"],
                    source="web",
                    uploaded_by=username,
                    status_value="duplicate_blocked",
                    duplicate_of_document_id=existing.get("document_id") or None,
                    message="检测到重复文档，等待管理员确认。",
                )
            db.commit()
        raise _duplicate_response(duplicates)

    items: list[UploadBatchItem] = []
    enqueued: list[tuple[Document, Job]] = []
    with sqlite_write_lock:
        concurrent_duplicates: list[dict] = []
        if not confirm_duplicates:
            for spec in specs:
                duplicate_doc = db.execute(select(Document).where(Document.sha256 == spec["sha256"]).limit(1)).scalar_one_or_none()
                if duplicate_doc is not None:
                    concurrent_duplicates.append(
                        {
                            "incoming_filename": spec["file"].filename,
                            "existing": _duplicate_info(duplicate_doc).model_dump(mode="json"),
                        }
                    )
            if concurrent_duplicates:
                for duplicate in concurrent_duplicates:
                    existing = duplicate["existing"]
                    _create_upload_log(
                        db,
                        filename=duplicate["incoming_filename"],
                        sha256=existing["sha256"],
                        source="web",
                        uploaded_by=username,
                        status_value="duplicate_blocked",
                        duplicate_of_document_id=existing.get("document_id") or None,
                        message="检测到重复文档，等待管理员确认。",
                    )
                db.commit()
                raise _duplicate_response(concurrent_duplicates)

        for spec in specs:
            file = spec["file"]
            document = Document(filename=file.filename, stored_path="", original_path=None, status="uploaded")
            db.add(document)
            db.flush()
            stored_path, sha256, size = save_upload(document.id, file.filename, file.file)
            document.stored_path = str(stored_path)
            document.sha256 = sha256
            document.size_bytes = size
            document.updated_at = datetime.utcnow()
            db.commit()
            job = _create_parse_job(db, document)
            _create_upload_log(
                db,
                filename=file.filename,
                sha256=sha256,
                source="web",
                uploaded_by=username,
                status_value="queued",
                document_id=document.id,
                message="文档已上传，等待解析。",
            )
            db.commit()
            items.append(
                UploadBatchItem(
                    filename=file.filename,
                    status="queued",
                    message="已上传，等待解析",
                    document=DocumentOut.model_validate(document),
                    job=JobOut.model_validate(job),
                )
            )
            enqueued.append((document, job))

    for document, job in enqueued:
        _enqueue_or_fail(
            action="parse",
            db=db,
            document=document,
            job=job,
            enqueue_call=lambda document_id=document.id, job_id=job.id: enqueue_parse(background_tasks, document_id, job_id),
        )

    return UploadBatchResponse(items=items)


@router.post("/import/local", response_model=UploadBatchResponse)
def import_local(
    payload: LocalImportRequest,
    background_tasks: BackgroundTasks,
    username: AdminUser,
    db: Session = Depends(get_session),
) -> UploadBatchResponse:
    source = Path(payload.path).expanduser()
    if not source.exists():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="路径不存在")
    files = [source] if source.is_file() else sorted(source.rglob("*.pdf"))
    if not files:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="没有找到 PDF 文件")

    duplicates: list[dict] = []
    specs: list[dict] = []
    seen_hashes: dict[str, str] = {}
    for pdf_path in files:
        if pdf_path.suffix.lower() != ".pdf":
            continue
        sha256, size = probe_file(pdf_path)
        duplicate_doc = db.execute(select(Document).where(Document.sha256 == sha256).limit(1)).scalar_one_or_none()
        duplicate_batch_filename = seen_hashes.get(sha256)
        if duplicate_doc is not None:
            duplicates.append({"incoming_filename": pdf_path.name, "existing": _duplicate_info(duplicate_doc).model_dump(mode="json")})
        elif duplicate_batch_filename:
            duplicates.append(
                {
                    "incoming_filename": pdf_path.name,
                    "existing": {
                        "document_id": "",
                        "filename": duplicate_batch_filename,
                        "sha256": sha256,
                        "status": "batch_duplicate",
                        "uploaded_at": datetime.utcnow().isoformat(),
                    },
                }
            )
        else:
            seen_hashes[sha256] = pdf_path.name
        specs.append({"path": pdf_path, "sha256": sha256, "size": size})

    if duplicates and not payload.confirm_duplicates:
        with sqlite_write_lock:
            for duplicate in duplicates:
                existing = duplicate["existing"]
                _create_upload_log(
                    db,
                    filename=duplicate["incoming_filename"],
                    sha256=existing["sha256"],
                    source="local",
                    uploaded_by=username,
                    status_value="duplicate_blocked",
                    duplicate_of_document_id=existing.get("document_id") or None,
                    message="导入路径中存在重复文档，等待管理员确认。",
                )
            db.commit()
        raise _duplicate_response(duplicates)

    items: list[UploadBatchItem] = []
    enqueued: list[tuple[Document, Job]] = []
    with sqlite_write_lock:
        concurrent_duplicates: list[dict] = []
        if not payload.confirm_duplicates:
            for spec in specs:
                duplicate_doc = db.execute(select(Document).where(Document.sha256 == spec["sha256"]).limit(1)).scalar_one_or_none()
                if duplicate_doc is not None:
                    concurrent_duplicates.append(
                        {
                            "incoming_filename": spec["path"].name,
                            "existing": _duplicate_info(duplicate_doc).model_dump(mode="json"),
                        }
                    )
            if concurrent_duplicates:
                for duplicate in concurrent_duplicates:
                    existing = duplicate["existing"]
                    _create_upload_log(
                        db,
                        filename=duplicate["incoming_filename"],
                        sha256=existing["sha256"],
                        source="local",
                        uploaded_by=username,
                        status_value="duplicate_blocked",
                        duplicate_of_document_id=existing.get("document_id") or None,
                        message="导入路径中存在重复文档，等待管理员确认。",
                    )
                db.commit()
                raise _duplicate_response(concurrent_duplicates)

        for spec in specs:
            pdf_path = spec["path"]
            document = Document(filename=pdf_path.name, original_path=str(pdf_path), stored_path="", status="uploaded")
            db.add(document)
            db.flush()
            stored_path, sha256, size = import_local_file(document.id, pdf_path)
            document.stored_path = str(stored_path)
            document.sha256 = sha256
            document.size_bytes = size
            document.updated_at = datetime.utcnow()
            db.commit()
            job = _create_parse_job(db, document)
            _create_upload_log(
                db,
                filename=pdf_path.name,
                sha256=sha256,
                source="local",
                uploaded_by=username,
                status_value="queued",
                document_id=document.id,
                message=f"已从服务器路径导入：{pdf_path}",
            )
            db.commit()
            items.append(
                UploadBatchItem(
                    filename=pdf_path.name,
                    status="queued",
                    message="已导入，等待解析",
                    document=DocumentOut.model_validate(document),
                    job=JobOut.model_validate(job),
                )
            )
            enqueued.append((document, job))

    for document, job in enqueued:
        _enqueue_or_fail(
            action="parse",
            db=db,
            document=document,
            job=job,
            enqueue_call=lambda document_id=document.id, job_id=job.id: enqueue_parse(background_tasks, document_id, job_id),
        )

    return UploadBatchResponse(items=items)


@router.get("/documents", response_model=list[DocumentOut])
def list_documents(_: AdminUser, db: Session = Depends(get_session)) -> list[DocumentOut]:
    documents = db.execute(select(Document).order_by(desc(Document.created_at))).scalars().all()
    return [DocumentOut.model_validate(document) for document in documents]


@router.get("/jobs", response_model=list[JobOut])
def list_jobs(_: AdminUser, db: Session = Depends(get_session)) -> list[JobOut]:
    jobs = db.execute(select(Job).order_by(desc(Job.created_at)).limit(120)).scalars().all()
    return [JobOut.model_validate(job) for job in jobs]


@router.get("/upload-logs", response_model=list[UploadLogOut])
def list_upload_logs(_: AdminUser, db: Session = Depends(get_session)) -> list[UploadLogOut]:
    logs = db.execute(select(UploadLog).order_by(desc(UploadLog.created_at)).limit(200)).scalars().all()
    results: list[UploadLogOut] = []
    for item in logs:
        document_status = item.document.status if item.document else None
        results.append(
            UploadLogOut(
                id=item.id,
                filename=item.filename,
                sha256=item.sha256,
                source=item.source,
                uploaded_by=item.uploaded_by,
                status=item.status,
                message=item.message,
                created_at=item.created_at,
                updated_at=item.updated_at,
                document_id=item.document_id,
                duplicate_of_document_id=item.duplicate_of_document_id,
                document_status=document_status,
            )
        )
    return results


@router.get("/documents/{document_id}/review", response_model=DocumentReview)
def document_review(document_id: str, _: AdminUser, db: Session = Depends(get_session)) -> DocumentReview:
    document = db.get(Document, document_id)
    if not document:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="文档不存在")
    jobs = db.execute(select(Job).where(Job.document_id == document_id).order_by(desc(Job.created_at))).scalars().all()
    sample_chunks = (
        db.execute(select(Chunk).where(Chunk.document_id == document_id).order_by(Chunk.page_number).limit(16)).scalars().all()
    )
    chunks = db.execute(select(Chunk).where(Chunk.document_id == document_id).order_by(Chunk.page_number, Chunk.kind)).scalars().all()
    page_assets = (
        db.execute(select(Asset).where(Asset.document_id == document_id, Asset.kind == "page").order_by(Asset.page_number))
        .scalars()
        .all()
    )
    image_assets = (
        db.execute(select(Asset).where(Asset.document_id == document_id, Asset.kind == "image").order_by(Asset.page_number))
        .scalars()
        .all()
    )

    # Deduplicate chunks by content: keep latest created_at per unique content
    seen_chunks: dict[tuple, Chunk] = {}
    for chunk in chunks:
        key = (chunk.page_number, chunk.kind, chunk.content)
        existing = seen_chunks.get(key)
        if existing is None or (chunk.created_at and existing.created_at and chunk.created_at > existing.created_at):
            seen_chunks[key] = chunk
    chunks = list(seen_chunks.values())
    chunks.sort(key=lambda c: (c.page_number, c.kind or ""))

    # Deduplicate page assets: keep latest per page_number
    seen_pages: dict[int, Asset] = {}
    for asset in page_assets:
        existing = seen_pages.get(asset.page_number)
        if existing is None or (asset.created_at and existing.created_at and asset.created_at > existing.created_at):
            seen_pages[asset.page_number] = asset
    page_assets = list(seen_pages.values())
    page_assets.sort(key=lambda a: a.page_number)

    # Deduplicate image assets: keep latest per (page_number, region_index, region_type)
    seen_images: dict[tuple, Asset] = {}
    for asset in image_assets:
        key = (asset.page_number, asset.region_index, asset.region_type)
        existing = seen_images.get(key)
        if existing is None or (asset.created_at and existing.created_at and asset.created_at > existing.created_at):
            seen_images[key] = asset
    image_assets = list(seen_images.values())
    image_assets.sort(key=lambda a: a.page_number)

    return DocumentReview(
        document=DocumentOut.model_validate(document),
        jobs=[JobOut.model_validate(job) for job in jobs],
        stats={
            **(document.parse_stats or {}),
            "total_chunks": len(chunks),
            "approved_chunks": sum(1 for chunk in chunks if chunk.approved),
            "indexed_chunks": sum(1 for chunk in chunks if chunk.indexed),
            "embedded_chunks": sum(1 for chunk in chunks if chunk.embedding),
            "image_embedded_chunks": sum(1 for chunk in chunks if chunk.kind == "image" and chunk.embedding),
        },
        sample_chunks=sample_chunks,
        chunks=chunks,
        page_assets=[_asset_out(asset) for asset in page_assets],
        image_assets=[_asset_out(asset) for asset in image_assets],
    )


@router.patch("/chunks/{chunk_id}", response_model=ChunkOut)
def update_chunk(
    chunk_id: str,
    payload: ChunkUpdateRequest,
    _: AdminUser,
    db: Session = Depends(get_session),
) -> ChunkOut:
    with sqlite_write_lock:
        chunk = db.get(Chunk, chunk_id)
        if not chunk:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="内容块不存在")
        if payload.content is not None:
            chunk.content = payload.content
            chunk.indexed = False
            chunk.embedding = None
            chunk.embedding_model = None
            chunk.embedding_dim = None
            chunk.secondary_embedding = None
            chunk.secondary_embedding_model = None
            chunk.secondary_embedding_dim = None
        if payload.approved is not None:
            chunk.approved = payload.approved
        document = db.get(Document, chunk.document_id)
        if document:
            document.status = "parsed" if document.status == "approved" else document.status
            document.updated_at = datetime.utcnow()
        db.commit()
        db.refresh(chunk)
    return ChunkOut.model_validate(chunk)


@router.delete("/chunks/{chunk_id}")
def delete_chunk(chunk_id: str, _: AdminUser, db: Session = Depends(get_session)) -> dict[str, bool]:
    with sqlite_write_lock:
        chunk = db.get(Chunk, chunk_id)
        if not chunk:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="内容块不存在")
        document = db.get(Document, chunk.document_id)
        db.delete(chunk)
        if document:
            document.status = "parsed" if document.status == "approved" else document.status
            document.updated_at = datetime.utcnow()
        db.commit()
    return {"ok": True}


@router.post("/documents/{document_id}/approve")
def approve_document(
    document_id: str,
    background_tasks: BackgroundTasks,
    _: AdminUser,
    db: Session = Depends(get_session),
) -> dict:
    with sqlite_write_lock:
        document = db.get(Document, document_id)
        if not document:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="文档不存在")
        if document.status not in {"parsed", "approved"}:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="文档尚未完成解析")

        active_job = _active_job(db, document_id=document.id, job_type="index")
        if active_job:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail={
                    "code": "job_already_running",
                    "message": "该文档已有入库任务正在处理中，请稍后刷新查看结果。",
                    "job": JobOut.model_validate(active_job).model_dump(mode="json"),
                },
            )

        job = Job(document_id=document.id, job_type="index", status="queued", progress=0, message="等待入库")
        db.add(job)
        db.commit()
        db.refresh(job)

    _enqueue_or_fail(
        action="index",
        db=db,
        document=document,
        job=job,
        enqueue_call=lambda: enqueue_index(background_tasks, document.id, job.id),
    )
    return {"job": JobOut.model_validate(job)}


@router.post("/documents/{document_id}/reparse")
def reparse_document(
    document_id: str,
    background_tasks: BackgroundTasks,
    _: AdminUser,
    db: Session = Depends(get_session),
) -> dict:
    previous_status = None
    with sqlite_write_lock:
        document = db.get(Document, document_id)
        if not document:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="文档不存在")

        active_job = _active_job(db, document_id=document.id, job_type="parse")
        if active_job or document.status in {"parsing"}:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail={
                    "code": "job_already_running",
                    "message": "该文档已有重新解析任务正在处理中，请稍后刷新查看结果。",
                    "job": JobOut.model_validate(active_job).model_dump(mode="json") if active_job else None,
                },
            )

        previous_status = document.status
        document.status = "uploaded"
        document.updated_at = datetime.utcnow()
        document.error_message = None
        job = Job(document_id=document.id, job_type="parse", status="queued", progress=0, message="等待重新解析")
        db.add(job)
        db.commit()
        db.refresh(job)

    try:
        _enqueue_or_fail(
            action="parse",
            db=db,
            document=document,
            job=job,
            enqueue_call=lambda: enqueue_parse(background_tasks, document.id, job.id),
        )
    except HTTPException as exc:
        with sqlite_write_lock:
            document.status = previous_status
            document.updated_at = datetime.utcnow()
            db.commit()
        raise exc

    return {"job": JobOut.model_validate(job)}


@router.delete("/documents/{document_id}")
def delete_document(document_id: str, _: AdminUser, db: Session = Depends(get_session)) -> dict[str, bool]:
    with sqlite_write_lock:
        document = db.get(Document, document_id)
        if not document:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="文档不存在")
        active_jobs = _active_jobs(db, document_id=document_id, job_types=("parse", "index"))
        if active_jobs:
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="文档正在处理中，请稍后删除")
        db.delete(document)
        db.commit()
    VectorStore().delete_document(document_id)
    remove_document_storage(document_id)
    return {"ok": True}


@router.get("/jobs/{job_id}", response_model=JobOut)
def get_job(job_id: str, _: AdminUser, db: Session = Depends(get_session)) -> JobOut:
    job = db.get(Job, job_id)
    if not job:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="任务不存在")
    return JobOut.model_validate(job)


@router.get("/assets/{asset_id}")
def get_asset(asset_id: str) -> FileResponse:
    with SessionLocal() as db:
        asset = db.get(Asset, asset_id)
        if not asset:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="资源不存在")
        path = Path(asset.path)
        media_type = asset.mime_type
    if not path.exists():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="资源不存在")
    return FileResponse(path, media_type=media_type, filename=path.name)


@router.get("/documents/{document_id}/file")
def get_document_file(document_id: str) -> FileResponse:
    with SessionLocal() as db:
        document = db.get(Document, document_id)
        if not document:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="文档不存在")
        path = Path(document.stored_path)
        filename = document.filename
    if not path.exists():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="文档不存在")
    return FileResponse(path, media_type="application/pdf", filename=filename)
