# 机房运维 PDF 多模态检索系统

本项目是一个本地优先的 PDF 多模态检索 MVP，面向机房运维、工程建设、勘察设计类资料。它提供：

- 前台中文语义检索、知识问答、图片/图纸定位、来源文件与页码引用。
- 后台 PDF 上传/本地目录导入、自动解析切分、文档级审核确认后入库。
- 文本块与图像/页面块双索引，支持 Qdrant，也支持本机开发时的数据库回退检索。
- Ollama 本地 embedding/LLM 适配；未安装模型时使用确定性本地 embedding 和抽取式答案兜底。
- OCR 适配层，优先支持 PaddleOCR，也可用 Tesseract；未安装 OCR 时会保留页面截图并标记 `needs_ocr`。

## 正式模式启动

当前仓库的默认 `.env` 已切到正式模式，使用以下容器化架构：

- `Postgres`：共享业务数据库
- `Redis + RQ worker`：文档解析、入库和重建索引任务队列
- `Qdrant`：向量库
- `API + worker + frontend`：统一由 `docker compose` 管理
- `OCR`：本轮默认走云 OCR，不要求本机安装 PaddleOCR / Tesseract

启动前先确认：

1. 已启动 Docker Desktop
2. `.env` 中的云模型、embedding、云 OCR 相关密钥可用
3. `CLOUD_OCR_ENABLED=true`
4. `USE_RQ=true`
5. `QDRANT_URL=http://qdrant:6333`

启动命令：

```bash
docker compose up --build
```

常用访问地址：

- 检索站：`http://localhost:5173`
- 后台管理：`http://localhost:5173/admin`
- 上传日志：`http://localhost:5173/admin/logs`
- API 文档：`http://localhost:8000/docs`
- 健康检查：`http://localhost:8000/api/health`
- Qdrant 控制台：`http://localhost:6333/dashboard`

正式模式启动后，后台“系统状态”里应至少看到：

- Qdrant 已启用
- RQ / Redis 已启用
- OCR 已启用

默认管理员账号见 `.env.example`，上线前必须修改。

## 本机开发

本机开发默认使用统一启动入口。它会自动：

- 检查 Docker Desktop
- 自动拉起 `Redis` 与 `Qdrant`
- 启动本机 `RQ worker`
- 启动本机 `API`
- 启动本机 `Frontend`

启动命令：

```powershell
.\dev-up.ps1
```

或：

```bash
make dev-up
```

停止本机服务：

```powershell
.\dev-down.ps1
```

如果还要顺手停止 `Redis` 和 `Qdrant` 容器：

```powershell
.\dev-down.ps1 -StopDependencies
```

本机开发会显式叠加 `.env.dev`，默认使用：

- `DATABASE_URL=sqlite:///./data/app.db`
- `REDIS_URL=redis://localhost:6379/0`
- `QDRANT_URL=http://localhost:6333`
- `USE_RQ=true`
- `STORAGE_DIR=./storage`

因此后台状态页默认应看到：

- `Qdrant` 已启用
- `RQ / Redis` 已启用

文档解析、入库和重建索引任务会进入真实队列，不再退回进程内任务。

如仍需分别单独启动前后端，可以继续使用原有命令，但这不再是推荐默认方式：

```bash
cd backend
uvicorn app.main:app --reload
```

```bash
cd frontend
npm run dev
```

## OCR

第一版 OCR 是可插拔适配器。若要启用 PaddleOCR：

```bash
cd backend
pip install -r requirements-ocr.txt
OCR_BACKEND=paddle uvicorn app.main:app --reload
```

如果 OCR 未安装，扫描 PDF 仍会保存页面截图、生成待 OCR 标记，并可在后台审核页看到低置信度提示。

## 样例验收查询

上传并审核样例 PDF 后，可用以下查询验证：

- `运维中心机房图`
- `机房楼层高分析图`
- `园区总平面图`
- `BIM 运维管理`
- `机房工作流程`

返回结果应包含答案、来源文件、页码、文字片段，以及相关页面/图片预览。
