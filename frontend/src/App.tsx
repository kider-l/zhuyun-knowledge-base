import { FormEvent, useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  ArrowUp,
  BookOpen,
  CheckCircle2,
  ChevronLeft,
  ChevronRight,
  Database,
  FileSearch,
  FileText,
  History,
  Image,
  Loader2,
  LogIn,
  MessageSquarePlus,
  MoreHorizontal,
  PanelLeftClose,
  PanelLeftOpen,
  RefreshCcw,
  Search,
  ServerCog,
  Trash2,
  Upload,
  Workflow,
  X,
  ZoomIn,
  ZoomOut,
} from "lucide-react";
import {
  API_BASE,
  ApiError,
  approveDocument,
  assetUrl,
  deleteChunk,
  deleteDocument,
  documentFileUrl,
  generateAnswer,
  getReview,
  getSystemStatus,
  importLocal,
  listDocuments,
  listJobs,
  listUploadLogs,
  login,
  readableError,
  rebuildIndex,
  reparseDocument,
  searchDocuments,
  updateChunk,
  uploadDocuments,
} from "./api";
import type {
  AnswerResponse,
  AssetOut,
  ChunkOut,
  DocumentOut,
  DocumentReview,
  JobOut,
  SearchResult,
  SearchResponse,
  SystemStatus,
  UploadLogOut,
} from "./types";

type Mode = "all" | "images" | "process" | "text";
type KindFilter = "all" | "text" | "image";
type AdminSection = "documents" | "logs" | "status";

interface ChatTurn {
  id: string;
  query: string;
  response: AnswerResponse;
  createdAt: string;
}

interface ChatSession {
  id: string;
  summary: string;
  createdAt: string;
  turns: ChatTurn[];
}

interface CitationSelection {
  sourceIndex: number;
  turnId: string;
}

interface DuplicatePromptState {
  files: File[];
  localPath?: string;
  items: Array<{
    incoming_filename: string;
    existing: {
      document_id?: string;
      filename: string;
      sha256: string;
      status: string;
      uploaded_at?: string;
    };
  }>;
}

interface ImagePreviewState {
  items: Array<{ result: SearchResult; index: number }>;
  currentIndex: number;
}

interface ReviewPreviewState {
  asset: AssetOut;
  documentName: string;
}


const suggestedQuestions = [
  "C4 疏散路线及应急物资存放点在哪里？",
  "施工安全承诺书有哪些关键要求？",
  "机房平面图或图纸在哪几页？",
  "应急救援流程和职责如何划分？",
  "巡检制度里对关键设备有哪些要求？",
  "安全生产应急预案里现场疏散路线怎么走？",
];

const CHAT_STORAGE_KEY = "graphsearch_chat_sessions_v1";
const EMPTY_SESSION_SUMMARY = "新对话";
const SUMMARY_STOP_WORDS = ["请问", "麻烦", "帮我", "一下", "您好", "你好", "请帮", "我想", "我想知道", "告诉我", "帮忙", "给我", "是否", "有没有"];

function clampZoom(value: number, min = 0.8, max = 5.5): number {
  return Math.min(max, Math.max(min, Number(value.toFixed(2))));
}

function highlightNotice(result: SearchResult): { tone: "warning" | "info"; text: string } | null {
  const precision = String(result.metadata.highlight_precision || "none");
  if (precision === "approximate") {
    return {
      tone: "warning",
      text: "当前黄色标注为估算定位，用于大致指示相关区域，建议结合页内正文和原始 PDF 一起核对。",
    };
  }
  if (precision === "none") {
    return {
      tone: "warning",
      text: "当前页缺少可用坐标信息，暂时无法精确标黄。",
    };
  }
  if (precision === "page_search") {
    return {
      tone: "info",
      text: "当前标黄来自页内搜索匹配，通常可用，但局部位置仍可能有轻微偏差。",
    };
  }
  return null;
}

function navigateTo(path: string) {
  window.history.pushState({}, "", path);
  window.dispatchEvent(new PopStateEvent("popstate"));
}

function formatSize(size: number): string {
  if (size > 1024 * 1024 * 1024) return `${(size / 1024 / 1024 / 1024).toFixed(1)} GB`;
  if (size > 1024 * 1024) return `${(size / 1024 / 1024).toFixed(1)} MB`;
  return `${(size / 1024).toFixed(1)} KB`;
}

function formatTime(value?: string | null): string {
  if (!value) return "-";
  // 无时区标记时视为 UTC 时间，避免被 JS 误解析为本地时间
  const normalized = /(Z|[+-]\d{2}:?\d{2})$/.test(value) ? value : value + "Z";
  const date = new Date(normalized);
  return Number.isNaN(date.getTime()) ? value : date.toLocaleString("zh-CN");
}

function statusLabel(status: string): string {
  const labels: Record<string, string> = {
    uploaded: "已上传",
    parsing: "解析中",
    parsed: "待确认",
    approved: "已入库",
    failed: "失败",
    queued: "排队中",
    running: "处理中",
    succeeded: "已完成",
    duplicate_blocked: "重复待确认",
    batch_duplicate: "本批次重复",
    same_name_size: "同名同大小待确认",
  };
  return labels[status] || status;
}

function jobInProgress(job?: JobOut | null): boolean {
  return Boolean(job && (job.status === "queued" || job.status === "running"));
}

function latestJobForDocument(jobs: JobOut[], documentId: string, jobType: string): JobOut | null {
  return jobs.find((job) => job.document_id === documentId && job.job_type === jobType) || null;
}

function friendlyTaskError(message?: string | null): string | null {
  if (!message) return null;
  if (message.includes("DuplicatePreparedStatement")) {
    return "数据库连接在任务执行时发生冲突，本次任务未完成。修复后可重新解析或再次确认入库。";
  }
  return message;
}

function itemKindLabel(result: { kind: string; metadata: Record<string, unknown> }): string {
  if (result.metadata.asset_kind === "table") return "表格";
  if (result.metadata.asset_kind === "figure_region") return "切图区域";
  if (result.metadata.asset_kind === "whole_figure_fallback") return "整图兜底";
  if (result.metadata.asset_kind === "figure") return "整幅图";
  if (result.metadata.asset_kind === "embedded_image") return "嵌入图片";
  if (result.metadata.asset_kind === "page") return "页面截图";
  if (result.kind === "image") return "图纸/页面";
  return "文本";
}

function cleanAnswerText(text: string): string {
  return text
    .replace(/[ \t]+/g, " ")
    .replace(/\n{3,}/g, "\n\n")
    .trim();
}

function isImageSeekingQuery(query: string): boolean {
  const normalized = query.replace(/\s+/g, "");
  const keywords = [
    "图纸", "平面图", "示意图", "附件图", "位置图", "路线图",
    "照片", "截图", "看图", "看看图", "发图", "发我",
    "图发我", "给我图", "给我看", "现场图", "现场照片",
    "现场图片", "图片", "给我图看",
  ];
  return keywords.some((keyword) => normalized.includes(keyword));
}

function isPureImageQuery(query: string): boolean {
  const normalized = query.replace(/\s+/g, "");
  const keywords = ["只要图", "只看图", "只发图", "仅图片", "纯图", "仅看图"];
  return keywords.some((keyword) => normalized.includes(keyword));
}

function citationIndexes(answer: string, max: number): number[] {
  const indexes: number[] = [];
  for (const match of answer.matchAll(/\[(\d+)\]/g)) {
    const index = Number(match[1]) - 1;
    if (index >= 0 && index < max && !indexes.includes(index)) indexes.push(index);
  }
  return indexes;
}

function displayableResults(response: AnswerResponse): Array<{ result: SearchResult; index: number; cited: boolean }> {
  const allResults = response.results
    .map((result, index) => ({ result, index }))
    .filter((item) => {
      if (item.result.kind !== "image" || !item.result.asset_url) return false;
      const meta = item.result.metadata as Record<string, unknown>;
      const imageClass = String(meta?.image_class || "DOC_IMAGE");
      if (!imageClass && meta?.paper_record === true) return false;
      const assetKind = String(meta?.asset_kind || "");
      return ["page", "embedded_image", "figure", "figure_region", "whole_figure_fallback", "table"].includes(assetKind);
    });
  const cited = citationIndexes(response.answer, response.results.length);
  const citedSet = new Set(cited);

  function sourceKey(item: { result: SearchResult }): string {
    return item.result.asset_url || item.result.chunk_id;
  }

  const seen = new Set<string>();
  const seenOverlapPage = new Set<string>();
  // 去重前把被引用的排前面，确保被引用的优先保留
  const merged = allResults
    .slice()
    .sort((a, b) => {
      const aCited = citedSet.has(a.index) ? 0 : 1;
      const bCited = citedSet.has(b.index) ? 0 : 1;
      return aCited - bCited;
    })
    .filter((item) => {
    const key = sourceKey(item);
    if (seen.has(key)) return false;
    seen.add(key);
    const meta = item.result.metadata as Record<string, unknown>;
    const assetKind = String(meta?.asset_kind || "");
    if (["figure", "figure_region", "whole_figure_fallback"].includes(assetKind)) {
      const pageKey = `${item.result.document_id}:${item.result.page_number}`;
      if (seenOverlapPage.has(pageKey)) return false;
      seenOverlapPage.add(pageKey);
    }
    return true;
  });
  // 被引用的按原始索引顺序排列（匹配答案中 [1][2][3] 顺序），未引用的排在后面
  return merged
    .sort((a, b) => {
      const aCited = citedSet.has(a.index);
      const bCited = citedSet.has(b.index);
      if (aCited && bCited) return a.index - b.index;
      if (aCited) return -1;
      if (bCited) return 1;
      return a.index - b.index;
    })
    .slice(0, 12)
    .map((item) => ({ ...item, cited: citedSet.has(item.index) }));
}

function stripSummaryNoise(query: string): string {
  let cleaned = query.trim();
  for (const phrase of SUMMARY_STOP_WORDS) {
    cleaned = cleaned.split(phrase).join("");
  }
  return cleaned.replace(/[？?。，“”’'":：、，.\s]+/g, " ").trim();
}

function buildSessionSummary(query: string): string {
  const cleaned = stripSummaryNoise(query);
  if (!cleaned) return EMPTY_SESSION_SUMMARY;
  const condensed = cleaned.replace(/\s+/g, " ");
  return condensed.length > 18 ? `${condensed.slice(0, 18)}…` : condensed;
}

function sessionMeta(session: ChatSession): string {
  if (!session.turns.length) return "新会话";
  return `${session.turns.length} 轮对话`;
}

function createChatSession(summary = EMPTY_SESSION_SUMMARY): ChatSession {
  return {
    id: `session-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`,
    summary,
    createdAt: new Date().toISOString(),
    turns: [],
  };
}

function serializeChatSessions(sessions: ChatSession[], activeSessionId: string | null): void {
  try {
    localStorage.setItem(CHAT_STORAGE_KEY, JSON.stringify({ activeSessionId, sessions }));
  } catch {
    // ignore local storage failures
  }
}

function parseStoredChatSessions(): { activeSessionId: string | null; sessions: ChatSession[] } {
  try {
    const raw = localStorage.getItem(CHAT_STORAGE_KEY);
    if (!raw) return { activeSessionId: null, sessions: [] };
    const parsed = JSON.parse(raw) as { activeSessionId?: unknown; sessions?: unknown };
    const sessions = Array.isArray(parsed.sessions)
      ? parsed.sessions.flatMap((item) => {
          if (!item || typeof item !== "object") return [];
          const session = item as Partial<ChatSession>;
          if (typeof session.id !== "string" || typeof session.summary !== "string" || typeof session.createdAt !== "string") {
            return [];
          }
          const turns = Array.isArray(session.turns)
            ? session.turns.flatMap((turn) => {
                if (!turn || typeof turn !== "object") return [];
                const candidate = turn as Partial<ChatTurn>;
                if (
                  typeof candidate.id !== "string" ||
                  typeof candidate.query !== "string" ||
                  typeof candidate.createdAt !== "string" ||
                  !candidate.response ||
                  typeof candidate.response !== "object"
                ) {
                  return [];
                }
                return [
                  {
                    id: candidate.id,
                    query: candidate.query,
                    response: candidate.response as AnswerResponse,
                    createdAt: candidate.createdAt,
                  },
                ];
              })
            : [];
          return [
            {
              id: session.id,
              summary: session.summary || EMPTY_SESSION_SUMMARY,
              createdAt: session.createdAt,
              turns,
            },
          ];
        })
      : [];
    const activeSessionId = typeof parsed.activeSessionId === "string" ? parsed.activeSessionId : null;
    return { activeSessionId, sessions };
  } catch {
    return { activeSessionId: null, sessions: [] };
  }
}

function GeneratingIndicator() {
  return (
    <div className="generating-indicator" aria-live="polite">
      <span className="generating-ring" aria-hidden="true" />
      <span>正在生成回答</span>
    </div>
  );
}

function ModalFrame({
  title,
  subtitle,
  onClose,
  children,
  actions,
  cardClassName,
}: {
  title: string;
  subtitle?: string;
  onClose: () => void;
  children: React.ReactNode;
  actions?: React.ReactNode;
  cardClassName?: string;
}) {
  return (
    <div className="modal-backdrop" onClick={onClose}>
      <div className={`modal-card${cardClassName ? ` ${cardClassName}` : ''}`} onClick={(event) => event.stopPropagation()}>
        <div className="modal-head">
          <div>
            <h2>{title}</h2>
            {subtitle && <p>{subtitle}</p>}
          </div>
          <button type="button" className="icon-button" onClick={onClose}>
            <X size={18} />
          </button>
        </div>
        <div className="modal-body">{children}</div>
        {actions && <div className="modal-actions">{actions}</div>}
      </div>
    </div>
  );
}

function AnswerWithCitations({
  answer,
  sourceCount,
  activeIndex,
  onSelect,
}: {
  answer: string;
  sourceCount: number;
  activeIndex: number | null;
  onSelect: (index: number) => void;
}) {
  const cleaned = cleanAnswerText(answer);
  const blocks = cleaned.split(/\n\n+/).filter(Boolean);

  function renderParagraph(text: string, keyBase: string): Array<string | JSX.Element> {
    const nodes: Array<string | JSX.Element> = [];
    const citationPattern = /\[(\d+)\]/g;
    let cursor = 0;
    let match: RegExpExecArray | null;
    let key = 0;

    while ((match = citationPattern.exec(text)) !== null) {
      const citationNumber = Number(match[1]);
      const sourceIndex = citationNumber - 1;
      if (match.index > cursor) {
        nodes.push(text.slice(cursor, match.index));
      }
      if (sourceIndex >= 0 && sourceIndex < sourceCount) {
        nodes.push(
          <button
            key={`${keyBase}-cite-${key++}`}
            className={activeIndex === sourceIndex ? "citation-mark active" : "citation-mark"}
            type="button"
            onClick={() => onSelect(sourceIndex)}
          >
            {citationNumber}
          </button>,
        );
      } else {
        nodes.push(match[0]);
      }
      cursor = match.index + match[0].length;
    }
    if (cursor < text.length) {
      nodes.push(text.slice(cursor));
    }
    return nodes;
  }

  // 检测自然语言段落类型（三级标题）
  function paragraphKind(text: string): "h1" | "h2" | "h3" | "data" | "body" | "note" {
    const stripped = text.replace(/\[\d+\]/g, "").trim();
    // 一、二、三 → H1
    if (/^[一二三四五六七八九十]+[、.．]\s*/.test(stripped)) return "h1";
    // 1. 2. 3. 开头 → H2
    if (/^\d+[.．、]\s*\S/.test(stripped)) return "h2";
    // （1）（2）（3）或 (1) (2) (3) → H3（仅短行，超过12字为内容）
    if (/^[（(]\d+[)）]/.test(stripped) && stripped.length <= 12) return "h3";
    if (/^[A-Z][.．]\s/.test(stripped) && stripped.length <= 10) return "h3";
    // "说明：" → 辅助标注
    if (/^说明[：:]/.test(stripped)) return "note";
    // "项目名：值" 短行 → 数据行（但如果不是表头）
    if (/：/.test(stripped) && stripped.length <= 25) return "data";
    return "body";
  }

  // 将 "xx：" 中的 xx 加粗
  function renderWithLabelBold(text: string, keyBase: string): Array<string | JSX.Element> {
    const labelMatch = text.match(/^([^：:]+)([：:])(.*)/);
    if (labelMatch) {
      const [, label, colon, rest] = labelMatch;
      return [
        <strong key={`${keyBase}-lbl`} className="data-label">{label}</strong>,
        colon,
        ...renderParagraph(rest, `${keyBase}-rest`),
      ];
    }
    return renderParagraph(text, keyBase);
  }

  return (
    <>
      {blocks.map((block, blockIndex) => {
        const blockKey = `b-${blockIndex}`;
        const lines = block.split("\n").filter(Boolean);
        // 跳过 "--------" 分隔线
        if (/^-{5,}$/.test(lines[0])) return null;
        // 逐行检测，每行独立渲染
        const lineTypes = lines.map((line) => paragraphKind(line));
        // 合并相邻同类型行到一个 p 标签
        const groups: Array<{ type: string; lines: string[] }> = [];
        for (let i = 0; i < lines.length; i++) {
          const type = lineTypes[i];
          const last = groups[groups.length - 1];
          if (last && last.type === type) {
            last.lines.push(lines[i]);
          } else {
            groups.push({ type, lines: [lines[i]] });
          }
        }
        return groups.map((group, gi) => {
          const cls = group.type === "h1" ? "answer-h1"
                    : group.type === "h2" ? "answer-h2"
                    : group.type === "h3" ? "answer-h3"
                    : group.type === "data" ? "answer-data"
                    : group.type === "note" ? "answer-note"
                    : "answer-paragraph";
          const useLabelBold = group.type === "data";
          return (
            <p key={`${blockKey}-g${gi}`} className={cls}>
              {group.lines.flatMap((line, li) => {
                const text = useLabelBold ? renderWithLabelBold(line, `${blockKey}-g${gi}-${li}`) : renderParagraph(line, `${blockKey}-g${gi}`);
                if (li === 0) return text;
                return [<br key={`${blockKey}-br-${gi}-${li}`} />, ...text];
              })}
            </p>
          );
        });
      })}
    </>
  );
}

function CitationModal({
  result,
  index,
  onClose,
}: {
  result: SearchResult;
  index: number;
  onClose: () => void;
}) {
  const preview = assetUrl(result.asset_url);
  const [zoom, setZoom] = useState(1.0);
  const boxes = result.highlight_boxes || [];
  const stageRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    setZoom(1.0);
    // 打开时水平滚动到中间
    requestAnimationFrame(() => {
      if (!stageRef.current) return;
      stageRef.current.scrollLeft = (stageRef.current.scrollWidth - stageRef.current.clientWidth) / 2;
    });
  }, [result.chunk_id]);

  function adjustZoom(delta: number) {
    setZoom((value) => clampZoom(value + delta, 0.9, 4.8));
  }

  if (!preview) return null;

  return (
    <div className="lightbox-backdrop" onClick={onClose}>
      <div className="lightbox-card citation-modal-card" onClick={(event) => event.stopPropagation()}>
        <div className="lightbox-head">
          <div>
            <span className="panel-kicker">引用 {index + 1}</span>
            <h3>{result.document_name}</h3>
            <p>
              第 {result.page_number} 页 · {itemKindLabel(result)}
            </p>
          </div>
          <div className="lightbox-head-actions">
            <span className="zoom-percent">{Math.round(zoom * 100)}%</span>
            <button type="button" className="icon-button" onClick={() => adjustZoom(-0.2)} disabled={zoom <= 0.9}>
              <ZoomOut size={16} />
            </button>
            <button type="button" className="icon-button" onClick={() => adjustZoom(0.2)} disabled={zoom >= 4.8}>
              <ZoomIn size={16} />
            </button>
            <button type="button" className="icon-button" onClick={onClose}>
              <X size={18} />
            </button>
          </div>
        </div>
        <div className="lightbox-body citation-modal-body">
          <div className="lightbox-image-stage" ref={stageRef}>
            <div className="lightbox-image-zoom" style={{ width: `${Math.max(100, zoom * 100)}%` }}>
              <img src={preview} alt={`第 ${result.page_number} 页引用预览`} onError={(e) => { (e.currentTarget as HTMLElement).style.setProperty('display', 'none'); }} />
              {boxes.map((box, boxIndex) => (
                <span
                  className="citation-preview-highlight"
                  key={`${result.chunk_id}-${boxIndex}`}
                  style={{
                    left: `${box.x * 100}%`,
                    top: `${box.y * 100}%`,
                    width: `${box.width * 100}%`,
                    height: `${box.height * 100}%`,
                  }}
                />
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

function ImageLightbox({
  preview,
  onClose,
}: {
  preview: ImagePreviewState;
  onClose: () => void;
}) {
  const [currentIndex, setCurrentIndex] = useState(preview.currentIndex);
  const [zoom, setZoom] = useState(1.0);
  const item = preview.items[currentIndex];
  const imageUrl = item ? assetUrl(item.result.asset_url) : null;
  const boxes = item?.result.highlight_boxes || [];
  const imageStageRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    setCurrentIndex(preview.currentIndex);
  }, [preview.currentIndex, preview.items]);

  useEffect(() => {
    setZoom(1.0);
    requestAnimationFrame(() => {
      if (!imageStageRef.current) return;
      imageStageRef.current.scrollLeft = (imageStageRef.current.scrollWidth - imageStageRef.current.clientWidth) / 2;
    });
  }, [item?.result.chunk_id]);

  if (!item || !imageUrl) return null;

  function step(delta: number) {
    setCurrentIndex((value) => {
      const next = value + delta;
      if (next < 0) return preview.items.length - 1;
      if (next >= preview.items.length) return 0;
      return next;
    });
  }

  function adjustZoom(delta: number) {
    setZoom((value) => clampZoom(value + delta, 0.8, 5.5));
  }

  return (
    <div className="lightbox-backdrop" onClick={onClose}>
      <div className="lightbox-card" onClick={(event) => event.stopPropagation()}>
        <div className="lightbox-head">
          <div>
            <span className="panel-kicker">引用 {item.index + 1}</span>
            <h3>{item.result.document_name}</h3>
            <p>
              第 {item.result.page_number} 页 · {itemKindLabel(item.result)}
            </p>
          </div>
          <div className="lightbox-head-actions">
            <span className="zoom-percent">{Math.round(zoom * 100)}%</span>
            <button type="button" className="icon-button" onClick={() => adjustZoom(-0.2)} disabled={zoom <= 0.8}>
              <ZoomOut size={16} />
            </button>
            <button type="button" className="icon-button" onClick={() => adjustZoom(0.2)} disabled={zoom >= 5.5}>
              <ZoomIn size={16} />
            </button>
            <button type="button" className="icon-button" onClick={onClose}>
              <X size={18} />
            </button>
          </div>
        </div>
        <div className="lightbox-body">
          {preview.items.length > 1 && (
            <button type="button" className="lightbox-nav prev" onClick={() => step(-1)}>
              <ChevronLeft size={20} />
            </button>
          )}
          <div className="lightbox-image-stage" ref={imageStageRef}>
            <div className="lightbox-image-zoom" style={{ width: `${Math.max(100, zoom * 100)}%` }}>
              <img src={imageUrl} alt={`第 ${item.result.page_number} 页放大预览`} onError={(e) => { (e.currentTarget as HTMLElement).style.setProperty('display', 'none'); }} />
              {boxes.map((box, boxIndex) => (
                <span
                  className="citation-preview-highlight"
                  key={`${item.result.chunk_id}-${boxIndex}`}
                  style={{
                    left: `${box.x * 100}%`,
                    top: `${box.y * 100}%`,
                    width: `${box.width * 100}%`,
                    height: `${box.height * 100}%`,
                  }}
                />
              ))}
            </div>
          </div>
          {preview.items.length > 1 && (
            <button type="button" className="lightbox-nav next" onClick={() => step(1)}>
              <ChevronRight size={20} />
            </button>
          )}
        </div>
      </div>
    </div>
  );
}

// 弱相关内容关键词——匹配到的图片即使被 LLM 引用，也归入"引用图片资料"区
const WEAK_IMAGE_KEYWORDS = [
  "笔录", "询问人", "被询问人", "谈话记录", "签到", "签字", "盖章", "签章",
  "登记表", "审批表", "检查表", "记录表", "申请表",
  "巡查记录", "值班记录", "维修记录", "保养记录",
  "扫描件", "复印件",
  "备案", "存档", "归档", "档案",
  "通知书", "告知书", "确认书",
  "新闻", "报道", "日报", "周报", "月报", "简报",
  "公告", "公示", "声明",
  // 示意图、封面、目录、装饰等非现场类内容
  "示意图", "装饰", "背景", "背景图",
  "封面", "目录", "页眉", "页脚",
];

function isWeakImage(item: { result: SearchResult }): boolean {
  const meta = item.result.metadata as Record<string, unknown>;
  // 后端已标注的 paper_record（笔录、登记表等文书类）
  if (meta?.paper_record === true) return true;
  // 装饰性图片（logo、水印、背景图案等 VLM 分类）
  if (String(meta?.image_class || "") === "SCENE_IMAGE") return true;
  // VLM 判定为不可用于问答
  if (meta?.usable_for_qa === false) return true;
  // 内容片段命中弱相关关键词
  const snippet = item.result.snippet || "";
  if (WEAK_IMAGE_KEYWORDS.some((kw) => snippet.includes(kw))) return true;
  // 区域摘要（region_summary）也检查
  const regionSummary = String(meta?.region_summary || "");
  if (regionSummary && WEAK_IMAGE_KEYWORDS.some((kw) => regionSummary.includes(kw))) return true;
  // 核心主题（core_topic）也检查
  const coreTopic = String(meta?.core_topic || "");
  if (coreTopic && WEAK_IMAGE_KEYWORDS.some((kw) => coreTopic.includes(kw))) return true;
  return false;
}

function InlineEvidenceGallery({
  items,
  onOpen,
}: {
  items: Array<{ result: SearchResult; index: number; cited: boolean }>;
  onOpen: (index: number) => void;
}) {
  if (!items.length) return null;

  // 最高分为参照；得分低于此比例的被归入引用区
  const maxScore = Math.max(...items.map(i => i.result.score), 0.01);
  const SCORE_RATIO_FLOOR = 0.35;

  // 强相关图片（被 LLM 引用 + 非弱内容 + 得分未显著落后）
  // 弱相关图片 = 其余全部归入"引用图片资料"区
  const strongItems = items.filter((item) => {
    if (!item.cited) return false;
    if (isWeakImage(item)) return false;
    if (item.result.score < maxScore * SCORE_RATIO_FLOOR) return false;
    return true;
  });
  const refItems = items.filter((item) => {
    return !item.cited || isWeakImage(item) || item.result.score < maxScore * SCORE_RATIO_FLOOR;
  });

  return (
    <section className="inline-evidence-gallery">
      {strongItems.length > 0 && (
        <div className="inline-evidence-strip">
          {strongItems.map((item, thumbIndex) => {
            const imageUrl = assetUrl(item.result.asset_url);
            if (!imageUrl) return null;
            return (
              <button
                type="button"
                key={`${item.result.chunk_id}-${item.index}`}
                className="inline-evidence-thumb"
                onClick={() => onOpen(items.indexOf(item))}
              >
                <div className="inline-evidence-image-wrap">
                  <img src={imageUrl} alt={`${item.result.document_name} 第 ${item.result.page_number} 页`} onError={(e) => { (e.currentTarget.closest('button') as HTMLElement)?.style.setProperty('display', 'none'); }} />
                </div>
              </button>
            );
          })}
        </div>
      )}
      {refItems.length > 0 && (
        <>
          <div className="inline-evidence-group-label">引用图片资料</div>
          <div className="inline-evidence-strip">
            {refItems.map((item, thumbIndex) => {
              const imageUrl = assetUrl(item.result.asset_url);
              if (!imageUrl) return null;
              return (
                <button
                  type="button"
                  key={`${item.result.chunk_id}-${item.index}`}
                  className="inline-evidence-thumb"
                  onClick={() => onOpen(items.indexOf(item))}
                >
                  <div className="inline-evidence-image-wrap">
                    <img src={imageUrl} alt={`${item.result.document_name} 第 ${item.result.page_number} 页`} onError={(e) => { (e.currentTarget.closest('button') as HTMLElement)?.style.setProperty('display', 'none'); }} />
                  </div>
                </button>
              );
            })}
          </div>
        </>
      )}
    </section>
  );
}

function SessionActionMenu({
  session,
  open,
  onToggle,
  onDelete,
}: {
  session: ChatSession;
  open: boolean;
  onToggle: () => void;
  onDelete: () => void;
}) {
  return (
    <div className={open ? "session-actions open" : "session-actions"}>
      <button type="button" className="session-menu-button" aria-label={`会话 ${session.summary} 更多操作`} onClick={onToggle}>
        <MoreHorizontal size={20} />
      </button>
      {open && (
        <div className="session-action-popover" role="menu">
          <button type="button" className="session-action-item disabled" role="menuitem" onClick={onToggle}>
            导出
          </button>
          <button type="button" className="session-action-item danger" role="menuitem" onClick={onDelete}>
            删除
          </button>
        </div>
      )}
    </div>
  );
}

function AppSidebar({
  sessions,
  activeSessionId,
  collapsed,
  onToggle,
  onNewQuestion,
  onOpenSession,
  onDeleteSession,
}: {
  sessions: ChatSession[];
  activeSessionId: string | null;
  collapsed: boolean;
  onToggle: () => void;
  onNewQuestion: () => void;
  onOpenSession: (id: string) => void;
  onDeleteSession: (id: string) => void;
}) {
  const [openSessionMenuId, setOpenSessionMenuId] = useState<string | null>(null);
  const [pendingDeleteSessionId, setPendingDeleteSessionId] = useState<string | null>(null);

  const pendingDeleteSession = pendingDeleteSessionId ? sessions.find((s) => s.id === pendingDeleteSessionId) || null : null;

  return (
    <aside className={`search-sidebar${collapsed ? " collapsed" : ""}`}>
      <div className="sidebar-header-row">
        {!collapsed && (
          <button className="new-question" onClick={onNewQuestion}>
            <MessageSquarePlus size={20} />
            新对话
          </button>
        )}
        <button type="button" className="sidebar-toggle" onClick={onToggle} aria-label={collapsed ? "展开侧边栏" : "收起侧边栏"}>
          {collapsed ? <PanelLeftOpen size={18} /> : <PanelLeftClose size={18} />}
        </button>
      </div>
      {!collapsed && (
        <>
          <div className="sidebar-title">
            <History size={14} />
            会话列表
          </div>
          <div className="question-history">
            {sessions.map((session) => (
              <div key={session.id} className={activeSessionId === session.id ? "session-history-item active" : "session-history-item"}>
                <button type="button" className="session-history-button" onClick={() => onOpenSession(session.id)}>
                  <strong>{session.summary}</strong>
                </button>
                <SessionActionMenu
                  session={session}
                  open={openSessionMenuId === session.id}
                  onToggle={() => setOpenSessionMenuId((current) => (current === session.id ? null : session.id))}
                  onDelete={() => {
                    setOpenSessionMenuId(null);
                    setPendingDeleteSessionId(session.id);
                  }}
                />
              </div>
            ))}
            {!sessions.length && <p>暂无会话记录</p>}
          </div>
        </>
      )}
      {pendingDeleteSession && (
        <ModalFrame
          title="确认删除会话"
          onClose={() => setPendingDeleteSessionId(null)}
          cardClassName="delete-session-modal"
          actions={
            <>
              <button type="button" className="cancel-button" onClick={() => setPendingDeleteSessionId(null)}>
                取消
              </button>
              <button type="button" className="danger-button" onClick={() => {
                onDeleteSession(pendingDeleteSession.id);
                setPendingDeleteSessionId(null);
              }}>
                删除
              </button>
            </>
          }
        >
          <p>{`确定要删除会话"${pendingDeleteSession.summary}"吗？此操作不可撤销。`}</p>
        </ModalFrame>
      )}
    </aside>
  );
}

function SearchHome({
  query,
  mode,
  setQuery,
  setMode,
  sessions,
  activeSessionId,
  sidebarCollapsed,
  onSubmit,
  onNewQuestion,
  onOpenSession,
  onDeleteSession,
  onToggleSidebar,
}: {
  query: string;
  mode: Mode;
  setQuery: (value: string) => void;
  setMode: (value: Mode) => void;
  sessions: ChatSession[];
  activeSessionId: string | null;
  sidebarCollapsed: boolean;
  onSubmit: (event: FormEvent<HTMLFormElement>) => void;
  onNewQuestion: () => void;
  onOpenSession: (id: string) => void;
  onDeleteSession: (id: string) => void;
  onToggleSidebar: () => void;
}) {
  return (
    <div className="site-shell chat-site">
      <header className="search-topbar">
        <div className="search-brand">
          <ServerCog size={20} />
          <span>筑云智库知识平台</span>
        </div>
      </header>

      <main className="search-layout">
        <AppSidebar
          sessions={sessions}
          activeSessionId={activeSessionId}
          collapsed={sidebarCollapsed}
          onToggle={onToggleSidebar}
          onNewQuestion={onNewQuestion}
          onOpenSession={onOpenSession}
          onDeleteSession={onDeleteSession}
        />

        <section className="chat-main">
          <section className="hero-card" style={{ textAlign: "center", maxWidth: 1075, margin: "0 auto", padding: "60px 38px 34px" }}>
            <h1>筑云智库知识平台</h1>

            <form className="search-form" onSubmit={onSubmit}>
              <div className="search-input-row">
                <Search size={20} />
                <label className="sr-only" htmlFor="query">
                  检索问题
                </label>
                <input
                  id="query"
                  value={query}
                  onChange={(event) => setQuery(event.target.value)}
                  placeholder="输入资料问题，例如：C4 疏散路线及应急物资存放点在哪里？"
                />
                <button type="submit" className="search-submit-btn">
                  <ArrowUp size={22} />
                </button>
              </div>
            </form>

            <section className="prompt-panel prompt-panel-plain">
              <div>
                <h2>猜你想问</h2>
                <p>可直接检索规范条款、图纸页、表格信息、OCR 扫描页和资料来源。</p>
              </div>
              <div className="prompt-list">
                {suggestedQuestions.map((item) => (
                  <button key={item} type="button" onClick={() => setQuery(item)}>
                    {item}
                  </button>
                ))}
              </div>
            </section>
          </section>
        </section>
      </main>
    </div>
  );
}

function SearchChatNext({
  query,
  setQuery,
  mode,
  sessions,
  activeSession,
  activeSessionId,
  searching,
  answering,
  error,
  citationSelection,
  sidebarCollapsed,
  onSubmit,
  onNewQuestion,
  onOpenSession,
  onDeleteSession,
  onSelectCitation,
  onCloseCitation,
  onToggleSidebar,
}: {
  query: string;
  setQuery: (value: string) => void;
  mode: Mode;
  sessions: ChatSession[];
  activeSession: ChatSession | null;
  activeSessionId: string | null;
  searching: boolean;
  answering: boolean;
  error: string;
  citationSelection: CitationSelection | null;
  sidebarCollapsed: boolean;
  onSubmit: (event: FormEvent<HTMLFormElement>) => void;
  onNewQuestion: () => void;
  onOpenSession: (sessionId: string) => void;
  onDeleteSession: (sessionId: string) => void;
  onSelectCitation: (turnId: string, sourceIndex: number) => void;
  onCloseCitation: () => void;
  onToggleSidebar: () => void;
}) {
  const lastTurnRef = useRef<HTMLElement | null>(null);
  const [imagePreview, setImagePreview] = useState<ImagePreviewState | null>(null);
  const activeSourceTurn = useMemo(
    () => activeSession?.turns.find((turn) => turn.id === citationSelection?.turnId) || null,
    [activeSession, citationSelection],
  );
  const activeSource =
    citationSelection && activeSourceTurn ? activeSourceTurn.response.results[citationSelection.sourceIndex] || null : null;

  useEffect(() => {
    lastTurnRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [activeSession?.turns.length, searching, answering]);

  // 弹窗打开时锁定 body 滚动，防止滚动条显隐导致页面晃动
  useEffect(() => {
    const modalOpen = imagePreview !== null || activeSource !== null;
    document.body.style.overflow = modalOpen ? "hidden" : "";
    return () => { document.body.style.overflow = ""; };
  }, [imagePreview, activeSource]);

  function openPreview(items: Array<{ result: SearchResult; index: number }>, currentIndex: number) {
    if (!items.length) return;
    setImagePreview({ items, currentIndex });
  }

  return (
    <div className="site-shell chat-site">
      <header className="search-topbar">
        <div className="search-brand">
          <ServerCog size={20} />
          <span>知识检索</span>
        </div>
      </header>

      <main className="search-layout">
        <AppSidebar
          sessions={sessions}
          activeSessionId={activeSessionId}
          collapsed={sidebarCollapsed}
          onToggle={onToggleSidebar}
          onNewQuestion={onNewQuestion}
          onOpenSession={onOpenSession}
          onDeleteSession={onDeleteSession}
        />

        <section className="chat-main">
          {error && <div className="notice error">{error}</div>}

          <section className="qa-results">
            <div className="conversation-stack">
              {!activeSession?.turns.length && (
                <article className="empty-card chat-empty-card">
                  <h2>开始新对话</h2>
                  <p>输入你的第一个问题后，左侧会自动生成一条会话记录，后续追问会继续保留在这条会话中。</p>
                </article>
              )}

              {activeSession?.turns.map((turn, turnIndex) => {
                const turnIsGenerating = turnIndex === activeSession.turns.length - 1 && (searching || answering) && !turn.response.answer;
                const inlineEvidence = turn.response.answer
                  ? displayableResults(turn.response)
                  : isPureImageQuery(turn.query)
                    ? displayableResults(turn.response)
                    : [];

                return (
                  <article key={turn.id} className="chat-turn" ref={turnIndex === activeSession.turns.length - 1 ? lastTurnRef : null}>
                    <div className="message-row user">
                      <div className="message-bubble">{turn.query}</div>
                    </div>

                    <div className="message-row assistant">
                      <div className="assistant-message">
                        {turnIsGenerating && <GeneratingIndicator />}
                        <div className="message-bubble answer-text">
                          {turn.response.answer ? (
                            <AnswerWithCitations
                              answer={turn.response.answer}
                              sourceCount={turn.response.results.length}
                              activeIndex={citationSelection?.turnId === turn.id ? citationSelection.sourceIndex : null}
                              onSelect={(index) => {
                                const citedResult = turn.response.results[index];
                                if (citedResult?.asset_url) {
                                  const evidence = displayableResults(turn.response);
                                  const matchIndex = evidence.findIndex(e => e.result.asset_url === citedResult.asset_url);
                                  if (matchIndex >= 0) {
                                    onCloseCitation();
                                    openPreview(evidence, matchIndex);
                                    return;
                                  }
                                }
                                onSelectCitation(turn.id, index);
                              }}
                            />
                          ) : inlineEvidence.length > 0 ? (
                            "已为您筛选出以下图片内容："
                          ) : (
                            "已找到相关资料，正在生成带引用的回答..."
                          )}
                        </div>
                        {inlineEvidence.length > 0 && (
                          <InlineEvidenceGallery items={inlineEvidence} onOpen={(index) => openPreview(inlineEvidence, index)} />
                        )}
                      </div>
                    </div>

                    {turn.response.answer && !turn.response.results.length && (
                      <div className="notice compact-notice">当前回答没有找到足够相关的资料，请换个关键词再试试。</div>
                    )}
                  </article>
                );
              })}
            </div>
          </section>

          <form className="chat-composer-shell" onSubmit={onSubmit}>
            <div className="search-input-row compact chat-composer-row">
              <Search size={18} />
              <input value={query} onChange={(event) => setQuery(event.target.value)} placeholder="请输入资料问题，系统会结合当前会话继续回答" />
              <button type="submit" className="search-submit-btn" disabled={!query.trim() || searching || answering}>
                {searching || answering ? <Loader2 className="spin" size={22} /> : <ArrowUp size={22} />}
              </button>
            </div>
          </form>
          {activeSource && <CitationModal result={activeSource} index={citationSelection?.sourceIndex || 0} onClose={onCloseCitation} />}
          {imagePreview && <ImageLightbox preview={imagePreview} onClose={() => setImagePreview(null)} />}
        </section>
      </main>
    </div>
  );
}

function ratioText(done?: number, total?: number): string {
  if (!total) return "0 / 0";
  return `${done || 0} / ${total}`;
}

function detectLikelyDuplicateUploads(files: File[], documents: DocumentOut[]): DuplicatePromptState["items"] {
  return files.flatMap((file) => {
    const existing = documents.find((doc) => doc.filename === file.name && doc.size_bytes === file.size);
    if (!existing) return [];
    return [
      {
        incoming_filename: file.name,
        existing: {
          document_id: existing.id,
          filename: existing.filename,
          sha256: "pending-server-check",
          status: "same_name_size",
          uploaded_at: existing.created_at,
        },
      },
    ];
  });
}

function SystemStatusPanel({
  status,
  busy,
  onRefresh,
  onRebuild,
}: {
  status: SystemStatus | null;
  busy: boolean;
  onRefresh: () => Promise<void>;
  onRebuild: () => Promise<void>;
}) {
  if (!status) {
    return (
    <section>
      <div className="panel-header">
        <div>
          <span className="panel-kicker">系统状态</span>
          <h2>等待状态加载</h2>
        </div>
        <button onClick={onRefresh} disabled={busy}>
          <RefreshCcw size={16} />
          刷新
        </button>
      </div>
    </section>
    );
  }

  return (
    <section>
      <div className="panel-header">
        <div>
          <span className="panel-kicker">系统状态</span>
          <h2>当前运行情况</h2>
        </div>
        <div className="panel-actions">
          <button onClick={onRefresh} disabled={busy}>
            <RefreshCcw size={16} />
            刷新
          </button>
          <button onClick={onRebuild} disabled={busy || status.documents.indexable === 0}>
            <Database size={16} />
            重建索引
          </button>
        </div>
      </div>
      <div className="stats-grid">
        <div>
          <span>问答模型</span>
          <strong>{status.models.chat_model}</strong>
        </div>
        <div>
          <span>文本向量</span>
          <strong>{status.models.text_embedding_model}</strong>
        </div>
        <div>
          <span>图片向量</span>
          <strong>{status.models.image_embedding_model}</strong>
        </div>
        <div>
          <span>向量覆盖</span>
          <strong>{ratioText(status.chunks.embedded, status.chunks.total)}</strong>
        </div>
        <div>
          <span>向量库状态</span>
          <strong>{status.services.qdrant_available ? "Qdrant 已启用" : "Qdrant 未启用"}</strong>
        </div>
        <div>
          <span>任务队列</span>
          <strong>{status.services.use_rq ? "RQ / Redis 已启用" : "仅进程内任务"}</strong>
        </div>
        <div>
          <span>OCR 能力</span>
          <strong>{status.services.ocr_enabled ? `已启用 · ${status.services.ocr_backend}` : "未启用完整 OCR"}</strong>
        </div>
        <div>
          <span>重排服务</span>
          <strong>
            {!status.services.reranker_enabled
              ? "未启用"
              : status.services.reranker_healthy
                ? `已启用 · ${status.services.reranker_model} (${status.services.reranker_device})`
                : status.services.reranker_reachable
                  ? "服务异常"
                  : "服务不可达"}
          </strong>
        </div>
      </div>
      {status.services.reranker_enabled && !status.services.reranker_healthy && (
        <div className="notice warning">
          重排服务已启用但当前不可用。{status.services.reranker_error ? `原因：${status.services.reranker_error}` : "后端会自动回退到原始召回排序。"}
        </div>
      )}
    </section>
  );
}

function isVisualAssetKind(value: unknown): boolean {
  return ["embedded_image", "figure", "figure_region", "whole_figure_fallback", "page"].includes(String(value || ""));
}

function JobsPanelModal({ jobs, onClose }: { jobs: JobOut[]; onClose: () => void }) {
  return (
    <ModalFrame title="处理进度" subtitle="上传、解析、入库进度都在这里查看。" onClose={onClose}>
      <div className="job-list">
        {jobs.slice(0, 16).map((job) => (
          <article key={job.id} className="job-card">
            <div className="job-head">
              <strong>{job.document_id ? `文档任务 · ${job.job_type}` : job.job_type}</strong>
              <span className={`status-chip ${job.status}`}>{statusLabel(job.status)}</span>
            </div>
            <p>{job.message || "等待处理"}</p>
            <div className="progress-track">
              <span className="progress-fill" style={{ width: `${Math.max(4, job.progress)}%` }} />
            </div>
            <small>
              {job.progress}% · {formatTime(job.updated_at)}
            </small>
          </article>
        ))}
        {!jobs.length && <div className="empty">还没有处理任务。</div>}
      </div>
    </ModalFrame>
  );
}

function DuplicateConfirmModal({
  state,
  onCancel,
  onConfirm,
}: {
  state: DuplicatePromptState;
  onCancel: () => void;
  onConfirm: () => void;
}) {
  return (
    <ModalFrame
      title="检测到重复文档"
      subtitle="系统发现你本次上传的文件与库中已有文档内容相同。"
      onClose={onCancel}
      actions={
        <>
          <button type="button" className="ghost-button" onClick={onCancel}>
            取消
          </button>
          <button type="button" onClick={onConfirm}>
            继续上传
          </button>
        </>
      }
    >
      <div className="duplicate-list">
        {state.items.map((item, index) => (
          <article className="duplicate-card" key={`${item.incoming_filename}-${index}`}>
            <strong>{item.incoming_filename}</strong>
            <p>已存在文档：{item.existing.filename}</p>
            <p>
              上传时间：{formatTime(item.existing.uploaded_at)} · 状态：{statusLabel(item.existing.status)}
            </p>
            <p>
              {item.existing.status === "batch_duplicate"
                ? "该文件与本次批量上传中的另一份文件内容相同。"
                : item.existing.status === "same_name_size"
                  ? "该文件与库中已有文档文件名和大小一致，继续上传前仍会由服务端再做一次内容判重。"
                  : "该文件与库中已有文档内容相同。"}
            </p>
          </article>
        ))}
      </div>
    </ModalFrame>
  );
}

function ReviewPageLightbox({
  preview,
  onClose,
}: {
  preview: ReviewPreviewState;
  onClose: () => void;
}) {
  const previewUrl = assetUrl(preview.asset.url);
  const [zoom, setZoom] = useState(1.0);
  const reviewStageRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    setZoom(1.0);
    requestAnimationFrame(() => {
      if (!reviewStageRef.current) return;
      reviewStageRef.current.scrollLeft = (reviewStageRef.current.scrollWidth - reviewStageRef.current.clientWidth) / 2;
    });
  }, [preview.asset.id]);

  if (!previewUrl) return null;

  function adjustZoom(delta: number) {
    setZoom((value) => clampZoom(value + delta, 0.8, 4.8));
  }

  return (
    <div className="lightbox-backdrop" onClick={onClose}>
      <div className="lightbox-card review-preview-card" onClick={(event) => event.stopPropagation()}>
        <div className="lightbox-head">
          <div>
            <span className="panel-kicker">页预览</span>
            <h3>{preview.documentName}</h3>
            <p>第 {preview.asset.page_number} 页 · 点击按钮缩放</p>
          </div>
          <div className="lightbox-head-actions">
            <span className="zoom-percent">{Math.round(zoom * 100)}%</span>
            <button type="button" className="icon-button" onClick={() => adjustZoom(-0.2)} disabled={zoom <= 0.8}>
              <ZoomOut size={16} />
            </button>
            <button type="button" className="icon-button" onClick={() => adjustZoom(0.2)} disabled={zoom >= 4.8}>
              <ZoomIn size={16} />
            </button>
            <button type="button" className="icon-button" onClick={onClose}>
              <X size={18} />
            </button>
          </div>
        </div>
        <div className="lightbox-body">
          <div className="lightbox-image-stage" ref={reviewStageRef}>
            <div className="lightbox-image-zoom" style={{ width: `${Math.max(100, zoom * 100)}%` }}>
              <img src={previewUrl} alt={`第 ${preview.asset.page_number} 页预览`} />
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

function ReviewDrawer({
  review,
  busy,
  activeIndexJob,
  reload,
  onApprove,
  onClose,
}: {
  review: DocumentReview;
  busy: boolean;
  activeIndexJob?: JobOut | null;
  reload: (documentId: string) => Promise<void>;
  onApprove: (documentId: string) => Promise<void>;
  onClose: () => void;
}) {
  const [page, setPage] = useState<number | null>(null);
  const [kind, setKind] = useState<KindFilter>("all");
  const [editing, setEditing] = useState<string | null>(null);
  const [busyChunkId, setBusyChunkId] = useState<string | null>(null);
  const [draft, setDraft] = useState("");
  const [pagePreview, setPagePreview] = useState<ReviewPreviewState | null>(null);

  const chunks = useMemo(() => {
    return review.chunks
      .filter((chunk) => (page ? chunk.page_number === page : true))
      .filter((chunk) => (kind === "all" ? true : chunk.kind === kind));
  }, [review, page, kind]);
  const indexBusy = jobInProgress(activeIndexJob);

  async function removeChunk(chunk: ChunkOut) {
    if (busyChunkId === chunk.id) return;
    setBusyChunkId(chunk.id);
    try {
      await deleteChunk(chunk.id);
      if (editing === chunk.id) setEditing(null);
      await reload(chunk.document_id);
    } finally {
      setBusyChunkId(null);
    }
  }

  async function saveChunk(chunk: ChunkOut) {
    if (busyChunkId === chunk.id) return;
    setBusyChunkId(chunk.id);
    try {
      await updateChunk(chunk.id, { content: draft });
      setEditing(null);
      await reload(chunk.document_id);
    } finally {
      setBusyChunkId(null);
    }
  }

  return (
    <div className="drawer-backdrop" onClick={onClose}>
      <aside className="review-drawer" onClick={(event) => event.stopPropagation()}>
        <div className="drawer-head">
          <div>
            <span className="panel-kicker">文档审核</span>
            <h2>{review.document.filename}</h2>
            <p>
              {statusLabel(review.document.status)} · {formatSize(review.document.size_bytes)}
            </p>
          </div>
          <div className="drawer-actions">
            <button disabled={busy || indexBusy || review.document.status === "approved"} onClick={() => onApprove(review.document.id)}>
              <CheckCircle2 size={16} />
              确认入库
            </button>
            <button type="button" className="icon-button" onClick={onClose}>
              <X size={18} />
            </button>
          </div>
        </div>
        {indexBusy && <div className="notice">该文档已有入库任务正在处理中，请等待当前任务完成。</div>}

        <div className="preview-strip">
          {review.page_assets.map((asset) => (
            <button
              key={asset.id}
              className={page === asset.page_number ? "active" : ""}
              onClick={() => {
                setPage(asset.page_number);
                setPagePreview({ asset, documentName: review.document.filename });
              }}
            >
              <img loading="lazy" src={`${API_BASE}${asset.url}`} alt={`第 ${asset.page_number} 页`} />
              <small>点击预览</small>
              <span>第 {asset.page_number} 页</span>
            </button>
          ))}
        </div>

        <div className="chunk-toolbar">
          <strong>内容块</strong>
          <select value={kind} onChange={(event) => setKind(event.target.value as KindFilter)}>
            <option value="all">全部</option>
            <option value="text">文本</option>
            <option value="image">图纸/页面</option>
          </select>
          {page && <button onClick={() => setPage(null)}>清除页码</button>}
          <a className="ghost-link" href={documentFileUrl(review.document.id, page || undefined)} target="_blank" rel="noreferrer">
            打开原始 PDF
          </a>
        </div>

        <div className="chunk-list">
          {chunks
            // 对同一页面的image类型chunk去重，只保留第一个
            .reduce((acc: ChunkOut[], chunk) => {
              if (chunk.kind === "image") {
                const existing = acc.find((c) => c.page_number === chunk.page_number && c.kind === "image");
                if (!existing) {
                  acc.push(chunk);
                }
              } else {
                acc.push(chunk);
              }
              return acc;
            }, [])
            .map((chunk) => (
            <article key={chunk.id}>
              <div className="chunk-meta">
                <span>
                  第 {chunk.page_number} 页 · {chunk.kind === "image" ? "图纸/页面" : "文本"}
                </span>
                <span>{chunk.embedding_model || "无向量"}</span>
              </div>
              {editing === chunk.id ? (
                <>
                  <textarea value={draft} onChange={(event) => setDraft(event.target.value)} disabled={busy || busyChunkId === chunk.id} />
                  <div className="chunk-actions">
                    <button onClick={() => saveChunk(chunk)} disabled={busy || busyChunkId === chunk.id}>
                      保存
                    </button>
                    <button onClick={() => setEditing(null)} disabled={busy || busyChunkId === chunk.id}>
                      取消
                    </button>
                  </div>
                </>
              ) : (
                <>
                  <p>{chunk.content.slice(0, 420)}</p>
                  <div className="chunk-actions">
                    <button
                      onClick={() => {
                        setEditing(chunk.id);
                        setDraft(chunk.content);
                      }}
                      disabled={busy || busyChunkId === chunk.id}
                    >
                      编辑
                    </button>
                    <button onClick={() => removeChunk(chunk)} disabled={busy || busyChunkId === chunk.id}>
                      删除低质量块
                    </button>
                  </div>
                </>
              )}
            </article>
          ))}
          {!chunks.length && <div className="empty">当前筛选条件下没有内容块。</div>}
        </div>
        {pagePreview && <ReviewPageLightbox preview={pagePreview} onClose={() => setPagePreview(null)} />}
      </aside>
    </div>
  );
}

function UploadLogsPage({ logs }: { logs: UploadLogOut[] }) {
  return (
    <section>
      <div className="panel-header">
        <div>
          <span className="panel-kicker">上传日志</span>
          <h2>全部上传记录</h2>
        </div>
      </div>
      <div className="log-table">
        <div className="log-row log-head">
          <span>时间</span>
          <span>文件</span>
          <span>来源</span>
          <span>状态</span>
          <span>上传人</span>
        </div>
        {logs.map((item) => (
          <div className="log-row" key={item.id}>
            <span>{formatTime(item.created_at)}</span>
            <span className="log-name">{item.filename}</span>
            <span>{item.source === "local" ? "本地导入" : "网页上传"}</span>
            <span>{statusLabel(item.document_status || item.status)}</span>
            <span>{item.uploaded_by || "-"}</span>
          </div>
        ))}
        {!logs.length && <div className="empty">还没有上传日志。</div>}
      </div>
    </section>
  );
}

function DocumentsPage({
  documents,
  busy,
  busyDocumentId,
  review,
  onReview,
  onReparse,
  onDelete,
}: {
  documents: DocumentOut[];
  busy: boolean;
  busyDocumentId: string | null;
  review: DocumentReview | null;
  onReview: (documentId: string) => Promise<void>;
  onReparse: (documentId: string) => Promise<void>;
  onDelete: (documentId: string) => Promise<void>;
}) {
  return (
    <section>
      <div className="panel-header">
        <div>
          <span className="panel-kicker">文档列表</span>
          <h2>已上传文档</h2>
        </div>
      </div>
      <div className="document-table">
        <div className="table-head">
          <span>文档</span>
          <span>操作</span>
        </div>
        {documents.map((doc) => (
          <div className={review?.document.id === doc.id ? "table-row selected" : "table-row"} key={doc.id}>
            <span className="doc-title-group">
              <span className="doc-name">{doc.filename}</span>
              <span className="doc-meta-line">
                <span className={`status-chip ${doc.status}`}>{statusLabel(doc.status)}</span>
                <span>{doc.page_count || "-"} 页</span>
                <span>{formatSize(doc.size_bytes)}</span>
                <span>{formatTime(doc.created_at)}</span>
              </span>
              {doc.error_message && <span className="doc-error">{doc.error_message}</span>}
            </span>
            <span className="row-actions">
              <button disabled={busy} onClick={() => void onReview(doc.id)}>
                审核
              </button>
              <button disabled={busyDocumentId === doc.id || busy} onClick={() => void onReparse(doc.id)}>
                <RefreshCcw size={15} />
                重新解析
              </button>
              <button disabled={busyDocumentId === doc.id || busy} className="danger" onClick={() => void onDelete(doc.id)}>
                <Trash2 size={15} />
                删除
              </button>
            </span>
          </div>
        ))}
        {!documents.length && <div className="empty">还没有上传文档。</div>}
      </div>
    </section>
  );
}

function AdminSite({ section }: { section: AdminSection }) {
  const [token, setToken] = useState(localStorage.getItem("graphsearch_token"));
  const [username, setUsername] = useState("admin");
  const [password, setPassword] = useState("admin123");
  const [documents, setDocuments] = useState<DocumentOut[]>([]);
  const [jobs, setJobs] = useState<JobOut[]>([]);
  const [logs, setLogs] = useState<UploadLogOut[]>([]);
  const [review, setReview] = useState<DocumentReview | null>(null);
  const [systemStatus, setSystemStatus] = useState<SystemStatus | null>(null);
  const [localPath, setLocalPath] = useState("");
  const [message, setMessage] = useState("");
  const [busy, setBusy] = useState(false);
  const [busyDocumentId, setBusyDocumentId] = useState<string | null>(null);
  const [uploadProgress, setUploadProgress] = useState<number | null>(null);
  const [progressModalOpen, setProgressModalOpen] = useState(false);
  const [duplicatePrompt, setDuplicatePrompt] = useState<DuplicatePromptState | null>(null);

  const refreshAll = useCallback(async () => {
    const hasToken = Boolean(localStorage.getItem("graphsearch_token"));
    if (!hasToken) return;
    const [nextDocuments, nextJobs, nextLogs, nextStatus] = await Promise.all([
      listDocuments(),
      listJobs(),
      listUploadLogs(),
      getSystemStatus(),
    ]);
    setDocuments(nextDocuments);
    setJobs(nextJobs);
    setLogs(nextLogs);
    setSystemStatus(nextStatus);
  }, []);

  useEffect(() => {
    if (!token) return;
    let cancelled = false;
    let timer: number | undefined;

    const poll = async () => {
      try {
        await refreshAll();
      } catch {
        // ignore
      } finally {
        if (!cancelled) {
          timer = window.setTimeout(poll, 3000);
        }
      }
    };

    void poll();
    return () => {
      cancelled = true;
      if (timer) window.clearTimeout(timer);
    };
  }, [token, refreshAll]);

  useEffect(() => {
    function handleExpired(event: Event) {
      const detail = event instanceof CustomEvent ? String(event.detail || "") : "";
      setToken(null);
      setSystemStatus(null);
      setReview(null);
      setMessage(detail || "登录已过期，请重新登录。");
    }
    window.addEventListener("graphsearch:auth-expired", handleExpired);
    return () => window.removeEventListener("graphsearch:auth-expired", handleExpired);
  }, []);

  async function doLogin(event: FormEvent) {
    event.preventDefault();
    setBusy(true);
    setMessage("");
    try {
      const nextToken = await login(username, password);
      setToken(nextToken);
      await refreshAll();
    } catch (err) {
      setMessage(readableError(err, "登录失败"));
    } finally {
      setBusy(false);
    }
  }

  async function performUpload(files: File[], confirmDuplicates: boolean) {
    const result = await uploadDocuments(files, confirmDuplicates, setUploadProgress);
    setMessage(`已创建 ${result.items.length} 个上传任务。`);
    setProgressModalOpen(true);
    await refreshAll();
  }

  async function doUpload(files: File[]) {
    if (!files.length) return;
    const likelyDuplicates = detectLikelyDuplicateUploads(files, documents);
    if (likelyDuplicates.length) {
      setDuplicatePrompt({ files, items: likelyDuplicates });
      return;
    }
    setBusy(true);
    setUploadProgress(0);
    setMessage("");
    try {
      await performUpload(files, false);
    } catch (err) {
      if (err instanceof ApiError && err.status === 409) {
        const detail = (err.detail || {}) as { duplicates?: DuplicatePromptState["items"] };
        setDuplicatePrompt({ files, items: detail.duplicates || [] });
      } else {
        setMessage(readableError(err, "上传失败"));
      }
    } finally {
      setUploadProgress(null);
      setBusy(false);
    }
  }

  async function doImport(confirmDuplicates = false) {
    if (!localPath.trim()) return;
    setBusy(true);
    setMessage("");
    try {
      const result = await importLocal(localPath.trim(), confirmDuplicates);
      setMessage(`已创建 ${result.items.length} 个导入任务。`);
      setProgressModalOpen(true);
      await refreshAll();
    } catch (err) {
      if (err instanceof ApiError && err.status === 409) {
        const detail = (err.detail || {}) as { duplicates?: DuplicatePromptState["items"] };
        setDuplicatePrompt({ files: [], localPath, items: detail.duplicates || [] });
      } else {
        setMessage(readableError(err, "导入失败"));
      }
    } finally {
      setBusy(false);
    }
  }

  async function openReview(documentId: string) {
    setBusy(true);
    try {
      setReview(await getReview(documentId));
    } catch (err) {
      setMessage(readableError(err, "读取审核信息失败"));
    } finally {
      setBusy(false);
    }
  }

  async function approve(documentId: string) {
    setBusy(true);
    try {
      await approveDocument(documentId);
      setMessage("已提交入库任务，后台正在生成向量索引。");
      setProgressModalOpen(true);
      await refreshAll();
    } catch (err) {
      setMessage(readableError(err, "入库失败"));
    } finally {
      setBusy(false);
    }
  }

  async function remove(documentId: string) {
    const document = documents.find((item) => item.id === documentId);
    if (!window.confirm(`确认删除“${document?.filename || "该文档"}”？删除后会同时移除解析结果和向量索引。`)) return;
    setBusy(true);
    setBusyDocumentId(documentId);
    try {
      await deleteDocument(documentId);
      if (review?.document.id === documentId) setReview(null);
      setMessage("文档已删除。");
      await refreshAll();
    } catch (err) {
      setMessage(readableError(err, "删除失败"));
    } finally {
      setBusy(false);
      setBusyDocumentId(null);
    }
  }

  async function doReparse(documentId: string) {
    setBusy(true);
    setBusyDocumentId(documentId);
    try {
      await reparseDocument(documentId);
      setMessage("已提交重新解析任务。");
      setProgressModalOpen(true);
      await refreshAll();
    } catch (err) {
      setMessage(readableError(err, "重新解析失败"));
    } finally {
      setBusy(false);
      setBusyDocumentId(null);
    }
  }

  async function doRebuildIndex() {
    setBusy(true);
    try {
      const result = await rebuildIndex();
      setMessage(`已创建 ${result.jobs.length} 个重建索引任务。`);
      setProgressModalOpen(true);
      await refreshAll();
    } catch (err) {
      setMessage(readableError(err, "重建索引失败"));
    } finally {
      setBusy(false);
    }
  }

  async function confirmDuplicateFlow() {
    if (!duplicatePrompt) return;
    const pending = duplicatePrompt;
    setDuplicatePrompt(null);
    setBusy(true);
    setUploadProgress(0);
    try {
      if (pending.files.length) {
        await performUpload(pending.files, true);
      } else if (pending.localPath) {
        await doImport(true);
      }
    } catch (err) {
      setMessage(readableError(err, "重复文档确认后上传失败"));
    } finally {
      setUploadProgress(null);
      setBusy(false);
    }
  }

  if (!token) {
    return (
      <main className="admin-login-shell">
        <form className="login-box" onSubmit={doLogin}>
          <ServerCog size={28} />
          <h1>后台管理</h1>
          <p>仅管理员可登录，负责上传、解析、审核和文档维护。</p>
          <label>
            用户名
            <input value={username} onChange={(event) => setUsername(event.target.value)} />
          </label>
          <label>
            密码
            <input value={password} onChange={(event) => setPassword(event.target.value)} type="password" />
          </label>
          <button disabled={busy}>{busy ? <Loader2 className="spin" size={18} /> : <LogIn size={18} />}登录</button>
          {message && <p className="form-message">{message}</p>}
        </form>
      </main>
    );
  }

  return (
    <div className="site-shell admin-site">
      <header className="admin-topbar">
        <div className="search-brand">
          <ServerCog size={20} />
          <span>筑云智库知识平台 · 后台管理</span>
        </div>
      </header>

      <main className="admin-layout">
        <aside className="admin-sidebar admin-sidebar-left">
          <button type="button" className={section === "documents" ? "admin-nav-item active" : "admin-nav-item"} onClick={() => navigateTo("/admin")}>
            <BookOpen size={18} />
            <span>文档管理</span>
          </button>
          <button type="button" className={section === "logs" ? "admin-nav-item active" : "admin-nav-item"} onClick={() => navigateTo("/admin/logs")}>
            <History size={18} />
            <span>上传日志</span>
          </button>
          <button type="button" className={section === "status" ? "admin-nav-item active" : "admin-nav-item"} onClick={() => navigateTo("/admin/status")}>
            <ServerCog size={18} />
            <span>系统状态</span>
          </button>
        </aside>

        <section className="admin-shell">
          {section === "documents" && (
            <section className="admin-card admin-toolbar">
              <div>
                <span className="panel-kicker">文档处理入口</span>
                <h1>批量上传与解析</h1>
                <p>支持一次选择多个 PDF，上传后自动排队解析。任务进度会以弹窗形式显示。</p>
              </div>
              <div className="admin-tools">
                <label className="upload-button">
                  <Upload size={18} />
                  批量上传 PDF
                  <input
                    type="file"
                    accept="application/pdf"
                    multiple
                    disabled={busy}
                    onChange={(event) => {
                      const files = Array.from(event.target.files || []);
                      void doUpload(files);
                      event.currentTarget.value = "";
                    }}
                  />
                </label>

                <button type="button" className="ghost-button" onClick={() => setProgressModalOpen(true)}>
                  查看进度
                </button>
              </div>
            </section>
          )}

          {message && <div className="notice">{message}</div>}
          {section === "status" &&
            systemStatus &&
            (!systemStatus.services.use_rq ||
              !systemStatus.services.qdrant_available ||
              !systemStatus.services.ocr_enabled ||
              (systemStatus.services.reranker_enabled && !systemStatus.services.reranker_healthy)) && (
            <div className="notice warning">
              当前仍处于轻量模式：{!systemStatus.services.use_rq ? "任务队列未启用；" : ""}
              {!systemStatus.services.qdrant_available ? "向量库未启用；" : ""}
              {!systemStatus.services.ocr_enabled ? "完整 OCR 未启用。" : ""}
              {systemStatus.services.reranker_enabled && !systemStatus.services.reranker_healthy ? "重排服务当前不可用。" : ""}
              这会影响批量吞吐、检索召回和高亮精度。
            </div>
          )}

          {section === "documents" && (
            <DocumentsPage
              documents={documents}
              busy={busy}
              busyDocumentId={busyDocumentId}
              review={review}
              onReview={openReview}
              onReparse={doReparse}
              onDelete={remove}
            />
          )}
          {section === "logs" && <UploadLogsPage logs={logs} />}
          {section === "status" && <SystemStatusPanel status={systemStatus} busy={busy} onRefresh={refreshAll} onRebuild={doRebuildIndex} />}
        </section>
      </main>

      {progressModalOpen && <JobsPanelModal jobs={jobs} onClose={() => setProgressModalOpen(false)} />}
      {duplicatePrompt && <DuplicateConfirmModal state={duplicatePrompt} onCancel={() => setDuplicatePrompt(null)} onConfirm={() => void confirmDuplicateFlow()} />}
      {review && <ReviewDrawer review={review} busy={busy} reload={openReview} onApprove={approve} onClose={() => setReview(null)} />}
      {uploadProgress !== null && (
        <div className="floating-upload-indicator">
          <Loader2 className="spin" size={16} />
          <span>上传中 {uploadProgress}%</span>
        </div>
      )}
    </div>
  );
}

export default function App() {
  const storedChatState = useMemo(() => parseStoredChatSessions(), []);
  const [path, setPath] = useState(() => window.location.pathname);
  const [query, setQuery] = useState("");
  const [mode, setMode] = useState<Mode>("all");
  const [sessions, setSessions] = useState<ChatSession[]>(storedChatState.sessions);
  const [activeSessionId, setActiveSessionId] = useState<string | null>(storedChatState.activeSessionId);
  const [citationSelection, setCitationSelection] = useState<CitationSelection | null>(null);
  const [searching, setSearching] = useState(false);
  const [answering, setAnswering] = useState(false);
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  const [error, setError] = useState("");
  const sessionsRef = useRef<ChatSession[]>(storedChatState.sessions);
  const activeSessionIdRef = useRef<string | null>(storedChatState.activeSessionId);
  const activeSession = useMemo(
    () => sessions.find((session) => session.id === activeSessionId) || sessions[0] || null,
    [activeSessionId, sessions],
  );

  useEffect(() => {
    const handlePop = () => {
      setPath(window.location.pathname);
      if (window.location.pathname.startsWith("/chat")) {
        const params = new URLSearchParams(window.location.search);
        const sid = params.get("s");
        if (sid && sessionsRef.current.some((s) => s.id === sid)) {
          setActiveSessionId(sid);
          setCitationSelection(null);
          setError("");
        }
      }
    };
    window.addEventListener("popstate", handlePop);
    return () => window.removeEventListener("popstate", handlePop);
  }, []);

  // On mount or path change, select session from URL if on /chat
  useEffect(() => {
    if (path.startsWith("/chat")) {
      const params = new URLSearchParams(window.location.search);
      const sid = params.get("s");
      if (sid && sessions.some((s) => s.id === sid)) {
        setActiveSessionId(sid);
        setCitationSelection(null);
        setError("");
      }
    }
  }, [path]);

  // Sync activeSessionId to URL on /chat page
  useEffect(() => {
    if (path.startsWith("/chat")) {
      const search = activeSessionId ? `?s=${activeSessionId}` : "";
      if (window.location.search !== search) {
        window.history.replaceState({}, "", `/chat${search}`);
      }
    }
  }, [path, activeSessionId]);

  useEffect(() => {
    sessionsRef.current = sessions;
  }, [sessions]);

  useEffect(() => {
    activeSessionIdRef.current = activeSessionId;
  }, [activeSessionId]);

  useEffect(() => {
    if (!sessions.length) {
      setActiveSessionId(null);
      return;
    }
    if (!activeSessionId || !sessions.some((session) => session.id === activeSessionId)) {
      setActiveSessionId(sessions[0].id);
    }
  }, [activeSessionId, sessions]);

  useEffect(() => {
    if (!citationSelection || !activeSession) return;
    if (!activeSession.turns.some((turn) => turn.id === citationSelection.turnId)) {
      setCitationSelection(null);
    }
  }, [activeSession, citationSelection]);

  useEffect(() => {
    serializeChatSessions(sessions, activeSessionId);
  }, [activeSessionId, sessions]);

  function upsertTurnInSession(sessionId: string, nextTurn: ChatTurn) {
    setSessions((items) =>
      items.map((session) => {
        if (session.id !== sessionId) return session;
        const existingIndex = session.turns.findIndex((entry) => entry.id === nextTurn.id);
        if (existingIndex === -1) {
          return { ...session, turns: [...session.turns, nextTurn] };
        }
        const updated = [...session.turns];
        updated[existingIndex] = nextTurn;
        return { ...session, turns: updated };
      }),
    );
  }

  function removeTurnFromSession(sessionId: string, turnId: string) {
    setSessions((items) =>
      items.map((session) => {
        if (session.id !== sessionId) return session;
        const nextTurns = session.turns.filter((turn) => turn.id !== turnId);
        return { ...session, turns: nextTurns, summary: nextTurns.length ? session.summary : EMPTY_SESSION_SUMMARY };
      }),
    );
  }

  function createEmptySession(options?: { activate?: boolean }) {
    const session = createChatSession();
    setSessions((items) => [session, ...items]);
    if (options?.activate !== false) {
      setActiveSessionId(session.id);
    }
    return session.id;
  }

  function resolveTargetSessionId(forceNewSession: boolean): string {
    if (!forceNewSession) {
      const currentSession = sessionsRef.current.find((session) => session.id === activeSessionIdRef.current) || null;
      if (currentSession) return currentSession.id;
    }
    return createEmptySession();
  }

  async function runSearch(trimmed: string, selectedMode: Mode, options?: { forceNewSession?: boolean }) {
    const targetSessionId = resolveTargetSessionId(Boolean(options?.forceNewSession));
    const id = `${Date.now()}`;
    const pendingTurn: ChatTurn = {
      id,
      query: trimmed,
      response: {
        query: trimmed,
        answer: "",
        results: [],
        diagnostics: {},
      },
      createdAt: new Date().toISOString(),
    };

    setError("");
    setSearching(true);
    setAnswering(false);
    setQuery("");
    setActiveSessionId(targetSessionId);
    setCitationSelection(null);
    setSessions((items) =>
      items.map((session) => {
        if (session.id !== targetSessionId) return session;
        const summary = session.turns.length ? session.summary : buildSessionSummary(trimmed);
        return {
          ...session,
          summary,
          turns: [...session.turns, pendingTurn],
        };
      }),
    );

    try {
      const isPureImage = isPureImageQuery(trimmed);
      const searchOptions: SearchFilters = { top_k: isPureImage ? 30 : 20 };
      if (isPureImage) searchOptions.kind = "image";

      const searchResult: SearchResponse = await searchDocuments(trimmed, selectedMode, searchOptions);
      if (isImageSeekingQuery(trimmed)) {
        const imageResult = await searchDocuments(trimmed, selectedMode, { kind: "image", top_k: 25 });
        const seenIds = new Set(searchResult.results.map((r) => r.chunk_id));
        for (const r of imageResult.results) {
          if (!seenIds.has(r.chunk_id)) searchResult.results.push(r);
        }
        searchResult.results.sort((a, b) => b.score - a.score);
      }

      if (isPureImage) {
        // 纯图模式：过滤出 embedded_image 类型，跳过 LLM 回答
        const pureImages = searchResult.results.filter((r) => {
          if (r.kind !== "image") return false;
          const assetKind = (r.metadata as Record<string, unknown>).asset_kind;
          return isVisualAssetKind(assetKind);
        });
        if (!pureImages.length) {
          setError("未找到纯视觉类型的图片资源。");
          removeTurnFromSession(targetSessionId, id);
          return;
        }
        const imageResponse: AnswerResponse = {
          query: trimmed,
          answer: "",
          results: pureImages,
          diagnostics: searchResult.diagnostics,
        };
        upsertTurnInSession(targetSessionId, { ...pendingTurn, response: imageResponse });
        return;
      }
      const initialResponse: AnswerResponse = {
        query: trimmed,
        answer: "",
        results: searchResult.results,
        diagnostics: searchResult.diagnostics,
      };
      upsertTurnInSession(targetSessionId, { ...pendingTurn, response: initialResponse });
      setSearching(false);
      setAnswering(true);

      const finalResponse = await generateAnswer(trimmed, searchResult.results);
      upsertTurnInSession(targetSessionId, { ...pendingTurn, response: finalResponse });
    } catch (err) {
      setError(readableError(err, "????"));
      removeTurnFromSession(targetSessionId, id);
    } finally {
      setSearching(false);
      setAnswering(false);
    }
  }

  async function submitHomeSearch(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const trimmed = query.trim();
    if (!trimmed) return;
    navigateTo("/chat");
    setPath("/chat");
    void runSearch(trimmed, mode, { forceNewSession: true });
  }

  function handleNewQuestion() {
    setQuery("");
    setCitationSelection(null);
    setError("");
    setActiveSessionId(null);
    navigateTo("/");
    setPath("/");
  }

  function openSession(sessionId: string) {
    setActiveSessionId(sessionId);
    setCitationSelection(null);
    setError("");
    navigateTo(`/chat?s=${sessionId}`);
  }

  function selectCitation(turnId: string, sourceIndex: number) {
    setCitationSelection({ turnId, sourceIndex });
  }

  function closeCitation() {
    setCitationSelection(null);
  }

  function deleteSession(sessionId: string) {
    const targetSession = sessions.find((session) => session.id === sessionId) || null;
    if (!targetSession) return;

    const remainingSessions = sessions.filter((session) => session.id !== sessionId);
    setSessions(remainingSessions);
    setCitationSelection(null);
    setError("");

    if (activeSessionId === sessionId) {
      const nextActiveSession = remainingSessions[0] || null;
      setActiveSessionId(nextActiveSession?.id || null);
      if (!nextActiveSession) {
        setQuery("");
        navigateTo("/");
        setPath("/");
      }
    }
  }

  if (path.startsWith("/admin/status")) {
    return <AdminSite section="status" />;
  }
  if (path.startsWith("/admin/logs")) {
    return <AdminSite section="logs" />;
  }
  if (path.startsWith("/admin")) {
    return <AdminSite section="documents" />;
  }
  if (path.startsWith("/chat")) {
    return (
      <>
        <SearchChatNext
          query={query}
          setQuery={setQuery}
          mode={mode}
          sessions={sessions}
          activeSession={activeSession}
          activeSessionId={activeSessionId}
          searching={searching}
          answering={answering}
          error={error}
          citationSelection={citationSelection}
          sidebarCollapsed={sidebarCollapsed}
          onSubmit={(event) => {
            event.preventDefault();
            const trimmed = query.trim();
            if (!trimmed || searching || answering) return;
            void runSearch(trimmed, mode);
          }}
          onNewQuestion={handleNewQuestion}
          onOpenSession={openSession}
          onDeleteSession={deleteSession}
          onSelectCitation={selectCitation}
          onCloseCitation={closeCitation}
          onToggleSidebar={() => setSidebarCollapsed((v) => !v)}
        />
      </>
    );
  }

  return (
    <SearchHome
      query={query}
      mode={mode}
      setQuery={setQuery}
      setMode={setMode}
      sessions={sessions}
      activeSessionId={activeSessionId}
      sidebarCollapsed={sidebarCollapsed}
      onSubmit={submitHomeSearch}
          onNewQuestion={handleNewQuestion}
          onOpenSession={openSession}
          onDeleteSession={deleteSession}
      onToggleSidebar={() => setSidebarCollapsed((v) => !v)}
    />
  );
}
