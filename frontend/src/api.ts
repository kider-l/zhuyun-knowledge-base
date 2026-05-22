import type {
  AnswerHistoryTurn,
  AnswerResponse,
  ChunkOut,
  DocumentOut,
  DocumentReview,
  JobOut,
  SearchFilters,
  SearchResponse,
  SystemStatus,
  UploadBatchResponse,
  UploadLogOut,
} from "./types";

export const API_BASE = import.meta.env.VITE_API_BASE_URL || "http://localhost:8000";

export class ApiError extends Error {
  status: number;
  detail: unknown;

  constructor(message: string, status: number, detail: unknown) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.detail = detail;
  }
}

function authHeaders(): HeadersInit {
  const token = localStorage.getItem("graphsearch_token");
  return token ? { Authorization: `Bearer ${token}` } : {};
}

async function parseResponse<T>(response: Response): Promise<T> {
  if (!response.ok) {
    let detail: unknown = null;
    let message = response.statusText;
    try {
      const data = await response.json();
      detail = data.detail ?? data;
      if (typeof detail === "string") {
        message = detail;
      } else if (detail && typeof detail === "object" && "message" in (detail as Record<string, unknown>)) {
        message = String((detail as Record<string, unknown>).message || message);
      }
    } catch {
      // ignore non-json error bodies
    }
    if (response.status === 401) {
      localStorage.removeItem("graphsearch_token");
      window.dispatchEvent(new CustomEvent("graphsearch:auth-expired", { detail: message }));
    }
    if (response.status >= 500) {
      message = "后台正在处理文档，请稍后重试。";
    }
    throw new ApiError(message, response.status, detail);
  }
  return response.json() as Promise<T>;
}

export function readableError(err: unknown, fallback = "操作失败"): string {
  if (
    err instanceof Error &&
    (err.name === "AbortError" || /failed to fetch|network request failed|load failed|fetch/i.test(err.message))
  ) {
    return "网络连接异常，暂时无法连接服务器，请稍后重试。";
  }
  return err instanceof Error ? err.message : fallback;
}

export async function login(username: string, password: string): Promise<string> {
  const response = await fetch(`${API_BASE}/api/auth/login`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ username, password }),
  });
  const data = await parseResponse<{ access_token: string }>(response);
  localStorage.setItem("graphsearch_token", data.access_token);
  return data.access_token;
}

export async function searchDocuments(query: string, mode: string, filters: SearchFilters = {}): Promise<SearchResponse> {
  const response = await fetch(`${API_BASE}/api/search`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query, mode, top_k: filters.top_k ?? 6, ...filters }),
  });
  return parseResponse<SearchResponse>(response);
}

export async function generateAnswer(
  query: string,
  results: SearchResponse["results"],
  history: AnswerHistoryTurn[] = [],
): Promise<AnswerResponse> {
  const started = performance.now();
  const response = await fetch(`${API_BASE}/api/answer`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query, results, history }),
  });
  const data = await parseResponse<AnswerResponse>(response);
  data.diagnostics = {
    ...data.diagnostics,
    total_ms: Number(((data.diagnostics?.search_ms as number | undefined) ?? 0)),
  };
  data.diagnostics.answer_ms = Number(((data.diagnostics.answer_ms as number | undefined) ?? (performance.now() - started)).toFixed(2));
  const searchMs = Number((data.diagnostics.search_ms as number | undefined) ?? 0);
  data.diagnostics.total_ms = Number((searchMs + Number(data.diagnostics.answer_ms)).toFixed(2));
  return data;
}

export async function listDocuments(): Promise<DocumentOut[]> {
  const response = await fetch(`${API_BASE}/api/documents`, { headers: authHeaders() });
  return parseResponse<DocumentOut[]>(response);
}

export function uploadDocuments(
  files: File[],
  confirmDuplicates = false,
  onProgress?: (percent: number) => void,
): Promise<UploadBatchResponse> {
  const token = localStorage.getItem("graphsearch_token");
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open("POST", `${API_BASE}/api/documents/upload-batch`);
    if (token) {
      xhr.setRequestHeader("Authorization", `Bearer ${token}`);
    }
    xhr.upload.onprogress = (event) => {
      if (!onProgress || !event.lengthComputable) return;
      onProgress(Math.round((event.loaded / event.total) * 100));
    };
    xhr.onerror = () => reject(new TypeError("Network request failed"));
    xhr.onload = () => {
      let parsed: unknown = null;
      try {
        parsed = xhr.responseText ? JSON.parse(xhr.responseText) : null;
      } catch {
        parsed = null;
      }
      if (xhr.status >= 200 && xhr.status < 300) {
        resolve(parsed as UploadBatchResponse);
        return;
      }
      const detail = parsed && typeof parsed === "object" ? (parsed as Record<string, unknown>).detail ?? parsed : parsed;
      const message =
        typeof detail === "string"
          ? detail
          : detail && typeof detail === "object" && "message" in (detail as Record<string, unknown>)
            ? String((detail as Record<string, unknown>).message || xhr.statusText)
            : xhr.statusText;
      reject(new ApiError(message, xhr.status, detail));
    };
    const formData = new FormData();
    files.forEach((file) => formData.append("files", file));
    formData.append("confirm_duplicates", String(confirmDuplicates));
    xhr.send(formData);
  });
}

export async function importLocal(path: string, confirmDuplicates = false): Promise<UploadBatchResponse> {
  const response = await fetch(`${API_BASE}/api/import/local`, {
    method: "POST",
    headers: { "Content-Type": "application/json", ...authHeaders() },
    body: JSON.stringify({ path, confirm_duplicates: confirmDuplicates }),
  });
  return parseResponse<UploadBatchResponse>(response);
}

export async function getReview(documentId: string): Promise<DocumentReview> {
  const response = await fetch(`${API_BASE}/api/documents/${documentId}/review`, { headers: authHeaders() });
  return parseResponse<DocumentReview>(response);
}

export async function listJobs(): Promise<JobOut[]> {
  const response = await fetch(`${API_BASE}/api/jobs`, { headers: authHeaders() });
  return parseResponse<JobOut[]>(response);
}

export async function listUploadLogs(): Promise<UploadLogOut[]> {
  const response = await fetch(`${API_BASE}/api/upload-logs`, { headers: authHeaders() });
  return parseResponse<UploadLogOut[]>(response);
}

export async function approveDocument(documentId: string): Promise<{ job: JobOut }> {
  const response = await fetch(`${API_BASE}/api/documents/${documentId}/approve`, {
    method: "POST",
    headers: authHeaders(),
  });
  return parseResponse<{ job: JobOut }>(response);
}

export async function reparseDocument(documentId: string): Promise<{ job: JobOut }> {
  const response = await fetch(`${API_BASE}/api/documents/${documentId}/reparse`, {
    method: "POST",
    headers: authHeaders(),
  });
  return parseResponse<{ job: JobOut }>(response);
}

export async function deleteDocument(documentId: string): Promise<void> {
  const response = await fetch(`${API_BASE}/api/documents/${documentId}`, {
    method: "DELETE",
    headers: authHeaders(),
  });
  await parseResponse<{ ok: boolean }>(response);
}

export async function updateChunk(chunkId: string, payload: { content?: string; approved?: boolean }): Promise<ChunkOut> {
  const response = await fetch(`${API_BASE}/api/chunks/${chunkId}`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json", ...authHeaders() },
    body: JSON.stringify(payload),
  });
  return parseResponse<ChunkOut>(response);
}

export async function deleteChunk(chunkId: string): Promise<void> {
  const response = await fetch(`${API_BASE}/api/chunks/${chunkId}`, {
    method: "DELETE",
    headers: authHeaders(),
  });
  await parseResponse<{ ok: boolean }>(response);
}

export async function getSystemStatus(): Promise<SystemStatus> {
  const response = await fetch(`${API_BASE}/api/system/status`, { headers: authHeaders() });
  return parseResponse<SystemStatus>(response);
}

export async function rebuildIndex(): Promise<{ jobs: JobOut[] }> {
  const response = await fetch(`${API_BASE}/api/index/rebuild`, {
    method: "POST",
    headers: authHeaders(),
  });
  return parseResponse<{ jobs: JobOut[] }>(response);
}

export function assetUrl(path?: string | null): string | null {
  return path ? `${API_BASE}${path}` : null;
}

export function documentFileUrl(documentId: string, pageNumber?: number): string {
  const pageAnchor = pageNumber ? `#page=${pageNumber}` : "";
  return `${API_BASE}/api/documents/${documentId}/file${pageAnchor}`;
}
