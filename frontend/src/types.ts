export type DocumentStatus = "uploaded" | "parsing" | "parsed" | "approved" | "failed";

export interface DocumentOut {
  id: string;
  filename: string;
  original_path?: string | null;
  size_bytes: number;
  page_count: number;
  status: DocumentStatus | string;
  parse_stats: Record<string, unknown>;
  error_message?: string | null;
  created_at: string;
  updated_at: string;
}

export interface JobOut {
  id: string;
  document_id?: string | null;
  document_filename?: string | null;
  job_type: string;
  status: string;
  progress: number;
  message?: string | null;
  error_message?: string | null;
  created_at: string;
  updated_at: string;
}

export interface AssetOut {
  id: string;
  document_id: string;
  page_number: number;
  kind: string;
  width?: number | null;
  height?: number | null;
  bbox?: Record<string, unknown> | null;
  parent_asset_id?: string | null;
  region_index?: number | null;
  region_type?: string | null;
  region_summary?: string | null;
  caption?: string | null;
  ocr_text?: string | null;
  url?: string | null;
}

export interface ChunkOut {
  id: string;
  document_id: string;
  asset_id?: string | null;
  page_number: number;
  kind: string;
  content: string;
  title_path?: string | null;
  chunk_metadata: Record<string, unknown>;
  embedding_model?: string | null;
  embedding_dim?: number | null;
  approved: boolean;
  indexed: boolean;
}

export interface DocumentReview {
  document: DocumentOut;
  jobs: JobOut[];
  stats: Record<string, unknown>;
  sample_chunks: ChunkOut[];
  chunks: ChunkOut[];
  page_assets: AssetOut[];
  image_assets: AssetOut[];
}

export interface SearchResult {
  chunk_id: string;
  document_id: string;
  document_name: string;
  page_number: number;
  kind: string;
  score: number;
  snippet: string;
  title_path?: string | null;
  asset_id?: string | null;
  asset_url?: string | null;
  match_reason?: string | null;
  highlight_boxes?: Array<{ x: number; y: number; width: number; height: number }>;
  metadata: Record<string, unknown>;
}

export interface AnswerResponse {
  query: string;
  answer: string;
  results: SearchResult[];
  diagnostics: Record<string, unknown>;
}

export interface AnswerHistoryTurn {
  query: string;
  answer: string;
}

export interface SearchResponse {
  query: string;
  mode: string;
  results: SearchResult[];
  diagnostics: Record<string, unknown>;
}

export interface SearchFilters {
  document_id?: string;
  page_from?: number;
  page_to?: number;
  kind?: "all" | "text" | "image";
  top_k?: number;
}

export interface DuplicateDocumentInfo {
  document_id: string;
  filename: string;
  sha256: string;
  status: string;
  uploaded_at: string;
}

export interface UploadBatchItem {
  filename: string;
  status: string;
  message?: string | null;
  document?: DocumentOut | null;
  job?: JobOut | null;
  duplicate?: DuplicateDocumentInfo | null;
}

export interface UploadBatchResponse {
  items: UploadBatchItem[];
}

export interface UploadLogOut {
  id: string;
  filename: string;
  sha256: string;
  source: string;
  uploaded_by?: string | null;
  status: string;
  message?: string | null;
  created_at: string;
  updated_at: string;
  document_id?: string | null;
  duplicate_of_document_id?: string | null;
  document_status?: string | null;
}

export interface SystemStatus {
  documents: {
    total: number;
    approved: number;
    parsed_waiting_approval: number;
    failed: number;
    indexable: number;
  };
  chunks: {
    total: number;
    embedded: number;
    text: number;
    text_embedded: number;
    image: number;
    image_embedded: number;
    image_secondary_embedded?: number;
  };
  models: {
    chat_provider: string;
    chat_model: string;
    text_embedding_model: string;
    text_embedding_configured: boolean;
    image_embedding_provider: string;
    image_embedding_model: string;
    image_embedding_configured: boolean;
  };
  services: {
    qdrant_available: boolean;
    use_rq: boolean;
    running_jobs: number;
    ocr_enabled: boolean;
    ocr_backend: string;
    ocr_paddle_enabled?: boolean;
    ocr_paddle_available?: boolean;
    ocr_cloud_enabled?: boolean;
    ocr_cloud_available?: boolean;
    ocr_last_error?: string | null;
    ocr_stats?: {
      paddle_ocr_pages: number;
      cloud_ocr_pages: number;
      cloud_ocr_attempted_pages?: number;
      ocr_fallback_pages: number;
      ocr_failed_pages: number;
      table_structured_pages: number;
    };
    reranker_enabled: boolean;
    reranker_reachable: boolean;
    reranker_healthy: boolean;
    reranker_model: string;
    reranker_device: string;
    reranker_requested_device: string;
    reranker_use_fp16: boolean;
    reranker_error?: string | null;
    schema_version_ok?: boolean;
    image_schema_columns_ready?: boolean;
    requires_reindex?: boolean;
    missing_schema_columns?: Record<string, string[]>;
  };
}
