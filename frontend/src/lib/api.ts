/* eslint-disable @typescript-eslint/no-explicit-any */

import {
  API_TIMEOUTS,
  ApiError,
  type ApiRequestHandle,
  apiPathSegment,
  apiUrl,
  downloadBlob,
  getErrorMessage,
  requestBlob,
  requestJson,
  requestJsonSse,
  requestText,
} from "./api-client";

export { ApiError, apiPathSegment, apiUrl };
export type { ApiRequestHandle };

export interface PipelineStartPayload {
  file_paths: string[];
  server_url?: string;
  model_name?: string;
  workers?: number;
  max_concurrent?: number;
  max_retries?: number;
  target_dim?: number;
  guided_decoding?: boolean;
}

export interface RagQueryPayload {
  query: string;
  mode?: string;
  model_url?: string;
  model_name?: string;
  top_k?: number;
  case_id?: string;
  doc_type?: string;
  author?: string;
  date_from?: string;
  date_to?: string;
  stream?: boolean;
  use_reranker?: boolean;
  reranker_model?: string;
  reranker_device?: string;
  max_output_tokens?: number;
}

export interface RagStreamStatus {
  type: "status";
  stage: "starting" | "retrieving" | "preparing" | "generating" | "complete";
  message: string;
  progress?: number;
}

interface RagStreamEvent {
  type?: "content" | "status";
  chunk?: string;
  stage?: RagStreamStatus["stage"];
  message?: string;
  progress?: number;
}

type ApiResult = any;

function failedResult(error: unknown): ApiResult {
  return { success: false, message: getErrorMessage(error) };
}

function jsonPost<T = ApiResult>(path: string, payload?: unknown): Promise<T> {
  return requestJson<T>(path, {
    method: "POST",
    ...(payload === undefined ? {} : { json: payload }),
  });
}

export async function fetchSystemHealth() {
  try {
    return await requestJson<ApiResult>("/api/health");
  } catch (error) {
    if (error instanceof ApiError) {
      return { status: "error", message: error.message, services: [], gpu: null };
    }
    return { status: "offline", error: getErrorMessage(error), services: [], gpu: null };
  }
}

export async function fetchCaseSummary() {
  try {
    return await requestJson<ApiResult>("/api/case-summary");
  } catch (error) {
    return { error: getErrorMessage(error) };
  }
}

export async function fetchCaseTimeline(runId: string) {
  try {
    return await requestJson<ApiResult>(`/api/cases/${apiPathSegment(runId)}/timeline`);
  } catch (error) {
    return { run_id: runId, events: [], error: getErrorMessage(error) };
  }
}

export async function fetchPipelineRuns() {
  try {
    return await requestJson<ApiResult[]>("/api/pipeline/runs");
  } catch {
    return [];
  }
}

export async function fetchDockerStatus(role: "ocr" | "analysis" = "ocr") {
  try {
    return await requestJson<ApiResult>(`/api/docker/status?role=${role}`);
  } catch {
    return { status: "unknown", is_ready: false };
  }
}

export async function fetchDockerModels() {
  try {
    return await requestJson<ApiResult>("/api/docker/models");
  } catch {
    return { models: [] };
  }
}

export async function fetchDockerLogs(tail: number = 200, role: "ocr" | "analysis" = "ocr") {
  const query = new URLSearchParams({ tail: String(tail), role });
  try {
    return await requestJson<ApiResult>(`/api/docker/logs?${query}`);
  } catch (error) {
    return { logs: getErrorMessage(error), container_status: "error" };
  }
}

export async function setVllmRoleRunning(role: "ocr" | "analysis", running: boolean) {
  try {
    return await jsonPost(`/api/docker/roles/${role}/${running ? "start" : "stop"}`);
  } catch (error) {
    return failedResult(error);
  }
}

export async function setExtendedAnalysisContext(extended: boolean) {
  try {
    return await jsonPost("/api/docker/analysis/context-mode", { extended });
  } catch (error) {
    return failedResult(error);
  }
}

export async function startDockerContainer(payload: {
  hf_token?: string;
  model_name?: string;
  port?: number;
  gpu_mem?: number;
  max_model_len?: number;
}) {
  try {
    return await jsonPost("/api/docker/start", payload);
  } catch (error) {
    return failedResult(error);
  }
}

export async function stopDockerContainer() {
  try {
    return await jsonPost("/api/docker/stop");
  } catch (error) {
    return failedResult(error);
  }
}

export async function shutdownApp() {
  try {
    return await jsonPost("/api/system/shutdown", { confirmation: "SHUTDOWN" });
  } catch (error) {
    return failedResult(error);
  }
}

export async function uploadPipelineFiles(files: File[]): Promise<string[]> {
  const formData = new FormData();
  files.forEach((file) => formData.append("files", file));

  const data = await requestJson<{ file_paths?: string[] }>("/api/pipeline/upload", {
    method: "POST",
    body: formData,
    timeoutMs: API_TIMEOUTS.upload,
  });
  return data.file_paths || [];
}

export function triggerIngestSSE(
  payload: PipelineStartPayload,
  onMessage: (data: unknown) => void,
  onError: (error: unknown) => void,
  onComplete: () => void,
): ApiRequestHandle {
  return requestJsonSse(
    "/api/pipeline/start",
    { method: "POST", json: payload },
    { onMessage, onError, onComplete },
  );
}

export async function stopPipelineRun(runId: string = "") {
  try {
    return await jsonPost(`/api/pipeline/stop/${apiPathSegment(runId)}`);
  } catch (error) {
    return failedResult(error);
  }
}

export function triggerRagChatSSE(
  payload: RagQueryPayload,
  onChunk: (chunk: string) => void,
  onError: (error: unknown) => void,
  onComplete: () => void,
  onStatus?: (status: RagStreamStatus) => void,
): ApiRequestHandle {
  return requestJsonSse<RagStreamEvent>(
    "/api/rag/query",
    {
      method: "POST",
      json: { ...payload, stream: true },
    },
    {
      onMessage: (data) => {
        if (typeof data.chunk === "string") onChunk(data.chunk);
        if (data.type === "status" && data.stage && typeof data.message === "string") {
          onStatus?.({
            type: "status",
            stage: data.stage,
            message: data.message,
            ...(typeof data.progress === "number" ? { progress: data.progress } : {}),
          });
        }
      },
      onError,
      onComplete,
    },
  );
}

export async function deleteCases(runIds: string[], deleteAll: boolean = false) {
  try {
    return await jsonPost("/api/rag/cases/delete", {
      run_ids: runIds,
      delete_all: deleteAll,
    });
  } catch (error) {
    return failedResult(error);
  }
}

export async function fetchEmbeddingTelemetry() {
  try {
    return await requestJson<ApiResult>("/api/rag/embedding/telemetry");
  } catch (error) {
    return {
      telemetry_html: `<div style='color:red;'>Error loading telemetry: ${getErrorMessage(error)}</div>`,
    };
  }
}

export async function saveEmbeddingConfig(config: {
  embedding_model: string;
  embedding_device: string;
  chunk_size: number;
  chunk_overlap: number;
  embedding_batch_size: number;
}) {
  try {
    return await jsonPost("/api/rag/embedding/config", config);
  } catch (error) {
    return failedResult(error);
  }
}

export async function purgeVectorCache() {
  try {
    return await jsonPost("/api/rag/embedding/purge-cache");
  } catch (error) {
    return failedResult(error);
  }
}

export async function indexRun(runDir: string) {
  try {
    return await jsonPost("/api/rag/index", { run_dir: runDir });
  } catch (error) {
    return failedResult(error);
  }
}

export async function indexAllRuns() {
  try {
    return await jsonPost("/api/rag/index-all");
  } catch (error) {
    return failedResult(error);
  }
}

export function triggerIndexRunSSE(
  runDir: string,
  onMessage: (message: string) => void,
  onError: (error: unknown) => void,
  onComplete: () => void,
): ApiRequestHandle {
  return requestJsonSse<{ message?: string }>(
    "/api/rag/index/stream",
    { method: "POST", json: { run_dir: runDir } },
    {
      onMessage: (data) => {
        if (typeof data.message === "string") onMessage(data.message);
      },
      onError,
      onComplete,
    },
  );
}

export function triggerIndexAllRunsSSE(
  onMessage: (message: string) => void,
  onError: (error: unknown) => void,
  onComplete: () => void,
): ApiRequestHandle {
  return requestJsonSse<{ message?: string }>(
    "/api/rag/index-all/stream",
    { method: "POST" },
    {
      onMessage: (data) => {
        if (typeof data.message === "string") onMessage(data.message);
      },
      onError,
      onComplete,
    },
  );
}

export async function uploadMarkdownFiles(files: File[], caseOption: string, newCaseName: string) {
  const formData = new FormData();
  files.forEach((file) => formData.append("files", file));
  formData.append("case_option", caseOption);
  formData.append("new_case_name", newCaseName);

  try {
    return await requestJson<ApiResult>("/api/rag/upload-markdown", {
      method: "POST",
      body: formData,
      timeoutMs: API_TIMEOUTS.upload,
    });
  } catch (error) {
    return failedResult(error);
  }
}

export async function exportChatHistory(
  history: unknown[],
  mode: string,
  caseId: string,
  exportFormat: string,
) {
  try {
    const blob = await requestBlob("/api/rag/export", {
      method: "POST",
      json: {
        history,
        mode,
        case_id: caseId,
        export_format: exportFormat,
      },
      timeoutMs: API_TIMEOUTS.download,
    });
    const extension = exportFormat === "timeline_docx" ? "docx" : exportFormat;
    downloadBlob(blob, `export_${mode}_${exportFormat}.${extension}`);
    return { success: true };
  } catch (error) {
    return failedResult(error);
  }
}

export async function createDockerContainer(payload: {
  hf_token?: string;
  port?: number;
  model?: string;
  gpu_mem?: number;
  max_model_len?: number;
  tensor_parallel_size?: number;
}) {
  try {
    return await jsonPost("/api/docker/create", payload);
  } catch (error) {
    return failedResult(error);
  }
}

export async function shutdownDockerContainer() {
  try {
    return await jsonPost("/api/docker/shutdown");
  } catch (error) {
    return failedResult(error);
  }
}

export async function startRagInfra() {
  try {
    return await jsonPost("/api/rag/infra/start");
  } catch (error) {
    return failedResult(error);
  }
}

export async function stopRagInfra() {
  try {
    return await jsonPost("/api/rag/infra/stop");
  } catch (error) {
    return failedResult(error);
  }
}

export async function fetchRagInfraStatus() {
  try {
    return await requestJson<ApiResult>("/api/rag/infra/status");
  } catch {
    return { postgres: "offline", redis: "offline", minio: "offline", qdrant: "offline" };
  }
}

export async function fetchCorpusStats() {
  try {
    return await requestJson<ApiResult>("/api/rag/corpus/stats");
  } catch {
    return { indexed_runs: 0, indexed_documents: 0, total_chunks: 0, unique_authors: 0 };
  }
}

export async function fetchDocumentRuns() {
  try {
    return await requestJson<ApiResult[]>("/api/documents/runs");
  } catch {
    return [];
  }
}

export async function fetchRunFiles(runName: string) {
  try {
    return await requestJson<string[]>(
      `/api/documents/runs/${apiPathSegment(runName)}/files`,
    );
  } catch {
    return [];
  }
}

export async function fetchMarkdownContent(runName: string, filename: string) {
  try {
    return await requestText(
      `/api/documents/runs/${apiPathSegment(runName)}/markdown/${apiPathSegment(filename)}`,
    );
  } catch {
    return "";
  }
}

export async function downloadRunMarkdownZip(runName: string) {
  try {
    const blob = await requestBlob(
      `/api/documents/runs/${apiPathSegment(runName)}/markdown.zip`,
      { timeoutMs: API_TIMEOUTS.download },
    );
    downloadBlob(blob, `${runName}_markdown.zip`);
    return { success: true };
  } catch (error) {
    return failedResult(error);
  }
}

export async function fetchDocumentInfo(runName: string, filename: string) {
  const query = new URLSearchParams({ filename });
  try {
    return await requestJson<ApiResult>(
      `/api/documents/runs/${apiPathSegment(runName)}/info?${query}`,
    );
  } catch {
    return null;
  }
}

export async function executeCleanup(components: string[]) {
  try {
    return await jsonPost("/api/diagnostics/cleanup", { components });
  } catch (error) {
    return failedResult(error);
  }
}

export async function downloadDiagnosticReport() {
  try {
    const blob = await requestBlob("/api/diagnostics/report", {
      timeoutMs: API_TIMEOUTS.download,
    });
    downloadBlob(blob, "diagnostic_report.md");
    return { success: true };
  } catch (error) {
    return failedResult(error);
  }
}

export async function fetchSettings() {
  try {
    return await requestJson<ApiResult>("/api/settings");
  } catch {
    return {};
  }
}

export async function updateSettings(payload: Record<string, unknown>) {
  try {
    return await requestJson<ApiResult>("/api/settings", {
      method: "PUT",
      json: payload,
    });
  } catch (error) {
    return failedResult(error);
  }
}

export interface InstalledModelItem {
  id: string;
  name: string;
  folder: string;
  path: string;
  cache_source?: string;
  copyCount?: number;
  size_bytes: number;
  human_size: string;
  context_length: number;
  model_type: string;
  is_active: boolean;
  is_stub?: boolean;
  modified_at: string;
}

export interface InstalledModelsResponse {
  models: InstalledModelItem[];
  total_count: number;
  total_size_bytes: number;
  total_human_size: string;
}

const EMPTY_MODELS: InstalledModelsResponse = {
  models: [],
  total_count: 0,
  total_size_bytes: 0,
  total_human_size: "0 B",
};

export async function fetchInstalledModels(): Promise<InstalledModelsResponse> {
  try {
    return await requestJson<InstalledModelsResponse>("/api/diagnostics/models");
  } catch {
    return EMPTY_MODELS;
  }
}

export async function deleteInstalledModels(modelIds: string[]) {
  try {
    return await requestJson<ApiResult>("/api/diagnostics/models", {
      method: "DELETE",
      json: { model_ids: modelIds },
    });
  } catch (error) {
    return {
      ...failedResult(error),
      deleted_models: [],
      reclaimed_bytes: 0,
      reclaimed_str: "0 B",
    };
  }
}
