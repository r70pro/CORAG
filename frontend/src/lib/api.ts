const getApiBaseUrl = () => {
  if (process.env.NEXT_PUBLIC_API_URL) {
    return process.env.NEXT_PUBLIC_API_URL;
  }
  if (typeof window !== "undefined") {
    const host = window.location.hostname;
    // Standardize localhost -> 127.0.0.1 to prevent Firefox IPv6 ::1 resolution connection errors
    const targetHost = !host || host === "localhost" ? "127.0.0.1" : host;
    return `${window.location.protocol}//${targetHost}:8001`;
  }
  return "http://127.0.0.1:8001";
};

export const API_BASE_URL = getApiBaseUrl();

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
}

// Fetch System Health
export async function fetchSystemHealth() {
  try {
    const res = await fetch(`${API_BASE_URL}/api/health`);
    if (!res.ok) {
      return { status: "error", message: `HTTP ${res.status}: ${res.statusText}`, services: [], gpu: null };
    }
    return await res.json();
  } catch (err) {
    return { status: "offline", error: String(err), services: [], gpu: null };
  }
}

// Fetch Case Summaries
export async function fetchCaseSummary() {
  try {
    const res = await fetch(`${API_BASE_URL}/api/case-summary`);
    return await res.json();
  } catch (err) {
    return { error: String(err) };
  }
}

// Fetch Case Timeline Events
export async function fetchCaseTimeline(runId: string) {
  try {
    const res = await fetch(`${API_BASE_URL}/api/cases/${runId}/timeline`);
    return await res.json();
  } catch (err) {
    return { run_id: runId, events: [], error: String(err) };
  }
}


// Fetch Pipeline Runs
export async function fetchPipelineRuns() {
  try {
    const res = await fetch(`${API_BASE_URL}/api/pipeline/runs`);
    return await res.json();
  } catch {
    return [];
  }
}

// Fetch Docker Server Status
export async function fetchDockerStatus() {
  try {
    const res = await fetch(`${API_BASE_URL}/api/docker/status`);
    return await res.json();
  } catch {
    return { status: "unknown", is_ready: false };
  }
}

// Fetch Cached / Available Docker Models
export async function fetchDockerModels() {
  try {
    const res = await fetch(`${API_BASE_URL}/api/docker/models`);
    if (!res.ok) return { models: [] };
    return await res.json();
  } catch {
    return { models: [] };
  }
}


// Fetch Docker Container Logs
export async function fetchDockerLogs(tail: number = 200) {
  try {
    const res = await fetch(`${API_BASE_URL}/api/docker/logs?tail=${tail}`);
    if (!res.ok) return { logs: `HTTP ${res.status}: Failed to fetch container logs`, container_status: "error" };
    return await res.json();
  } catch (err) {
    return { logs: `Error fetching container logs: ${String(err)}`, container_status: "error" };
  }
}

// Start Docker Container
export async function startDockerContainer(payload: {
  hf_token?: string;
  model_name?: string;
  port?: number;
  gpu_mem?: number;
  max_model_len?: number;
}) {
  try {
    const res = await fetch(`${API_BASE_URL}/api/docker/start`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    return await res.json();
  } catch (err) {
    return { success: false, message: String(err) };
  }
}

// Stop Docker Container
export async function stopDockerContainer() {
  try {
    const res = await fetch(`${API_BASE_URL}/api/docker/stop`, { method: "POST" });
    return await res.json();
  } catch (err) {
    return { success: false, message: String(err) };
  }
}

// Trigger Ingestion Pipeline via SSE
export function triggerIngestSSE(
  payload: PipelineStartPayload,
  onMessage: (data: unknown) => void,
  onError: (err: unknown) => void,
  onComplete: () => void
) {
  fetch(`${API_BASE_URL}/api/pipeline/start`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  })
    .then((res) => {
      if (!res.body) return;
      const reader = res.body.getReader();
      const decoder = new TextDecoder();

      function readChunk() {
        reader.read().then(({ done, value }) => {
          if (done) {
            onComplete();
            return;
          }
          const text = decoder.decode(value);
          const lines = text.split("\n\n");
          for (const line of lines) {
            if (line.startsWith("data: ")) {
              const dataStr = line.replace("data: ", "").trim();
              if (dataStr === "[DONE]") {
                onComplete();
                return;
              }
              try {
                const parsed = JSON.parse(dataStr);
                onMessage(parsed);
              } catch {
                // Ignore parse errors
              }
            }
          }
          readChunk();
        });
      }

      readChunk();
    })
    .catch((err) => onError(err));
}

// Stop pipeline run
export async function stopPipelineRun(runId: string = "") {

  try {
    const res = await fetch(`${API_BASE_URL}/api/pipeline/stop/${runId}`, { method: "POST" });
    return await res.json();
  } catch (err) {
    return { success: false, message: String(err) };
  }
}

// Trigger RAG Query via SSE
export function triggerRagChatSSE(
  payload: RagQueryPayload,
  onChunk: (chunk: string) => void,
  onError: (err: unknown) => void,
  onComplete: () => void
) {
  fetch(`${API_BASE_URL}/api/rag/query`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ ...payload, stream: true }),
  })
    .then((res) => {
      if (!res.body) return;
      const reader = res.body.getReader();
      const decoder = new TextDecoder();

      function readChunk() {
        reader.read().then(({ done, value }) => {
          if (done) {
            onComplete();
            return;
          }
          const text = decoder.decode(value);
          const lines = text.split("\n\n");
          for (const line of lines) {
            if (line.startsWith("data: ")) {
              const dataStr = line.replace("data: ", "").trim();
              if (dataStr === "[DONE]") {
                onComplete();
                return;
              }
              try {
                const parsed = JSON.parse(dataStr);
                if (parsed.chunk) {
                  onChunk(parsed.chunk);
                }
              } catch {
                // Ignore
              }
            }
          }
          readChunk();
        });
      }

      readChunk();
    })
    .catch((err) => onError(err));
}

// Delete cases
export async function deleteCases(runIds: string[], deleteAll: boolean = false) {
  try {
    const res = await fetch(`${API_BASE_URL}/api/rag/cases/delete`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ run_ids: runIds, delete_all: deleteAll }),
    });
    return await res.json();
  } catch (err) {
    return { success: false, message: String(err) };
  }
}

// Fetch Embedding Telemetry
export async function fetchEmbeddingTelemetry() {
  try {
    const res = await fetch(`${API_BASE_URL}/api/rag/embedding/telemetry`);
    return await res.json();
  } catch (err) {
    return { telemetry_html: `<div style='color:red;'>Error loading telemetry: ${String(err)}</div>` };
  }
}

// Save Embedding Configuration
export async function saveEmbeddingConfig(config: {
  embedding_model: string;
  embedding_device: string;
  chunk_size: number;
  chunk_overlap: number;
  embedding_batch_size: number;
}) {
  try {
    const res = await fetch(`${API_BASE_URL}/api/rag/embedding/config`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(config),
    });
    return await res.json();
  } catch (err) {
    return { success: false, message: String(err) };
  }
}

// Purge Vector Cache
export async function purgeVectorCache() {
  try {
    const res = await fetch(`${API_BASE_URL}/api/rag/embedding/purge-cache`, { method: "POST" });
    return await res.json();
  } catch (err) {
    return { success: false, message: String(err) };
  }
}

// Index Selected Run
export async function indexRun(runDir: string) {
  try {
    const res = await fetch(`${API_BASE_URL}/api/rag/index`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ run_dir: runDir }),
    });
    return await res.json();
  } catch (err) {
    return { success: false, message: String(err) };
  }
}

// Index All Runs
export async function indexAllRuns() {
  try {
    const res = await fetch(`${API_BASE_URL}/api/rag/index-all`, { method: "POST" });
    return await res.json();
  } catch (err) {
    return { success: false, message: String(err) };
  }
}

// Upload & Index Markdown Files
export async function uploadMarkdownFiles(files: File[], caseOption: string, newCaseName: string) {
  try {
    const formData = new FormData();
    files.forEach((f) => formData.append("files", f));
    formData.append("case_option", caseOption);
    formData.append("new_case_name", newCaseName);

    const res = await fetch(`${API_BASE_URL}/api/rag/upload-markdown`, {
      method: "POST",
      body: formData,
    });
    return await res.json();
  } catch (err) {
    return { success: false, message: String(err) };
  }
}

// Export Chat History
export async function exportChatHistory(
  history: unknown[],
  mode: string,
  caseId: string,
  exportFormat: string
) {
  try {
    const res = await fetch(`${API_BASE_URL}/api/rag/export`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        history,
        mode,
        case_id: caseId,
        export_format: exportFormat,
      }),
    });
    if (res.ok) {
      const blob = await res.blob();
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `export_${mode}_${exportFormat}.${exportFormat === "timeline_docx" ? "docx" : exportFormat}`;
      document.body.appendChild(a);
      a.click();
      a.remove();
      return { success: true };
    } else {
      return { success: false, message: "Export failed" };
    }
  } catch (err) {
    return { success: false, message: String(err) };
  }
}

// Create/Recreate Docker Container
export async function createDockerContainer(payload: {
  hf_token?: string;
  port?: number;
  model?: string;
  gpu_mem?: number;
  max_model_len?: number;
}) {
  try {
    const res = await fetch(`${API_BASE_URL}/api/docker/create`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    return await res.json();
  } catch (err) {
    return { success: false, message: String(err) };
  }
}

// Shutdown Docker Container
export async function shutdownDockerContainer() {
  try {
    const res = await fetch(`${API_BASE_URL}/api/docker/shutdown`, { method: "POST" });
    return await res.json();
  } catch (err) {
    return { success: false, message: String(err) };
  }
}

// RAG Infrastructure Start / Stop / Status
export async function startRagInfra() {
  try {
    const res = await fetch(`${API_BASE_URL}/api/rag/infra/start`, { method: "POST" });
    return await res.json();
  } catch (err) {
    return { success: false, message: String(err) };
  }
}

export async function stopRagInfra() {
  try {
    const res = await fetch(`${API_BASE_URL}/api/rag/infra/stop`, { method: "POST" });
    return await res.json();
  } catch (err) {
    return { success: false, message: String(err) };
  }
}

export async function fetchRagInfraStatus() {
  try {
    const res = await fetch(`${API_BASE_URL}/api/rag/infra/status`);
    return await res.json();
  } catch {
    return { postgres: "offline", redis: "offline", minio: "offline", qdrant: "offline" };
  }
}

// Fetch Corpus Stats
export async function fetchCorpusStats() {
  try {
    const res = await fetch(`${API_BASE_URL}/api/rag/corpus/stats`);
    return await res.json();
  } catch {
    return { indexed_runs: 0, indexed_documents: 0, total_chunks: 0, unique_authors: 0 };
  }
}

// Document browsing APIs
export async function fetchDocumentRuns() {
  try {
    const res = await fetch(`${API_BASE_URL}/api/documents/runs`);
    return await res.json();
  } catch {
    return [];
  }
}

export async function fetchRunFiles(runName: string) {
  try {
    const res = await fetch(`${API_BASE_URL}/api/documents/runs/${runName}/files`);
    return await res.json();
  } catch {
    return [];
  }
}

export async function fetchMarkdownContent(runName: string, filename: string) {
  try {
    const res = await fetch(`${API_BASE_URL}/api/documents/runs/${runName}/markdown/${filename}`);
    return await res.text();
  } catch {
    return "";
  }
}

export async function fetchDocumentInfo(runName: string, filename: string) {
  try {
    const res = await fetch(
      `${API_BASE_URL}/api/documents/runs/${runName}/info?filename=${encodeURIComponent(filename)}`
    );
    if (!res.ok) return null;
    return await res.json();
  } catch {
    return null;
  }
}

// Diagnostics Cleanup & Report APIs
export async function executeCleanup(components: string[]) {
  try {
    const res = await fetch(`${API_BASE_URL}/api/diagnostics/cleanup`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ components }),
    });
    if (!res.ok) {
      const errData = await res.json().catch(() => null);
      const msg = errData?.detail || errData?.message || `HTTP ${res.status}: ${res.statusText}`;
      return { success: false, message: msg };
    }
    return await res.json();
  } catch (err) {
    return {
      success: false,
      message: `Network error connecting to API server at ${API_BASE_URL}: ${String(err)}`,
    };
  }
}

export async function fetchSettings() {
  try {
    const res = await fetch(`${API_BASE_URL}/api/settings/`);
    return await res.json();
  } catch {
    return {};
  }
}

export async function updateSettings(payload: Record<string, unknown>) {
  try {
    const res = await fetch(`${API_BASE_URL}/api/settings/`, {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    return await res.json();
  } catch (err) {
    return { success: false, message: String(err) };
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

export async function fetchInstalledModels(): Promise<InstalledModelsResponse> {
  try {
    const res = await fetch(`${API_BASE_URL}/api/diagnostics/models`);
    if (!res.ok) {
      return { models: [], total_count: 0, total_size_bytes: 0, total_human_size: "0 B" };
    }
    return await res.json();
  } catch (err) {
    console.error("Failed to fetch installed models:", err);
    return { models: [], total_count: 0, total_size_bytes: 0, total_human_size: "0 B" };
  }
}

export async function deleteInstalledModels(modelIds: string[]) {
  try {
    const res = await fetch(`${API_BASE_URL}/api/diagnostics/models`, {
      method: "DELETE",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ model_ids: modelIds }),
    });
    if (!res.ok) {
      const errData = await res.json().catch(() => null);
      const msg = errData?.detail || errData?.message || `HTTP ${res.status}: ${res.statusText}`;
      return { success: false, message: msg, deleted_models: [], reclaimed_bytes: 0, reclaimed_str: "0 B" };
    }
    return await res.json();
  } catch (err) {
    return {
      success: false,
      message: `Network error connecting to API server at ${API_BASE_URL}: ${String(err)}`,
      deleted_models: [],
      reclaimed_bytes: 0,
      reclaimed_str: "0 B",
    };
  }
}


