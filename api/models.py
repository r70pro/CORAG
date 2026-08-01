"""
Pydantic models for KIRAG API request/response schemas.
"""

from __future__ import annotations

from datetime import date
from typing import Annotated, Any

from pydantic import BaseModel, Field, model_validator

MAX_MODEL_CONTEXT_LENGTH = 1_048_576
MAX_QUERY_LENGTH = 32_768
ContextLength = Annotated[int, Field(ge=1, le=MAX_MODEL_CONTEXT_LENGTH)]

# ── Pipeline ──────────────────────────────────────────────────────────────────


class PipelineStartRequest(BaseModel):
    """Request body to start an OCR pipeline run."""

    file_paths: list[str] = Field(
        ...,
        min_length=1,
        max_length=100,
        description="PDF filenames previously returned by the pipeline upload endpoint",
    )
    server_url: str = Field(
        "http://localhost:8000/v1", min_length=1, max_length=2048, description="vLLM server URL"
    )
    model_name: str = Field(
        "allenai/olmOCR-2-7B-1025-FP8",
        min_length=1,
        max_length=512,
        description="OCR model to use",
    )
    workers: int = Field(4, ge=1, le=32)
    max_concurrent: int = Field(20, ge=1, le=100)
    max_retries: int = Field(8, ge=0, le=50)
    target_dim: int = Field(1288, ge=256, le=4096)
    guided_decoding: bool = True


class PipelineStatusResponse(BaseModel):
    """Status of a pipeline run."""

    run_id: str
    status: str = Field(description="running | completed | failed | stopped | unknown")
    completed_pages: int = 0
    failed_pages: int = 0
    total_pages: int = 0
    log_tail: str = ""


class RunInfo(BaseModel):
    """Summary of an available OCR run."""

    display_name: str
    run_dir: str
    run_id: str = ""
    file_count: int = 0
    is_indexed: bool = False


# ── Docker ────────────────────────────────────────────────────────────────────


class DockerCreateRequest(BaseModel):
    """Parameters to create/recreate the vLLM inference container."""

    hf_token: str = ""
    port: int = Field(8000, ge=1, le=65535)
    model: str = Field("allenai/olmOCR-2-7B-1025-FP8", min_length=1, max_length=512)
    gpu_mem: float = Field(0.8, ge=0.1, le=1.0)
    max_model_len: int = Field(15360, ge=512, le=MAX_MODEL_CONTEXT_LENGTH)
    tensor_parallel_size: int = Field(1, ge=1, le=8)


class AnalysisContextModeRequest(BaseModel):
    """Switch between shared-GPU and OCR-off extended analysis modes."""

    extended: bool


class StartupModeRequest(BaseModel):
    """Select and persist the model profile used for this and future sessions."""

    mode: str = Field(pattern="^(analysis_262k|dual_32k|ocr_only)$")


class DockerStatusResponse(BaseModel):
    """vLLM container status."""

    status: str = Field(description="ready | starting | stopped | not_found | foreign | error")
    message: str = ""
    badge_html: str = ""


class DockerLogsResponse(BaseModel):
    """Container logs snapshot."""

    logs: str = ""
    container_status: str = ""


class DockerModelsResponse(BaseModel):
    """List of available / cached model names and their max content lengths."""

    models: list[str] = Field(default_factory=list, description="Available model identifiers")
    max_lengths: dict[str, ContextLength] = Field(
        default_factory=dict, description="Max content length limits per model"
    )


# ── RAG ───────────────────────────────────────────────────────────────────────


class RAGQueryRequest(BaseModel):
    """Request body for a RAG analysis query."""

    query: str = Field(..., min_length=1, max_length=MAX_QUERY_LENGTH)
    mode: str = Field("free_qa", min_length=1, max_length=128, description="Analysis mode key")
    model_url: str = Field("http://localhost:8000/v1", min_length=1, max_length=2048)
    model_name: str = Field("nvidia/Phi-4-reasoning-plus-NVFP4", min_length=1, max_length=512)
    top_k: int = Field(15, ge=1, le=100)
    case_id: str | None = Field(None, max_length=256)
    doc_type: str | None = Field(None, max_length=128)
    author: str | None = Field(None, max_length=512)
    date_from: date | None = None
    date_to: date | None = None
    use_reranker: bool = True
    reranker_model: str = Field("BAAI/bge-reranker-large", min_length=1, max_length=512)
    reranker_device: str = Field("cuda", min_length=1, max_length=32)
    stream: bool = Field(True, description="If True, response is SSE-streamed")
    reasoning_audit: bool = Field(
        False,
        description="Request visible reasoning; honored only for verified administrators",
    )
    session_id: str | None = Field(None, min_length=1, max_length=128)

    @model_validator(mode="after")
    def validate_date_range(self):
        if self.date_from and self.date_to and self.date_from > self.date_to:
            raise ValueError("date_from must be on or before date_to")
        return self


class RAGQueryResponse(BaseModel):
    """Complete non-streaming RAG response."""

    response: str
    reasoning: str | None = None


class IndexRunRequest(BaseModel):
    """Request body to index a specific run."""

    run_dir: str = Field(..., description="Workspace run name beginning with run_")


class CorpusStatsResponse(BaseModel):
    """Corpus-level aggregate statistics."""

    indexed_runs: int = 0
    indexed_documents: int = 0
    total_chunks: int = 0
    unique_authors: int = 0
    earliest_date: str | None = None
    latest_date: str | None = None
    vectors_count: int = 0


class CaseInfo(BaseModel):
    """Summary of an indexed case."""

    label: str
    run_id: str


class InfraStatusResponse(BaseModel):
    """Status of RAG backing services."""

    postgres: str = "unknown"
    redis: str = "unknown"
    minio: str = "unknown"
    qdrant: str = "unknown"


# ── Settings ──────────────────────────────────────────────────────────────────


class SettingsUpdateRequest(BaseModel):
    """Partial settings update — only provided fields are merged."""

    server_url: str | None = Field(None, min_length=1, max_length=2048)
    model_name: str | None = Field(None, min_length=1, max_length=512)
    workers: int | None = Field(None, ge=1, le=32)
    max_concurrent_requests: int | None = Field(None, ge=1, le=100)
    target_longest_image_dim: int | None = Field(None, ge=256, le=4096)
    max_page_retries: int | None = Field(None, ge=0, le=50)
    guided_decoding: bool | None = None
    docker_port: int | None = Field(None, ge=1, le=65535)
    docker_gpu_mem: float | None = Field(None, ge=0.1, le=1.0)
    docker_max_model_len: int | None = Field(None, ge=512, le=MAX_MODEL_CONTEXT_LENGTH)
    docker_tensor_parallel: int | None = Field(None, ge=1, le=8)
    hf_token: str | None = Field(None, max_length=4096)
    analysis_model_name: str | None = Field(None, min_length=1, max_length=512)
    analysis_server_url: str | None = Field(None, min_length=1, max_length=2048)
    embedding_model: str | None = Field(None, min_length=1, max_length=512)
    embedding_device: str | None = Field(None, min_length=1, max_length=32)
    embedding_batch_size: int | None = Field(None, ge=1, le=1024)
    chunk_size: int | None = Field(None, ge=1, le=100_000)
    chunk_overlap: int | None = Field(None, ge=0, le=99_999)
    retrieval_top_k: int | None = Field(None, ge=1, le=100)
    rag_auto_start_infra: bool | None = None
    startup_mode: str | None = Field(
        None, pattern="^(analysis_262k|dual_32k|ocr_only)$"
    )
    use_reranker: bool | None = None
    reranker_model: str | None = Field(None, min_length=1, max_length=512)
    reranker_device: str | None = Field(None, min_length=1, max_length=32)

    @model_validator(mode="after")
    def validate_chunk_overlap(self):
        if (
            self.chunk_size is not None
            and self.chunk_overlap is not None
            and self.chunk_overlap >= self.chunk_size
        ):
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        return self


# ── Diagnostics ───────────────────────────────────────────────────────────────


class ServiceHealth(BaseModel):
    """Health status of a single backing service."""

    name: str
    is_up: bool
    latency_ms: float = 0.0
    extra_info: str | None = None


class GPUInfo(BaseModel):
    """GPU hardware metrics."""

    cuda_available: bool = False
    gpu_name: str = "N/A"
    vram_used_mb: float = 0.0
    vram_total_mb: float = 0.0
    vram_pct: float = 0.0
    vram_free_mb: float = 0.0
    vram_reclaimable_mb: float = 0.0
    processes: list[dict] = Field(default_factory=list)


class HealthResponse(BaseModel):
    """Full system health snapshot."""

    overall: str = Field(description="healthy | degraded")
    services: list[ServiceHealth] = Field(default_factory=list)
    gpu: GPUInfo = Field(default_factory=GPUInfo)


# ── Generic ───────────────────────────────────────────────────────────────────


class MessageResponse(BaseModel):
    """Simple status/message envelope."""

    success: bool = True
    message: str = ""


class AppShutdownRequest(BaseModel):
    """Explicit acknowledgement required before stopping KIRAG."""

    confirmation: str = Field(pattern="^SHUTDOWN$")


class ErrorDetail(BaseModel):
    """Machine-readable API error."""

    code: str
    message: str
    details: Any | None = None


class ErrorEnvelope(BaseModel):
    """Consistent error body returned for non-successful HTTP responses."""

    error: ErrorDetail


# ── Case Management & Deletion ────────────────────────────────────────────────


class DeleteCasesRequest(BaseModel):
    """Request body to delete one, multiple, or all indexed cases."""

    run_ids: list[str] = Field(default_factory=list, description="List of run IDs to delete")
    delete_all: bool = Field(
        False, description="If True, delete all cases from vector store and DB"
    )


# ── Embedding & Vector Store ──────────────────────────────────────────────────


class EmbeddingTelemetryResponse(BaseModel):
    """Telemetry data for Qdrant vector store and Redis embedding cache."""

    active_device: str = "auto"
    device_target: str = "auto"
    qdrant_points: int = 0
    collection_name: str = "cases"
    vector_dim: int = Field(1024, ge=1, le=100_000)
    metric: str = "Cosine Similarity"
    redis_cached_count: str = "N/A"
    telemetry_html: str = ""


class EmbeddingConfigRequest(BaseModel):
    """Configuration updates for dense vector embedding."""

    embedding_model: str = Field("BAAI/bge-large-en-v1.5", min_length=1, max_length=512)
    embedding_device: str = Field("auto", min_length=1, max_length=32)
    chunk_size: int = Field(800, ge=1, le=100_000)
    chunk_overlap: int = Field(100, ge=0, le=99_999)
    embedding_batch_size: int = Field(64, ge=1, le=1024)

    @model_validator(mode="after")
    def validate_chunk_overlap(self):
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        return self


# ── Chat Export ───────────────────────────────────────────────────────────────


class ExportChatRequest(BaseModel):
    """Request payload to export chat history."""

    history: list[dict] = Field(
        ..., description="List of message objects [{role, content}] or tuples"
    )
    mode: str = Field("free_qa", description="Analysis mode key")
    case_id: str = Field("", description="Active case ID")
    export_format: str = Field("md", description="md | txt | csv | docx | timeline_docx")
    include_reasoning: bool = Field(False, description="Admin-only reasoning audit export")


# ── Diagnostics & Cleanup ─────────────────────────────────────────────────────


class CleanupRequest(BaseModel):
    """Request body for disk space cleanup."""

    components: list[str] = Field(
        default_factory=list,
        description="List of component keys to clean e.g. ['runs', 'cache', 'logs', 'embeddings']",
    )


class CleanupResponse(BaseModel):
    """Response envelope for system cleanup."""

    success: bool = True
    message: str = ""
    reclaimed_bytes: int = 0
    reclaimed_str: str = ""


class InstalledModelItem(BaseModel):
    """Metadata for an installed / cached HuggingFace model."""

    id: str = Field(..., description="HuggingFace model ID e.g. allenai/olmOCR-2-7B-1025-FP8")
    name: str = Field(..., description="Short model name")
    folder: str = Field(
        ..., description="Cache directory name e.g. models--allenai--olmOCR-2-7B-1025-FP8"
    )
    path: str = Field(..., description="Absolute path on disk")
    cache_source: str = Field(
        "User HF Cache",
        description="Source location description e.g. KIRAG Workspace | User HF Cache | IQRAG Cache",
    )
    size_bytes: int = Field(..., description="Total size in bytes")
    human_size: str = Field(..., description="Formatted size string e.g. 19.10 GB")
    context_length: int = Field(
        ..., ge=1, le=MAX_MODEL_CONTEXT_LENGTH, description="Context window max token length"
    )
    model_type: str = Field("LLM", description="Vision LLM | LLM | Embedding | Reranker")
    is_active: bool = Field(
        False, description="True if currently loaded in active container or settings"
    )
    is_stub: bool = Field(
        False,
        description="True if folder contains only reference stub files without real weight blobs",
    )
    modified_at: str = Field("", description="Last modified timestamp")


class InstalledModelsResponse(BaseModel):
    """List of all installed models with summary disk metrics."""

    models: list[InstalledModelItem] = Field(default_factory=list)
    total_count: int = 0
    total_size_bytes: int = 0
    total_human_size: str = "0 B"


class DeleteModelsRequest(BaseModel):
    """Payload to delete selected cached models."""

    model_ids: list[str] = Field(..., description="List of model IDs or folder names to remove")


class DeleteModelsResponse(BaseModel):
    """Response from model deletion operation."""

    success: bool = True
    message: str = ""
    deleted_models: list[str] = Field(default_factory=list)
    reclaimed_bytes: int = 0
    reclaimed_str: str = "0 B"
