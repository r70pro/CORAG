"""
Pydantic models for KIRAG API request/response schemas.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

# ── Pipeline ──────────────────────────────────────────────────────────────────


class PipelineStartRequest(BaseModel):
    """Request body to start an OCR pipeline run."""

    file_paths: list[str] = Field(..., description="Absolute paths to PDF files to process")
    server_url: str = Field("http://localhost:8000/v1", description="vLLM server URL")
    model_name: str = Field("allenai/olmOCR-2-7B-1025-FP8", description="OCR model to use")
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


# ── Docker ────────────────────────────────────────────────────────────────────


class DockerCreateRequest(BaseModel):
    """Parameters to create/recreate the vLLM inference container."""

    hf_token: str = ""
    port: int = Field(8000, ge=1, le=65535)
    model: str = "allenai/olmOCR-2-7B-1025-FP8"
    gpu_mem: float = Field(0.8, ge=0.1, le=1.0)
    max_model_len: int = Field(15360, ge=512)
    tensor_parallel_size: int = Field(1, ge=1, le=8)


class DockerStatusResponse(BaseModel):
    """vLLM container status."""

    status: str = Field(description="ready | starting | stopped | not_found | error")
    message: str = ""
    badge_html: str = ""


class DockerLogsResponse(BaseModel):
    """Container logs snapshot."""

    logs: str = ""
    container_status: str = ""


# ── RAG ───────────────────────────────────────────────────────────────────────


class RAGQueryRequest(BaseModel):
    """Request body for a RAG analysis query."""

    query: str = Field(..., min_length=1)
    mode: str = Field("free_qa", description="Analysis mode key")
    model_url: str = "http://localhost:8000/v1"
    model_name: str = "nvidia/Phi-4-reasoning-plus-NVFP4"
    top_k: int = Field(15, ge=1, le=100)
    case_id: str | None = None
    doc_type: str | None = None
    author: str | None = None
    date_from: str | None = None
    date_to: str | None = None
    use_reranker: bool = True
    reranker_model: str = "BAAI/bge-reranker-large"
    reranker_device: str = "cuda"
    stream: bool = Field(True, description="If True, response is SSE-streamed")


class IndexRunRequest(BaseModel):
    """Request body to index a specific run."""

    run_dir: str = Field(..., description="Absolute path to the run directory")


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

    server_url: str | None = None
    model_name: str | None = None
    workers: int | None = None
    max_concurrent_requests: int | None = None
    target_longest_image_dim: int | None = None
    max_page_retries: int | None = None
    guided_decoding: bool | None = None
    docker_port: int | None = None
    docker_gpu_mem: float | None = None
    docker_max_model_len: int | None = None
    docker_tensor_parallel: int | None = None
    hf_token: str | None = None
    analysis_model_name: str | None = None
    analysis_server_url: str | None = None
    embedding_model: str | None = None
    embedding_device: str | None = None
    chunk_size: int | None = None
    chunk_overlap: int | None = None
    retrieval_top_k: int | None = None
    rag_auto_start_infra: bool | None = None
    use_reranker: bool | None = None
    reranker_model: str | None = None
    reranker_device: str | None = None


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
    vector_dim: int = 1024
    metric: str = "Cosine Similarity"
    redis_cached_count: str = "N/A"
    telemetry_html: str = ""


class EmbeddingConfigRequest(BaseModel):
    """Configuration updates for dense vector embedding."""

    embedding_model: str = "BAAI/bge-large-en-v1.5"
    embedding_device: str = "auto"
    chunk_size: int = 800
    chunk_overlap: int = 100
    embedding_batch_size: int = 64


# ── Chat Export ───────────────────────────────────────────────────────────────


class ExportChatRequest(BaseModel):
    """Request payload to export chat history."""

    history: list[dict] = Field(
        ..., description="List of message objects [{role, content}] or tuples"
    )
    mode: str = Field("free_qa", description="Analysis mode key")
    case_id: str = Field("", description="Active case ID")
    export_format: str = Field("md", description="md | txt | csv | docx | timeline_docx")


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
