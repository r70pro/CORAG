# KIRAG: PDF OCR and medicolegal RAG workstation

KIRAG is a local-first workstation for converting PDFs to Markdown with an
olmOCR-compatible vision model, indexing the extracted text, and analysing it
with a retrieval-augmented language model. The Gradio interface is branded
**IQ-RAG Client**; the managed vLLM container is named `olmocr`.

The current application version is **2.0.3**. The version, supported inference
models, context limits, and runtime defaults are defined in
[`settings_manager.py`](settings_manager.py), not in this document.

> KIRAG assists review; it does not replace source-document verification or
> professional judgement. Metadata extraction, retrieval, and model output can
> be incomplete or wrong.

For the practitioner workflow, provenance rules, and case-management guidance,
see [`medicolegal_rag_guide.md`](medicolegal_rag_guide.md).

## What is implemented

- Layout-aware PDF-to-Markdown OCR through `python -m olmocr.pipeline` and an
  OpenAI-compatible vLLM endpoint.
- Six-view Gradio UI: Ingestion Pipeline, Layout Inspector, Embedding Pipeline,
  Case Dashboard, RAG Processing, and System Diagnostics.
- Alternative six-view Next.js frontend backed by the FastAPI API.
- Medicolegal-aware character chunking with heuristic extraction of dates,
  authors, document types, section types, patient names, and PDF page spans.
- Sentence-Transformer embeddings, model-specific Qdrant collections, optional
  Cross-Encoder reranking, and MMR-style diversity ranking.
- PostgreSQL registry, Redis caches, MinIO object storage, and Qdrant vector
  storage managed by Docker Compose.
- Five analysis modes: Free Q&A, Timeline, Injury Summary, Inconsistency Finder,
  and Medication Tracker.
- Markdown, text, timeline CSV, analysis DOCX, and timeline DOCX exports.
- REST API, headless CLI, diagnostics, reconciliation reporting, and managed
  vLLM/container lifecycle operations.

## Architecture and data flow

```mermaid
flowchart LR
    PDF[PDF uploads] --> OCR[olmOCR pipeline]
    VLLM[vLLM OpenAI-compatible server] --> OCR
    OCR --> RUN[workspace/run_*/\ninputs + results + markdown/inputs]
    MD[External Markdown] --> INDEX[Indexing service]
    RUN --> REVIEW[Layout Inspector]
    RUN --> INDEX
    INDEX --> PG[(PostgreSQL\nruns, documents, chunks)]
    INDEX --> QD[(Qdrant\nmodel-specific vectors)]
    INDEX --> MINIO[(MinIO\nPDF/Markdown objects)]
    INDEX --> REDIS[(Redis\nembedding cache + counters)]
    QUERY[Query + case/metadata filters] --> RETRIEVE[Dense retrieval\nreranking + diversity]
    PG --> RETRIEVE
    QD --> RETRIEVE
    RETRIEVE --> ANALYSE[Prompt + local/configured LLM]
    ANALYSE --> OUTPUT[Streaming answer and exports]
```

The primary implementation boundaries are:

| Area | Source |
|---|---|
| Gradio application and callbacks | [`app.py`](app.py), [`app_handlers.py`](app_handlers.py), [`rag_ui.py`](rag_ui.py), [`embedding_pipeline_ui.py`](embedding_pipeline_ui.py) |
| OCR process orchestration | [`pipeline_manager.py`](pipeline_manager.py), [`pdf_manager.py`](pdf_manager.py) |
| Indexing transaction and object archival | [`indexing_service.py`](indexing_service.py) |
| RAG core | [`rag/`](rag/) |
| Infrastructure lifecycle | [`rag_infra_manager.py`](rag_infra_manager.py), [`docker-compose.rag.yml`](docker-compose.rag.yml) |
| vLLM lifecycle and policy | [`docker_manager.py`](docker_manager.py) |
| REST API | [`api/main.py`](api/main.py), [`api/routes/`](api/routes/) |
| CLI | [`cli.py`](cli.py) |
| Next.js frontend and server-side API proxy | [`frontend/`](frontend/) |
| Configuration and secrets | [`settings_manager.py`](settings_manager.py), [`secrets_config.py`](secrets_config.py), [`.env.example`](.env.example) |

## Prerequisites

- Python **3.10 or newer**. The reproducible lock files target Python 3.12 on
  Linux x86_64.
- Docker Engine with the `docker compose` plugin for the RAG services.
- An NVIDIA GPU, compatible driver, and NVIDIA Container Toolkit to use the
  managed vLLM OCR/analysis container (`docker run --gpus all`). A CPU Python
  environment can run tests and CPU embeddings, but it does not make the
  managed vLLM container CPU-compatible.
- Node.js/npm compatible with Next.js 16 if using the alternative frontend.
- Sufficient storage for model weights and the persistent `workspace/` data.

PDF rendering is performed by `pypdfium2`; Poppler is not a project runtime
requirement.

## Installation

Create a virtual environment inside or outside the checkout. For the pinned
Linux x86_64 environments:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

CPU-only Python environment:

```bash
python -m pip install --require-hashes -r requirements-cpu.lock
python -m pip install --no-deps .
```

CUDA environment:

```bash
python -m pip install --require-hashes -r requirements-cuda.lock
python -m pip install --no-deps .
```

For an unpinned supported-platform install, use `python -m pip install .`.
The package installs the `kirag` console command. The lock files are generated
from `pyproject.toml` plus `requirements-cpu.in` or `requirements-cuda.in` by
[`scripts/compile-locks.sh`](scripts/compile-locks.sh).

## Configuration

Copy the environment template and replace every placeholder used by your
deployment:

```bash
cp .env.example .env
```

At minimum, review:

| Variable | Purpose |
|---|---|
| `HF_TOKEN` | Access to gated Hugging Face models; not required for public models already cached |
| `OLMOCR_PG_PASS` | PostgreSQL password used by both the app and Compose |
| `OLMOCR_MINIO_ACCESS_KEY` / `OLMOCR_MINIO_SECRET_KEY` | MinIO credentials used by both the app and Compose |
| `OLMOCR_VLLM_IMAGE` | Immutable `repository@sha256:<digest>` vLLM image |
| `KIRAG_API_KEY` | General REST API credential |
| `KIRAG_ADMIN_API_KEY` | Separate credential for administrative REST operations |
| `KIRAG_GRADIO_USERNAME` / `KIRAG_GRADIO_PASSWORD` | Required together before Gradio may bind beyond loopback |
| `KIRAG_API_URL` | FastAPI origin used by the Next.js server-side proxy |
| `KIRAG_MAX_*` | PDF and Markdown upload count/per-file/aggregate limits |

`.env` and `.env.*` are ignored except example files. `settings.json` is the
application's persistent UI/CLI settings file and is tracked in this checkout;
the Gradio UI can save a Hugging Face token into it. Do not save a real token
there in a shared checkout or commit one. Environment values take precedence
where the relevant code explicitly reads them.

Runtime defaults include:

- OCR model: `allenai/olmOCR-2-7B-1025-FP8`
- Analysis model: `nvidia/Phi-4-reasoning-plus-NVFP4`
- vLLM URLs: `http://localhost:8000/v1`
- Embedding model: `BAAI/bge-large-en-v1.5`
- Embedding device: `auto` (CUDA when available, otherwise CPU)
- Chunk size/overlap: 800/100 **characters**
- Retrieval Top-K: 15 in code defaults (a saved `settings.json` value overrides it)
- Reranker: `BAAI/bge-reranker-large`, enabled, device `cuda`

The supported vLLM model allowlist and maximum context lengths are in
[`settings_manager.py`](settings_manager.py). Custom models require the explicit
`KIRAG_ADVANCED_MODEL_OVERRIDE=true` administrator setting. Remote model code
is independently disabled by default and has additional dedicated-network and
scoped-token requirements documented in [`.env.example`](.env.example).
The verified Qwen 3.6 option is `Qwen/Qwen3.6-35B-A3B`; the incompatible NVIDIA
NVFP4 checkpoint is intentionally not offered by the managed model selector.
For Qwen3-family models, the managed container configures vLLM's `qwen3`
reasoning parser and RAG chat requests set `enable_thinking=false`; analysis
responses and exports therefore contain the answer rather than exposed model
reasoning.

## Running KIRAG

### Gradio workstation

Start and initialise the RAG stack through the CLI:

```bash
kirag rag infra start
```

Then launch Gradio:

```bash
python app.py
```

Open `http://127.0.0.1:7860`. Gradio defaults to loopback. A non-loopback
`KIRAG_GRADIO_HOST` is rejected unless both Gradio credentials are configured.

The RAG stack can also be started from the RAG Processing view. Running only
`docker compose -f docker-compose.rag.yml up -d` starts the containers but does
not call KIRAG's PostgreSQL schema, MinIO bucket, and Qdrant collection
initialisers; use `kirag rag infra start` or the UI for first-time setup.

The inference container is separate from the four-service RAG stack. Create it
from the Gradio sidebar or with, for example:

```bash
kirag docker create \
  --model allenai/olmOCR-2-7B-1025-FP8 \
  --port 8000 \
  --gpu-mem 0.8 \
  --max-model-len 15360
```

### FastAPI server

Set two independent secrets before starting the API:

```bash
export KIRAG_API_KEY='replace-with-a-long-random-value'
export KIRAG_ADMIN_API_KEY='replace-with-a-different-long-random-value'
uvicorn api.main:app --host 127.0.0.1 --port 8001
```

`GET /health` and CORS preflight `OPTIONS` requests bypass API authentication.
Every other route, including `/docs`, requires a valid general or admin key.
Administrative routes additionally require the admin key. If neither key is
configured, protected requests fail closed rather than becoming anonymous. The
API refuses a non-loopback bind when API authentication is not configured.

Example authenticated health request:

```bash
curl -H "X-API-Key: $KIRAG_API_KEY" http://127.0.0.1:8001/api/health
```

### Next.js frontend

The browser calls a same-origin Next.js route. That server-side proxy injects
API credentials and forwards uploads, byte ranges, downloads, and SSE; secrets
must never use a `NEXT_PUBLIC_*` variable.

```bash
cd frontend
npm ci
export KIRAG_API_URL='http://127.0.0.1:8001'
export KIRAG_API_KEY='replace-with-the-same-general-key'
export KIRAG_ADMIN_API_KEY='replace-with-the-same-admin-key'
npm run dev -- --hostname 127.0.0.1
```

Run FastAPI separately as shown above, then open `http://127.0.0.1:3000`.
Deployment profiles and reverse-proxy requirements are documented in
[`frontend/README.md`](frontend/README.md).

## REST API contract

All `/api/*` calls below require `X-API-Key`, `X-Admin-API-Key`, or a Bearer
token matching a configured key. Rows marked **admin** also enforce the
dedicated admin key.

| Domain | Routes |
|---|---|
| Pipeline | `POST /api/pipeline/upload`, `POST /api/pipeline/start` (SSE), `GET /api/pipeline/runs`, `GET /api/pipeline/status/{run_id}`, `POST /api/pipeline/stop/{run_id}` (**admin**) |
| Docker | `GET /api/docker/models`, `GET /api/docker/status`, `GET /api/docker/logs`; start/stop/create/shutdown are **admin** |
| RAG | query, export, telemetry, config, external Markdown upload, corpus stats/cases, infrastructure start/status; cache purge, infrastructure stop, indexing, and case deletion are **admin** |
| Documents | list runs/files, retrieve Markdown, first run PDF, and page-map information |
| Diagnostics | health, GPU, services, report, installed models; cleanup and model deletion are **admin** |
| Settings | read settings (HF token masked); update settings is **admin** |
| Convenience | `POST /api/ingest`, `POST /api/chat`, `GET /api/health`, `GET /api/case-summary`, `GET /api/cases/{run_id}/timeline` |

Inspect the authenticated `/docs` endpoint or [`api/routes/`](api/routes/) for
the complete schemas.

API OCR is deliberately a two-step flow. `/api/pipeline/start` accepts only
server-generated filenames returned by `/api/pipeline/upload`, not arbitrary or
absolute host paths:

```bash
curl -H "X-API-Key: $KIRAG_API_KEY" \
  -F 'files=@case.pdf;type=application/pdf' \
  http://127.0.0.1:8001/api/pipeline/upload

# Substitute the returned opaque .pdf filename.
curl -N -H "X-API-Key: $KIRAG_API_KEY" \
  -H 'Content-Type: application/json' \
  -d '{"file_paths":["RETURNED_FILENAME.pdf"]}' \
  http://127.0.0.1:8001/api/pipeline/start
```

`POST /api/rag/index` likewise accepts a workspace run **name** such as
`run_20260726_120000_ab12cd34`, not an absolute path, and requires the admin key.

## CLI contract

Use either `kirag` after installation or `python cli.py` from the checkout:

```text
kirag pipeline runs|status|stop
kirag docker status|start|stop|create|shutdown
kirag rag query|index|index-all|stats|reconcile
kirag rag infra start|stop|status
kirag diagnostics health|gpu|report
kirag settings show|set
```

Examples:

```bash
kirag pipeline runs
kirag rag index /absolute/path/to/the/configured/workspace/run_name
kirag rag index /absolute/path/to/the/configured/workspace/run_name --full-reindex
kirag rag query 'List documented operations' --mode timeline --case RUN_ID
kirag rag reconcile --fail-on-drift
kirag diagnostics report
kirag settings set embedding_device cpu
```

The CLI does **not** provide a command to upload or start a new OCR batch; use
Gradio or the REST API for ingestion. `rag index` validates that the resolved
directory is exactly beneath KIRAG's configured workspace. Obtain its path from
`kirag pipeline runs` rather than assuming a fixed location.

## Storage and deletion semantics

KIRAG normally uses `<checkout>/workspace`. If that location is not writable,
it falls back to `~/.local/share/kirag/workspace`. Each OCR batch gets a
`run_<timestamp>_<uuid-prefix>/` directory. Completed runs are discoverable when
they contain Markdown under `markdown/inputs/`.

Indexing computes a stable 16-character SHA-256-derived `run_id` from the
resolved run path. The run ID is attached to PostgreSQL rows, Qdrant payloads,
and MinIO keys. Indexing commits PostgreSQL and journals Qdrant mutations so a
failed vector operation can restore the prior point set. MinIO upload occurs
after the searchable index commits and is a warning-only step, so an “indexed”
run is not proof that every object was archived successfully.

Case deletion removes the targeted PostgreSQL rows, Qdrant vectors, MinIO
objects, and matching local run directory. “Delete all” also sweeps all valid
run directories from the configured workspace. The diagnostics cleanup can
delete inactive run directories without removing their indexed database/vector
records. Treat both operations as destructive.

## Security and privacy boundaries

- Compose publishes PostgreSQL, Redis, MinIO, and Qdrant only on `127.0.0.1`.
  The managed vLLM port is also published on loopback.
- These backing services are not all application-authenticated. Loopback
  binding is a boundary, not a reason to expose them through a firewall or
  reverse proxy.
- Model downloads contact Hugging Face. A custom `server_url`,
  `analysis_server_url`, MinIO endpoint, database host, Redis host, or Qdrant
  host can send data off the workstation. “Local-first” is not a guarantee that
  a modified configuration is offline.
- Data at rest is not encrypted by KIRAG. Protect `.env`, `settings.json`,
  `workspace/`, exports, logs, backups, and host/model caches with filesystem,
  disk-encryption, backup, and retention controls appropriate to the matter.
- API keys authenticate the API; they do not provide per-user authorisation or
  matter-level access control. The Gradio “Active Role” selector is a UI choice,
  not an access-control boundary.
- Lifecycle and deletion events are written to a structured audit JSONL file,
  but the PostgreSQL registry and this log do not by themselves establish a
  legally sufficient chain of custody.

## Verification and development

Install the pinned development tools into the active project environment:

```bash
python -m pip install --require-hashes -r requirements-dev.lock
```

Run the Python quality gates:

```bash
python -m pytest
ruff check .
coverage run -m pytest
coverage report
```

Coverage is configured for branch coverage with an 85% minimum. Some tests
exercise optional local services or end-to-end paths; their prerequisites are
defined by the tests and `pytest.ini`.

Run frontend checks:

```bash
cd frontend
npm ci
npm run typecheck
npm run lint
npm test
npm run build
npm run test:e2e
```

Run the distribution acceptance gate from the repository root:

```bash
scripts/verify-distribution.sh
```

It builds wheel/sdist artifacts, checks their contents, installs the wheel in a
clean environment, runs CLI/import smoke tests outside the checkout, and runs
`pip check`. Avoid documenting fixed test counts: the suite changes with the
codebase.
