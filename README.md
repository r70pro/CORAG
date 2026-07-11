# OLMOCR PDF-to-Markdown Extraction & RAG Analysis Suite

A high-performance, layout-aware PDF OCR pipeline and interactive analysis dashboard built with Gradio and tailored to interface with multiple vision-language and reasoning models (including [nvidia/Phi-4-reasoning-plus-NVFP4](https://huggingface.co/nvidia/Phi-4-reasoning-plus-NVFP4) by default, and [allenai/olmOCR-2-7B-1025-FP8](https://huggingface.co/allenai/olmOCR-2-7B-1025-FP8)).

The suite features **built-in Docker lifecycle management** to dynamically run the vLLM inference backend, alongside a **fully integrated local RAG (Retrieval-Augmented Generation) analysis pipeline** optimized for large, complex medicolegal files.

---

## 📁 Repository Directory Structure

- OLMOCR/
  - [app.py](file:///home/owner/OLMOCR/app.py) - Main Gradio application entry point
  - [cleanup_manager.py](file:///home/owner/OLMOCR/cleanup_manager.py) - Disk cache and run file cleanup manager
  - [conftest.py](file:///home/owner/OLMOCR/conftest.py) - Pytest configuration redirecting Hugging Face cache
  - [docker-compose.rag.yml](file:///home/owner/OLMOCR/docker-compose.rag.yml) - Orchestration file for RAG local database/store stack
  - [docker_manager.py](file:///home/owner/OLMOCR/docker_manager.py) - Local vLLM Inference container lifecycle manager
  - [download_models.py](file:///home/owner/OLMOCR/download_models.py) - Downloader script for quantized NVFP4 models from Hugging Face
  - [html_utils.py](file:///home/owner/OLMOCR/html_utils.py) - HTML generators for progress bars, manifests, and tables
  - [pdf_manager.py](file:///home/owner/OLMOCR/pdf_manager.py) - PDF rendering (PDFium), page mapping, and zip operations
  - [pipeline_manager.py](file:///home/owner/OLMOCR/pipeline_manager.py) - Batch OCR pipeline engine & process wrapper
  - [pytest.ini](file:///home/owner/OLMOCR/pytest.ini) - Pytest configuration file to ignore RAG database directories
  - [rag_infra_manager.py](file:///home/owner/OLMOCR/rag_infra_manager.py) - RAG services (PG, Redis, MinIO, Qdrant) lifecycle orchestrator
  - [rag_ui.py](file:///home/owner/OLMOCR/rag_ui.py) - Gradio layout and callbacks for the RAG Analysis tab
  - [requirements.txt](file:///home/owner/OLMOCR/requirements.txt) - Python packages and third-party dependencies
  - [settings.json](file:///home/owner/OLMOCR/settings.json) - Persistent user configuration settings
  - [settings_manager.py](file:///home/owner/OLMOCR/settings_manager.py) - Loading, saving, and validation utility for configurations
  - [state.py](file:///home/owner/OLMOCR/state.py) - Thread-safe global active runs and state variables
  - [ui_theme.py](file:///home/owner/OLMOCR/ui_theme.py) - Styling parameters (custom CSS and dark mode theme)
  - rag/ - Local RAG Core Module:
    - [__init__.py](file:///home/owner/OLMOCR/rag/__init__.py) - Module initialization
    - [analyzer.py](file:///home/owner/OLMOCR/rag/analyzer.py) - Chat compiler, prompt templates, and streaming LLM client
    - [cache.py](file:///home/owner/OLMOCR/rag/cache.py) - Redis client for caching queries, embeddings, and chat history
    - [chunker.py](file:///home/owner/OLMOCR/rag/chunker.py) - Medicolegal-aware chunker, metadata (dates, authors) extractor
    - [db.py](file:///home/owner/OLMOCR/rag/db.py) - PostgreSQL client for run history, documents, and chunk metadata
    - [embedding.py](file:///home/owner/OLMOCR/rag/embedding.py) - Sentence-Transformer client and Qdrant collection manager
    - [retriever.py](file:///home/owner/OLMOCR/rag/retriever.py) - Dense vector retriever with cosine search and Jaccard-based MMR
    - [storage.py](file:///home/owner/OLMOCR/rag/storage.py) - MinIO client for PDF and Markdown blob archives
  - tests/ - Extensive unit and integration test suite:
    - [test_app.py](file:///home/owner/OLMOCR/tests/test_app.py), [test_app_callbacks.py](file:///home/owner/OLMOCR/tests/test_app_callbacks.py), [test_cleanup_manager.py](file:///home/owner/OLMOCR/tests/test_cleanup_manager.py), [test_docker_manager.py](file:///home/owner/OLMOCR/tests/test_docker_manager.py)
    - [test_e2e_app.py](file:///home/owner/OLMOCR/tests/test_e2e_app.py), [test_html_utils_all.py](file:///home/owner/OLMOCR/tests/test_html_utils_all.py), [test_integration_rag.py](file:///home/owner/OLMOCR/tests/test_integration_rag.py)
    - [test_pdf_manager.py](file:///home/owner/OLMOCR/tests/test_pdf_manager.py), [test_pdf_manager_all.py](file:///home/owner/OLMOCR/tests/test_pdf_manager_all.py), [test_pipeline_manager.py](file:///home/owner/OLMOCR/tests/test_pipeline_manager.py)
    - [test_rag.py](file:///home/owner/OLMOCR/tests/test_rag.py), [test_rag_analyzer_all.py](file:///home/owner/OLMOCR/tests/test_rag_analyzer_all.py), [test_rag_cache_all.py](file:///home/owner/OLMOCR/tests/test_rag_cache_all.py), [test_rag_db_errors.py](file:///home/owner/OLMOCR/tests/test_rag_db_errors.py)
    - [test_rag_embedding_all.py](file:///home/owner/OLMOCR/tests/test_rag_embedding_all.py), [test_rag_extended.py](file:///home/owner/OLMOCR/tests/test_rag_extended.py), [test_rag_infra_manager_all.py](file:///home/owner/OLMOCR/tests/test_rag_infra_manager_all.py)
    - [test_rag_retriever_all.py](file:///home/owner/OLMOCR/tests/test_rag_retriever_all.py), [test_rag_storage_all.py](file:///home/owner/OLMOCR/tests/test_rag_storage_all.py)
    - [test_state.py](file:///home/owner/OLMOCR/tests/test_state.py), [test_ui_callbacks.py](file:///home/owner/OLMOCR/tests/test_ui_callbacks.py)

---

## 🚀 Key Features

### 📄 Layout-Aware PDF OCR
- **Flexible Model Selection**: Dropdown model selector supporting `nvidia/Phi-4-reasoning-plus-NVFP4` (default), `allenai/olmOCR-2-7B-1025-FP8`, `nvidia/Qwen3.6-35B-A3B-NVFP4`, `nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4`, and `nvidia/Llama-3.3-70B-Instruct-NVFP4`.
- **Real-Time Inference Badge**: Monitor container health in real-time via container status indicators (`Offline`, `Starting`, or `Ready`).
- **Docker VRAM Control**: Create, start, stop, or recreate the inference backend with parameterized GPU memory allocation and model length configuration directly from the UI, running under the `vllm/vllm-openai:cu130-nightly` image.
- **Parallel Batch Processing**: Concurrently processes pages of multiple PDFs using `olmocr.pipeline` with unbuffered logging and granular progress cards. The pipeline runs within the active environment utilizing `sys.executable`.
- **Interactive 3-Window Viewer**: Symmetrical side-by-side layout displaying the original PDF, raw extracted Markdown text, and its rendered HTML preview with scroll-synchronization.

### 🧠 Medicolegal RAG Analysis
- **4-Tier Local Service Stack**: Deploys fully persistent, containerized instances of PostgreSQL (registry/provenance), Redis (cache), MinIO (blob archive), and Qdrant (vector store).
- **Medicolegal Chunker**: A custom chunking pipeline that respects letter/consultation boundaries, limits chunk sizes to ~800 characters with 100 character overlap, and extracts key clinical metadata:
  - **ISO Dates**: Standardizes date variants (e.g., `12.02.2018`, `Aug 27, 2020`) to `YYYY-MM-DD`.
  - **Authors**: Extracts names from signature blocks, letterheads, and clinical logs.
  - **Classifications**: Identifies document types (e.g. specialist letter, physiotherapy report, clinical notes) and clinical section types (findings, history, medications, diagnosis, treatment).
  - **Page Mapping**: Maps text chunks back to their specific original PDF page number.
- **Dynamic Multi-Model Embedding**: Change embedding models directly in the UI. Vector dimensions and collection conflicts are automatically handled by segregating Qdrant collections.
- **Retrieval Optimization**: Cosine similarity matching coupled with metadata filters and Jaccard-based **Maximal Marginal Relevance (MMR)** to ensure diversity in context documents.
- **Enhanced UI Layout**: Relocated RAG System Log console directly under the Chat question field, showing backend indexing, retrieval, and LLM streaming status in real-time, combined with a double-height (`1000px`) Chatbot window for enhanced review depth.
- **Reasoning Model Support**: Smart parameters for reasoning-focused models like `Phi-4-reasoning-plus`, such as temperature correction (to 0.7 if 0.1 is selected to avoid failure in generation paths) and repetition penalty (1.05).
- **Equivalent Model Normalization**: Resolves and maps identical models (e.g. `microsoft/Phi-4-reasoning-plus` and `nvidia/Phi-4-reasoning-plus-NVFP4`) to prevent false warnings or redundant fallback checks.
- **Analysis Modes**:
  - `💬 Free Q&A`: Streamed natural language chat grounded strictly in source documents.
  - `📅 Timeline Generator`: Auto-extracts dated medical events into a formatted chronological table.
  - `🏥 Injury Summary`: Generates structured injury summaries, treatments, outcomes, and practitioner registries.
  - `🔍 Inconsistency Finder`: Audits the record to expose discrepancies in symptoms, dates, or recommendations between providers.
  - `💊 Medication Tracker`: Tracks prescriptions, dosages, changes, and dates.

---

## 🛠️ Architecture Stack

| Tier | Component | Container | Port | Persistent Directory | Purpose |
|---|---|---|---|---|---|
| **Web UI** | Gradio Dashboard | Host Process | `7860` | — | Document management, batch execution, log viewer, and chat |
| **Inference** | vLLM Engine | `olmocr` | `8000` | `~/.cache/huggingface` | Executes olmOCR OCR model and swaps to analysis LLMs |
| **Local Cache** | Hugging Face Cache | Host Process | — | `workspace/huggingface` | Redirected HF cache for sentence-transformers and test suite |
| **Registry** | PostgreSQL 16 | `olmocr_postgres` | `5432` | `workspace/pg_data` | Tracks document registry, chunk mappings, metadata and runs |
| **Caching** | Redis 7.2 | `olmocr_redis` | `6379` | `workspace/redis_data` | Caches query answers, token embeddings, and chat history |
| **Storage** | MinIO | `olmocr_minio` | `9000/9001` | `workspace/minio_data` | PDF and Markdown blob archives |
| **Similarity** | Qdrant | `olmocr_qdrant` | `6333/6334` | `workspace/qdrant_storage` | Dense vector database |

---

## 📦 Dependencies

The application relies on the following key dependencies:
- **`gradio`**: Interactive web interface.
- **`psycopg2-binary`**: PostgreSQL database connector for run and document schemas.
- **`redis`**: Caching layer for responses, embeddings, and chat session variables.
- **`minio`**: Object storage access for archiving source PDFs and markdown content.
- **`qdrant-client`**: Client for dense vector indices (supports range `>=1.9.0,<1.11.0`).
- **`sentence-transformers`**: Generates high-dimensional vector representations.
- **`tiktoken`**: Token count measurement.
- **`pypdf` / `pypdfium2`**: PDF validation and high-performance page rendering.
- **`httpx`**: Asynchronous HTTP client for communicating with the vLLM server and equivalent checks.
- **`numpy`**: Multidimensional array operations (used in embeddings/retrievals).
- **`pytest` / `coverage`**: Core framework and statistics gathering for the comprehensive testing suite.

---

## 📋 Prerequisites

1. **System Tools**: Install `poppler-utils` for PDF rendering:
   ```bash
   # Ubuntu/Debian
   sudo apt-get update && sudo apt-get install -y poppler-utils
   ```
2. **Docker**: Ensure Docker is installed and the current user has permissions to run containers without `sudo`.
3. **NVIDIA Container Toolkit**: Required to pass GPU control to the vLLM containers (`--gpus all`).

---

## ⚙️ Installation & Setup

1. **Clone the Repository & Navigate**:
   ```bash
   cd OLMOCR
   ```

2. **Create and Activate Python Environment**:
   ```bash
   python3 -m venv ~/olmocr-env
   source ~/olmocr-env/bin/activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -U pip
   pip install olmocr
   pip install -r requirements.txt
   ```

4. **Pre-download Models (Optional)**:
   Pre-fetch the required NVFP4 models using the downloader script to avoid download latency during application start:
   ```bash
   python download_models.py
   ```

---

## 🚦 Running the Application

1. **Bring Up the RAG Services Stack**:
   Start the background databases and engines using Docker Compose:
   ```bash
   docker compose -f docker-compose.rag.yml up -d
   ```
   *(Alternatively, you can start/stop the RAG services directly from the Gradio UI inside the **🔧 RAG Infrastructure** panel).*

2. **Launch the Dashboard**:
   ```bash
   python app.py
   ```

3. **Access the GUI**:
   Open your browser and navigate to `http://localhost:7860/`.

4. **Analyze Documents**:
   - Run a batch OCR process in the center panel.
   - Go to the **🧠 Document Analysis (RAG)** tab at the bottom of the page.
   - Click **🔄 Refresh Stats** to load available indexes.
   - Select your completed OCR run and click **📥 Index Selected Run**.
   - Type your questions in the Chat input block or select a template analysis mode from the accordion settings.

---

## 🧪 Verification & Testing

The repository includes a comprehensive testing suite comprising **250+ unit and integration tests** (specifically 258 tests) validating components, lifecycle states, callbacks, and processing operations.

To run the test suite, ensure the virtual environment is active, then execute:

```bash
# Execute all unit and integration tests
pytest
```

> [!NOTE]
> The project includes a `pytest.ini` configuration file that automatically ignores the `workspace/` directory. This prevents the testing engine from attempting to scan Docker-mounted database directories (PostgreSQL, Redis), which would otherwise cause a permission error. Additionally, `conftest.py` ensures the Hugging Face cache path `HF_HOME` is dynamically isolated to `workspace/huggingface`.

### Tested Components:
- **`tests/test_app*.py` / `tests/test_ui*.py`**: Verification of Docker inference lifecycles, progress bar components, settings manager, and Gradio panel callbacks.
- **`tests/test_pipeline*.py` / `tests/test_pdf*.py`**: Unit tests for batch pipeline execution, PDF segmentation, file zip packaging, and PDFium image rendering.
- **`tests/test_rag*.py`**: Validation of the custom medicolegal chunker, PostgreSQL schema registration, MinIO upload pipelines, Redis key cache functions, Qdrant search cosine similarity, and LLM prompt compilers.
- **`tests/test_cleanup_manager.py`**: Ensures cache space metrics and reset cleanup routines run safely.
