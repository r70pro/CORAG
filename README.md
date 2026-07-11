# OLMOCR PDF-to-Markdown Extraction & RAG Analysis Suite

A high-performance, layout-aware PDF OCR pipeline and interactive analysis dashboard built with Gradio and tailored to interface with the `allenai/olmOCR-2-7B-1025-FP8` vision-language model.

The suite features **built-in Docker lifecycle management** to dynamically run the vLLM inference backend, alongside a **fully integrated local RAG (Retrieval-Augmented Generation) analysis pipeline** optimized for large, complex medicolegal files.

---

## Key Features

### 📄 Layout-Aware PDF OCR
- **Real-Time Inference Badge**: Real-time server status badge (`Offline`, `Starting`, or `Ready`) via container health probes.
- **Docker VRAM Control**: Start, stop, or recreation of the inference backend directly from settings. GPU memory (VRAM) is automatically reclaimed on application shutdown.
- **Parallel Batch Processing**: Upload multiple PDFs simultaneously for concurrent processing with unbuffered log streaming.
- **3-Window Viewer**: Symmetrical side-by-side layout displaying the original PDF, raw extracted Markdown text, and its rendered HTML preview with scroll-synchronisation.

### 🧠 Medicolegal RAG Analysis (New)
- **4-Tier Local Service Stack**: Deploys fully persistent, containerized instances of **PostgreSQL 16** (registry/provenance), **Redis 7.2** (cache), **MinIO** (blob archive), and **Qdrant** (vector store).
- **Medicolegal Chunker**: A custom chunking pipeline that respects letter/consultation boundaries, limits chunk sizes to ~800 characters with 100 character overlap, and extracts key clinical metadata:
  - **ISO Dates**: Standardises date variants (e.g., `12.02.2018`, `Aug 27, 2020`) to `YYYY-MM-DD`.
  - **Authors**: Extracts names from signature blocks and letterheads.
  - **Classifications**: Identifies document types (e.g. specialist letter, physiotherapy report, clinical notes) and clinical section types (findings, history, medications, diagnosis, treatment).
  - **Page Mapping**: Maps every text chunk back to its specific original PDF page number.
- **Dynamic Multi-Model Embedding**: Supports changing embedding models directly in the UI. Automatically manages separate Qdrant collections (e.g. `olmocr_documents_all_minilm_l6_v2`) to prevent vector dimension mismatch conflicts.
- **Retrieval Optimization**: Employs cosine similarity matching coupled with metadata filters and Jaccard-based **Maximal Marginal Relevance (MMR)** to ensure diversity in context documents.
- **Analysis Modes**:
  - `💬 Free Q&A`: Streamed natural language chat grounded strictly in source documents.
  - `📅 Timeline Generator`: Auto-extracts dated medical events into a formatted chronological table.
  - `🏥 Injury Summary`: Generates structured injury summaries, treatments, outcomes, and practitioner registries.
  - `🔍 Inconsistency Finder`: Audits the record to expose discrepancies in symptoms, dates, or recommendations between providers.
  - `💊 Medication Tracker`: Tracks prescriptions, dosages, changes, and dates.

---

## Architecture Stack

| Tier | Component | Container | Port | Persistent Directory | Purpose |
|---|---|---|---|---|---|
| **Web UI** | Gradio Dashboard | Host Process | `7860` | — | Document management, batch execution, log viewer, and chat |
| **Inference** | vLLM Engine | `olmocr` | `8000` | `~/.cache/huggingface` | Executes olmOCR OCR model and swaps to analysis LLMs |
| **Registry** | PostgreSQL 16 | `olmocr_postgres` | `5432` | `workspace/pg_data` | Tracks document registry, chunk mappings, metadata and runs |
| **Caching** | Redis 7.2 | `olmocr_redis` | `6379` | `workspace/redis_data` | Caches query answers, token embeddings, and chat history |
| **Storage** | MinIO | `olmocr_minio` | `9000/9001` | `workspace/minio_data` | PDF and Markdown blob archives |
| **Similarity** | Qdrant | `olmocr_qdrant` | `6333/6334` | `workspace/qdrant_storage` | Dense vector database |

---

## Prerequisites

1. **System Tools**: Install `poppler-utils` for PDF rendering:
   ```bash
   # Ubuntu/Debian
   sudo apt-get update && sudo apt-get install -y poppler-utils
   ```
2. **Docker**: Ensure Docker is installed and the current user has permissions to run containers without `sudo`.
3. **NVIDIA Container Toolkit**: Required to pass GPU control to the vLLM containers (`--gpus all`).

---

## Installation & Setup

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

---

## Running the Application

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

## Verification & Testing

Verify that both the core pipeline and the RAG subsystems function correctly by running the test suite:

```bash
# Activate the environment
source ~/olmocr-env/bin/activate

# Execute all tests
python -m unittest discover -v
```

This runs:
- `test_app.py`: Verification of the Docker inference lifecycle, layout builders, reset cleanups, settings manager, and progress parsers.
- `test_rag.py`: Unit tests for the custom medicolegal chunker, PostgreSQL schema registration, MinIO upload pipelines, Redis key cache functions, Qdrant upserts, and LLM prompt compiler.
