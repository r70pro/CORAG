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
  - [rag_export.py](file:///home/owner/OLMOCR/rag_export.py) - Chat session export (Markdown, plain text, CSV timeline)
  - [rag_infra_manager.py](file:///home/owner/OLMOCR/rag_infra_manager.py) - RAG services (PG, Redis, MinIO, Qdrant) lifecycle orchestrator
  - [rag_ui.py](file:///home/owner/OLMOCR/rag_ui.py) - Gradio layout and callbacks for the Case Dashboard and RAG Chat panels
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
    - [test_download_models.py](file:///home/owner/OLMOCR/tests/test_download_models.py), [test_e2e_app.py](file:///home/owner/OLMOCR/tests/test_e2e_app.py), [test_external_md_upload.py](file:///home/owner/OLMOCR/tests/test_external_md_upload.py), [test_html_utils_all.py](file:///home/owner/OLMOCR/tests/test_html_utils_all.py)
    - [test_integration_rag.py](file:///home/owner/OLMOCR/tests/test_integration_rag.py), [test_pdf_manager.py](file:///home/owner/OLMOCR/tests/test_pdf_manager.py), [test_pdf_manager_all.py](file:///home/owner/OLMOCR/tests/test_pdf_manager_all.py), [test_pipeline_manager.py](file:///home/owner/OLMOCR/tests/test_pipeline_manager.py)
    - [test_rag.py](file:///home/owner/OLMOCR/tests/test_rag.py), [test_rag_analyzer_all.py](file:///home/owner/OLMOCR/tests/test_rag_analyzer_all.py), [test_rag_cache_all.py](file:///home/owner/OLMOCR/tests/test_rag_cache_all.py), [test_rag_db_errors.py](file:///home/owner/OLMOCR/tests/test_rag_db_errors.py)
    - [test_rag_embedding_all.py](file:///home/owner/OLMOCR/tests/test_rag_embedding_all.py), [test_rag_export.py](file:///home/owner/OLMOCR/tests/test_rag_export.py), [test_rag_extended.py](file:///home/owner/OLMOCR/tests/test_rag_extended.py), [test_rag_infra_manager_all.py](file:///home/owner/OLMOCR/tests/test_rag_infra_manager_all.py)
    - [test_rag_retriever_all.py](file:///home/owner/OLMOCR/tests/test_rag_retriever_all.py), [test_rag_storage_all.py](file:///home/owner/OLMOCR/tests/test_rag_storage_all.py)
    - [test_state.py](file:///home/owner/OLMOCR/tests/test_state.py), [test_ui_callbacks.py](file:///home/owner/OLMOCR/tests/test_ui_callbacks.py)

---

## 🎨 Comprehensive Frontend Design

> For the complete practitioner routine and detailed workflow instructions, see [medicolegal_rag_guide.md](file:///home/owner/OLMOCR/medicolegal_rag_guide.md).

The frontend is a single-page Gradio application ([app.py](file:///home/owner/OLMOCR/app.py)) built around a dark-mode glassmorphism design system with a **5-panel navigation architecture**. A persistent left sidebar provides global navigation and Docker inference controls. The design is optimised for the daily workflow of medicolegal practitioners managing 3–5 separate client cases per day.

### 🗺️ Full-Page UI Layout Map

```mermaid
block-beta
    columns 5

    block:navsidebar:1
        columns 1
        nav["🧭 Navigation Sidebar<br/>📄 PDF Ingestion<br/>🔍 Layout Inspector<br/>📊 Case Dashboard<br/>💬 RAG Processing<br/>🖥️ System Diagnostics<br/>───────────<br/>🐳 Inference Server<br/>⚙️ Pipeline Settings"]
    end

    block:contentarea:4
        columns 4
        paneltitle["Dynamic Panel Title + System Health Badge"]
    end

    block:spacer1:1
        columns 1
        space1[" "]
    end

    block:panel1:4
        columns 4
        p1["Panel 1: PDF Ingestion<br/>Upload + Batch Processing + Monitoring + Log"]
    end

    block:spacer2:1
        columns 1
        space2[" "]
    end

    block:panel2:4
        columns 4
        p2a["📄 Original PDF"]
        p2b["✍️ Raw Markdown"]
        p2c["👁️ Rendered Preview"]
        p2d[" "]
    end

    block:spacer3:1
        columns 1
        space3[" "]
    end

    block:panel3:4
        columns 4
        p3["Panel 3: Case Dashboard<br/>Card Grid · Per-Case Metrics · Delete Case"]
    end

    block:spacer4:1
        columns 1
        space4[" "]
    end

    block:panel4:4
        columns 4
        p4sidebar["RAG Sidebar<br/>🔧 Infrastructure<br/>📦 Indexing<br/>📥 Upload MD<br/>⚙️ Settings<br/>🔍 Filters"]
        p4chat["💬 Chat Interface (1000px)<br/>Active Case Banner<br/>Analysis Mode Selector<br/>🚀 Ask / 🗑️ Clear<br/>📝.md 📄.txt 📊.csv Export<br/>📜 RAG System Log"]
    end

    block:spacer5:1
        columns 1
        space5[" "]
    end

    block:panel5:4
        columns 4
        p5["Panel 5: System Diagnostics<br/>Backing Services Health · GPU/VRAM Metrics · Reset & Cleanup"]
    end
```

---

### 🎨 Design System

The visual identity is defined in [ui_theme.py](file:///home/owner/OLMOCR/ui_theme.py) and applies globally via a Gradio `Base` theme override plus custom CSS injection.

#### Colour Palette

| Token | Value | Usage |
|:---|:---|:---|
| `body_background_fill` | `#090d16` | Deep navy page background |
| `block_background_fill` | `rgba(17, 24, 39, 0.7)` | Glassmorphism panel fill |
| `block_border_color` | `rgba(255, 255, 255, 0.08)` | Subtle frosted glass edges |
| `body_text_color` | `#e2e8f0` | Primary text (slate-200) |
| `body_text_color_subdued` | `#9ca3af` | Secondary/hint text (gray-400) |
| `input_background_fill` | `#1e293b` | Input fields (slate-800) |
| `button_primary` | `linear-gradient(135deg, #6366f1, #3b82f6)` | Indigo → blue gradient for primary actions |
| `button_secondary` | `rgba(30, 41, 59, 0.8)` | Muted panel buttons |
| `border_color_accent` | `rgba(99, 102, 241, 0.5)` | Focus rings and active borders |
| Log console text | `#38bdf8` | Sky-400 on `#020617` for high-contrast code output |
| Badge — idle | `#1e293b` bg / `#94a3b8` text | Neutral state |
| Badge — running | `#1e3a8a` bg / `#60a5fa` text | Pulsing animation (`2s infinite`) |
| Badge — success | `#064e3b` bg / `#34d399` text | Healthy / completed |
| Badge — failed/stopped | `#7f1d1d` bg / `#fca5a5` text | Error or halted |

#### Typography

| Element | Font | Weight | Size |
|:---|:---|:---:|:---|
| Body / UI text | [Outfit](https://fonts.google.com/specimen/Outfit) (Google Fonts) | 300–700 | System default |
| Page title | Outfit | 800 | `2.2rem`, gradient-filled (`#818cf8 → #3b82f6 → #60a5fa`) |
| Log consoles | [JetBrains Mono](https://fonts.google.com/specimen/JetBrains+Mono) | 400 | `0.85rem` |
| Stat card values | Outfit | 700 | `1.8rem` |
| Stat card labels | Outfit | 400 | `0.85rem`, uppercase, `0.05em` letter-spacing |

#### Glassmorphism Panel System

Every content region uses the `.glass-panel` CSS class:
```css
background: rgba(17, 24, 39, 0.7);
border: 1px solid rgba(255, 255, 255, 0.08);
border-radius: 16px;
box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.5);
padding: 20px;
```
This creates the frosted-glass appearance with subtle depth layering throughout the UI.

#### Scrollbar Styling

Custom WebKit scrollbars for premium feel:
- Track: `rgba(255, 255, 255, 0.03)` — nearly invisible
- Thumb: `rgba(255, 255, 255, 0.25)` — subtle visibility
- Thumb hover: `rgba(129, 140, 248, 0.6)` — indigo accent highlight
- Width/height: `8px`

---

### 🏗️ Interface Architecture — 5-Panel Navigation

The application uses a persistent **left navigation sidebar** with 5 content panels. Only one panel is visible at a time. The sidebar also hosts global controls (Inference Server, Pipeline Settings).

#### Global Navigation Sidebar ([app.py:L66-L233](file:///home/owner/OLMOCR/app.py#L66-L233))

| Section | Contents | Key Features |
|:---|:---|:---|
| **Panel Navigation** | 5 radio buttons: 📄 PDF Ingestion, 🔍 Layout Inspector, 📊 Case Dashboard, 💬 RAG Processing, 🖥️ System Diagnostics | `gr.Radio` with visibility toggling via `.change()` callbacks |
| **🐳 Inference Server (Docker)** | HF token, Docker port, GPU memory slider (0.1–1.0), max content length (up to 1M, model-dependent), Start/Stop/Recreate buttons | Creates and manages the `vllm/vllm-openai` Docker container |
| **⚙️ Pipeline Settings** | Model selector, workers (1–64), concurrency (1–2000), image dimension (512–2048px), retries (1–20), guided decoding | 💾 Save Configuration button persists to `settings.json` |
| **Active Role** | Dynamic badge showing current panel context | Updates as user navigates between panels |

#### Panel 1: PDF Ingestion ([app.py](file:///home/owner/OLMOCR/app.py))

| Component | Layout | Purpose |
|:---|:---|:---|
| **📥 Source Documents** | File upload widget (multi-file `.pdf`), 🚀 Start / 🛑 Stop buttons | Drag-and-drop PDF ingestion |
| **📊 Monitoring** | Status badge, progress bar (animated HTML), Completed/Failed counter cards | Real-time batch tracking with stat-cards |
| **📋 Upload Manifest** | HTML table (scrollable, max 200px) | Lists uploaded files with sizes |
| **📁 Per-File Status** | HTML table (scrollable, max 200px) | Per-document processing results |
| **📜 System Output Log** | `gr.Code` (shell syntax, 30 lines, `.log-console`) | Live subprocess stdout/stderr |

#### Panel 2: Layout Inspector ([app.py](file:///home/owner/OLMOCR/app.py))

**Control Bar:**

| Control | Type | Function |
|:---|:---|:---|
| **📄 Select Processed Document** | `gr.Dropdown` | Lists all markdown outputs from the active run |
| **👁️ View Mode** | `gr.Radio` | Toggle between `Page-by-Page` and `Full Document` |
| **Page Navigation** | ⬅️ Prev / `gr.Slider` / Next ➡️ | Page-by-page navigation (1-indexed) |
| **Sync Scroll** | `gr.Checkbox` (`#sync-scroll-checkbox`) | Enables proportional scroll synchronisation across all three panels |
| **Downloads** | `gr.File` (×2) | Individual Markdown file download + ZIP archive of all outputs |

**Three-Panel Viewer** (each panel `scale=1`, height `70vh`):

| Panel | Element ID | Content |
|:---|:---|:---|
| **📄 Original PDF** | `#pdf-scroll-container` | Embedded `<iframe>` rendering of the source PDF via PDFium |
| **✍️ Raw Markdown** | `#raw-scroll-container` | Syntax-highlighted raw Markdown with **📋 Copy** button |
| **👁️ Rendered Preview** | `#preview-scroll-container` | Live HTML render of the Markdown output via `gr.Markdown` |

**Scroll Synchronisation Engine** ([app.py:L1392-L1449](file:///home/owner/OLMOCR/app.py#L1392-L1449)):
- Listens for scroll events across all three containers
- Calculates scroll percentage: `scrollTop / (scrollHeight - clientHeight)`
- Propagates proportional scroll position to sibling panels
- Uses a 100ms debounce with `activeScrollSource` locking to prevent feedback loops

#### Panel 3: Case Dashboard ([rag_ui.py → build_case_dashboard_ui()](file:///home/owner/OLMOCR/rag_ui.py#L762-L845))

| Component | Description |
|:---|:---|
| **Dashboard HTML** | Card grid showing all indexed cases with per-case metrics (document count, chunk count, unique authors, date range, indexed timestamp) |
| **🔄 Refresh Dashboard** | Reloads case data from PostgreSQL |
| **🗑️ Delete Case** | Dropdown selector + delete button — removes all associated data from PostgreSQL, Qdrant, and MinIO |
| **Status** | Markdown output for operation feedback |

#### Panel 4: RAG Processing ([rag_ui.py → build_rag_chat_ui()](file:///home/owner/OLMOCR/rag_ui.py#L848-L1533))

**RAG Sidebar** (`scale=1`, `.sidebar-panel`):

| Accordion | Contents | Key Interactions |
|:---|:---|:---|
| **🔧 RAG Infrastructure** | Status badges for PostgreSQL, Redis, MinIO, Qdrant. ▶️ Start / ⏹️ Stop buttons | Starts/stops all 4 services via `docker compose`. Initialises schemas on start |
| **📦 Document Indexing** | Corpus statistics, 🔄 Refresh Stats, Run selector dropdown, 📥 Index Selected Run, 📥 Index All Runs | Triggers the chunking → embedding → upsert pipeline |
| **📥 Upload External Markdown** | File uploader (.md), Target Case dropdown (new/existing), New Case Name textbox, 📥 Upload & Index button | Bypass ingestion pipeline: upload pre-existing Markdown directly into a case |
| **⚙️ Analysis Settings** | Analysis LLM Server URL, Analysis Model Name (5 models), Retrieval Top-K slider (3–20), Embedding Model dropdown (`BAAI/bge-large-en-v1.5`), 💾 Save | All settings persisted to [settings.json](file:///home/owner/OLMOCR/settings.json) |
| **🔍 Search Filters** | **Active Case** dropdown (case isolation), **Document Type** dropdown (7 types), **Author** dropdown (dynamic), **Date From / Date To** text fields | Filters apply to the next query; case selection auto-populates author and date fields |

**RAG Chat Interface** (`scale=3`, `.glass-panel`):

| Component | Specification | Details |
|:---|:---|:---|
| **Active Case Banner** | `gr.HTML`, dynamic | Displays the currently active case name for visual confirmation of query scope |
| **Analysis Mode** | `gr.Dropdown` (5 modes) | Free Q&A, Timeline Generator, Injury Summary, Inconsistency Finder, Medication Tracker |
| **Chat Window** | `gr.Chatbot`, height `1000px`, copy buttons, `.analysis-chatbot` | Bot: `rgba(30, 41, 59, 0.7)`. User: `rgba(99, 102, 241, 0.15)`. Font: `0.95rem` |
| **Chat Input** | `gr.Textbox`, 2 lines, `scale=4` | Placeholder: *"e.g., What injuries did the patient sustain and when?"* |
| **🚀 Ask Button** | `gr.Button`, primary, `scale=1` | Triggers `user_message_submit()` → `bot_respond()` with streaming |
| **🗑️ Clear Chat** | `gr.Button`, secondary, `size=sm` | Resets chat history |
| **📝 Export .md / 📄 Export .txt / 📊 Export .csv** | `gr.Button` (×3), secondary, `size=sm` | One-click export via [rag_export.py](file:///home/owner/OLMOCR/rag_export.py) to `workspace/exports/` |
| **Keyboard Shortcut Hints** | `gr.HTML` | `Ctrl+Enter` Send, `Ctrl+Shift+N` Clear, `Ctrl+Shift+C` Copy |
| **📜 RAG System Log** | `gr.Code`, shell syntax, 30 lines, `.log-console` | Timestamped backend log |

**Five Analysis Modes** (selectable from the dropdown above the chat):

| Mode | System Prompt Focus | Output Format |
|:---|:---|:---|
| 💬 **Free Q&A** | Answer based strictly on retrieved excerpts; cite `[Source N]`, filename, page, date for every claim; flag gaps | Cited narrative paragraphs |
| 📅 **Timeline Generator** | Extract every dated clinical event in strict chronological order | Markdown table: `Date \| Event \| Provider/Author \| Source` |
| 🏥 **Injury Summary** | Structured report: Patient Details → Mechanism → Injuries → Treatment → Status → Medications → Providers → Outstanding Issues | Numbered heading report with citations |
| 🔍 **Inconsistency Finder** | Cross-reference accounts of the same events; rate severity (Minor/Moderate/Major) | Table: `Issue \| Source A Says \| Source B Says \| Severity` |
| 💊 **Medication Tracker** | Track prescriptions, dose changes, cessations, allergies | Table: `Medication \| Dose/Freq \| Date Started \| Date Stopped \| Prescriber \| Source` |

#### Panel 5: System Diagnostics ([app.py](file:///home/owner/OLMOCR/app.py))

| Component | Description |
|:---|:---|
| **Backing Services Health** | Real-time status cards for vLLM, PostgreSQL, Redis, MinIO, Qdrant — with latency and loaded model display |
| **Hardware Utilization** | GPU/VRAM metrics (nvidia-smi-based) with usage bars |
| **🧹 Reset & Cleanup** | Four checkboxes: obsolete run dirs, Gradio temp, pycache, HF cache. Warning label for HF cache deletion |

---

### ♿ Accessibility & WCAG Compliance

The frontend implements accessibility features via runtime JavaScript ([app.py:L1291-L1451](file:///home/owner/OLMOCR/app.py#L1291-L1451)):

| Feature | Standard | Implementation |
|:---|:---|:---|
| **Focus indicators** | WCAG 2.2 SC 2.4.7 | `2px solid #818cf8` outline + `4px rgba(129, 140, 248, 0.4)` box-shadow on all focusable elements |
| **Dark mode enforcement** | — | `MutationObserver` ensures `dark` class is never removed from `<html>` |
| **Language declaration** | WCAG 2.2 SC 3.1.1 | Sets `lang="en"` on `document.documentElement` |
| **ARIA labels** | WCAG 2.2 SC 1.1.1 | Dynamically applies `aria-label` to all buttons based on emoji/text content |
| **Decorative SVG hiding** | WCAG 2.2 SC 1.1.1 | Sets `aria-hidden="true"` on SVGs without `<title>` elements |
| **Dynamic re-application** | — | `MutationObserver` on `document.body` re-applies ARIA labels when Gradio re-renders components |
| **Keyboard shortcuts** | WCAG 2.2 SC 2.1.1 | `Ctrl+Enter` (submit query), `Ctrl+Shift+N` (clear chat), `Ctrl+Shift+C` (copy last response) |

---

### 🗓️ UX/UI Implementation Status & Roadmap

The following enhancements are documented in the [Medicolegal RAG Guide](file:///home/owner/OLMOCR/medicolegal_rag_guide.md). Features marked ✅ are fully implemented.

```mermaid
mindmap
  root(("Frontend<br/>Roadmap"))
    Case Isolation
      Active Case Selector Dropdown ✅
      Case Dashboard with Card Grid ✅
      Per-Case Delete & Cleanup ✅
    Granular Search Filtering
      Author Filter Dropdown ✅
      Document Type Filter Dropdown ✅
      Date Range Text Fields ✅
    Chat Export
      Markdown Export (.md) ✅
      Plain Text Export (.txt) ✅
      CSV Timeline Export (.csv) ✅
      Direct DOCX Export with Firm Letterhead
      PDF Report with Embedded Citations
    Productivity
      Keyboard Shortcuts ✅
      Saved Query Templates per Case Type
      Batch Query Execution
    Interactive Visualisation
      Clickable Clinical Timeline
      Event → PDF Page Auto-Scroll
      Conflict Markers on Timeline
    Annotation Workspace
      Text Highlighting on Markdown/PDF
      Tag Labels: Disputed / Critical / Key Evidence
      Annotation Export for Legal Submissions
```

| Priority | Feature | Status | Details |
|:---:|:---|:---:|:---|
| 1 | **Active Case Selector** | ✅ Done | Dropdown in 🔍 Search Filters accordion. Applies `run_id_filter` to isolate queries per case |
| 2 | **Interactive Metadata Filters** | ✅ Done | Document Type, Author, Date From/To fields in Search Filters accordion |
| 3 | **Chat Session Export** | ✅ Done | Three export buttons (.md, .txt, .csv) via [rag_export.py](file:///home/owner/OLMOCR/rag_export.py) |
| 4 | **Case Dashboard** | ✅ Done | Dedicated panel with card grid, per-case metrics, and delete functionality |
| 5 | **Keyboard Shortcuts** | ✅ Done | `Ctrl+Enter`, `Ctrl+Shift+N`, `Ctrl+Shift+C` |
| 6 | Interactive Clinical Timeline | 🔲 Planned | Visual timeline with clickable events and conflict markers |
| 7 | Annotation Workspace | 🔲 Planned | Text highlighting with labels (Disputed, Critical, Key Evidence) |
| 8 | Advanced Structured Export (DOCX/PDF) | 🔲 Planned | Word export with firm letterhead; PDF report with embedded citations |

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
   - Run a batch OCR process in the **📄 PDF Ingestion** panel.
   - Click **🔍 Layout Inspector** to review OCR output against the original PDF.
   - Click **📊 Case Dashboard** to view all indexed cases.
   - Click **💬 RAG Processing** to open the analysis chat.
   - Expand **📦 Document Indexing** and click **🔄 Refresh Stats** to load available indexes.
   - Select your completed OCR run and click **📥 Index Selected Run**.
   - Select the case from the **Active Case** dropdown in **🔍 Search Filters**.
   - Type your questions in the Chat input or select a template analysis mode.
   - Export results using the **📝 .md**, **📄 .txt**, or **📊 .csv** export buttons.

---

## 🧪 Verification & Testing

The repository includes a comprehensive testing suite comprising **299 unit and integration tests** validating components, lifecycle states, callbacks, and processing operations.

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
- **`tests/test_rag*.py`**: Validation of the custom medicolegal chunker, PostgreSQL schema registration, MinIO upload pipelines, Redis key cache functions, Qdrant search cosine similarity, LLM prompt compilers, and chat session export.
- **`tests/test_external_md_upload.py`**: Tests for the external Markdown upload and indexing pipeline.
- **`tests/test_download_models.py`**: Tests for the NVFP4 model downloader script.
- **`tests/test_cleanup_manager.py`**: Ensures cache space metrics and reset cleanup routines run safely.
