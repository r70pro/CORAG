# OLMOCR PDF-to-Markdown Extraction Suite

A high-performance layout-aware PDF OCR pipeline GUI built with Gradio and tailored to interface with the `allenai/olmOCR-2-7B-1025-FP8` vision-language model. 

The application features **built-in Docker lifecycle management**, which dynamically starts, stops, or provisions the underlying vLLM inference container, automatically reclaiming GPU memory (VRAM) when the application shuts down.

---

## Features

- **Inference Server Status Badge**: Shows the server status in real-time (`Offline`, `Starting`, or `Ready`) by checking container state and performing active health probes on the vLLM models API.
- **VRAM Control Settings**: Toggle container execution (`Start` / `Stop`) from settings. Automatic container shutdown hook triggers on exit to release GPU memory.
- **Automated Provisioning**: One-click container provisioning (`Recreate & Run`) with configurable Host Port, HF Token, GPU Utilization, and Model Length parameters.
- **Drag-and-Drop Batch Uploads**: Upload multiple PDFs simultaneously for parallel/concurrent processing.
- **Unbuffered Logs streaming**: Pipes raw stdout/stderr outputs of the pipeline into an interactive console block.
- **Metrics Dashboard**: Monitors page completion status and vLLM queues live.
- **Dual-View Results Viewer**: Side-by-side display of the raw extracted Markdown text and its fully rendered HTML preview.
- **Bulk & Individual Exports**: Download individual Markdown files or download a consolidated ZIP archive of all processed outputs.
- **Settings Persistence**: Remembers server URLs, model configurations, and pipeline variables across restarts.

---

## Prerequisites

1. **System Tools**: You must have `poppler-utils` installed on your host system for PDF page rendering:
   ```bash
   # Ubuntu/Debian
   sudo apt-get update && sudo apt-get install -y poppler-utils
   ```
2. **Docker Permissions**: Ensure Docker is installed and the current user is added to the `docker` group (no `sudo` required for Docker commands).
3. **NVIDIA Container Toolkit**: Required to pass GPU control to the container (`--gpus all` option).

---

## Installation & Setup

1. **Create and Activate Python Environment**:
   ```bash
   python3 -m venv ~/olmocr-env
   source ~/olmocr-env/bin/activate
   ```

2. **Install Core & GUI Dependencies**:
   ```bash
   pip install -U pip
   pip install olmocr
   pip install -r requirements.txt
   ```

---

## Running the Application

1. **Launch the Server**:
   ```bash
   python app.py
   ```

2. **Access the GUI**:
   Open your browser and navigate to:
   ```
   http://localhost:7860/
   ```

3. **Initialize the Inference Backend**:
   - On the sidebar, expand **🐳 Local Inference Server (Docker)**.
   - Enter your Hugging Face token (a default token is pre-populated).
   - If the container does not exist yet, click **🔄 Recreate & Run**. This will download the vLLM container and start model initialization.
   - Once the header status badge updates to green **"Inference Server: Ready"**, you can begin uploading and processing PDFs!
