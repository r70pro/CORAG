#!/usr/bin/env python3
"""
KIRAG Command-Line Interface.

Provides headless access to all KIRAG operations without requiring
Gradio or a browser.  Calls backend functions directly.

Usage:
    python cli.py pipeline runs
    python cli.py docker status
    python cli.py rag query "What injuries did the patient sustain?"
    python cli.py diagnostics health
    python cli.py settings show
"""

from __future__ import annotations

import argparse
import json
import sys

import settings_manager  # noqa: F401


def _print_json(data: dict | list) -> None:
    """Pretty-print a JSON-serialisable object."""
    print(json.dumps(data, indent=2, default=str))


# ── Pipeline commands ─────────────────────────────────────────────────────────


def cmd_pipeline_runs(_args: argparse.Namespace) -> None:
    from settings_manager import get_available_runs

    runs = get_available_runs()
    if not runs:
        print("No completed runs found.")
        return
    for display, run_dir in runs:
        print(f"  {display}")
        print(f"    → {run_dir}")


def cmd_pipeline_status(args: argparse.Namespace) -> None:
    import process_state

    run_id = args.run_id
    with process_state.active_runs_lock:
        run_info = process_state.active_runs.get(run_id)
    if not run_info:
        print(f"Run '{run_id}' not found in active runs.")
        return
    proc = run_info.get("proc")
    if proc and proc.poll() is None:
        print(f"Run '{run_id}': RUNNING")
    elif run_info.get("stop"):
        print(f"Run '{run_id}': STOPPED")
    else:
        print(f"Run '{run_id}': COMPLETED")


def cmd_pipeline_stop(args: argparse.Namespace) -> None:
    from pipeline_manager import stop_processing

    msg = stop_processing(args.run_id)
    print(msg)


# ── Docker commands ───────────────────────────────────────────────────────────


def cmd_docker_status(_args: argparse.Namespace) -> None:
    from docker_manager import get_docker_status_str
    from settings_manager import load_settings

    settings = load_settings()
    port = settings.get("docker_port", 8000)
    status_text, _badge = get_docker_status_str(port)
    print(status_text)


def cmd_docker_start(_args: argparse.Namespace) -> None:
    from docker_manager import start_docker_container

    success, msg = start_docker_container()
    print(msg)
    sys.exit(0 if success else 1)


def cmd_docker_stop(_args: argparse.Namespace) -> None:
    from docker_manager import stop_docker_container

    success, msg = stop_docker_container()
    print(msg)
    sys.exit(0 if success else 1)


def cmd_docker_create(args: argparse.Namespace) -> None:
    from docker_manager import create_docker_container

    success, msg = create_docker_container(
        args.hf_token or "",
        args.port,
        args.model,
        args.gpu_mem,
        args.max_model_len,
        args.tensor_parallel_size,
    )
    print(msg)
    sys.exit(0 if success else 1)


def cmd_docker_shutdown(_args: argparse.Namespace) -> None:
    from docker_manager import shutdown_docker_container

    success, msg = shutdown_docker_container()
    print(msg)
    sys.exit(0 if success else 1)


# ── RAG commands ──────────────────────────────────────────────────────────────


def cmd_rag_query(args: argparse.Namespace) -> None:
    from rag.analyzer import analyze
    from settings_manager import load_settings

    settings = load_settings()
    server_url = args.server_url or settings.get("analysis_server_url", "http://localhost:8000/v1")
    model_name = args.model or settings.get(
        "analysis_model_name", "nvidia/Phi-4-reasoning-plus-NVFP4"
    )
    top_k = args.top_k or settings.get("retrieval_top_k", 15)

    print(f"Querying: {args.query}\n")
    print(f"Mode: {args.mode} | Model: {model_name} | Top-K: {top_k}")
    print("-" * 60)

    try:
        for chunk in analyze(
            query=args.query,
            mode=args.mode,
            server_url=server_url,
            model_name=model_name,
            top_k=top_k,
            run_id_filter=args.case,
            stream=True,
            use_reranker=settings.get("use_reranker", True),
            reranker_model=settings.get("reranker_model", "BAAI/bge-reranker-large"),
            reranker_device=settings.get("reranker_device", "cuda"),
        ):
            print(chunk, end="", flush=True)
        print()
    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        sys.exit(1)


def cmd_rag_index(args: argparse.Namespace) -> None:
    from indexing_service import CorpusIndexingService

    for msg in CorpusIndexingService.index_run(args.run_dir):
        print(msg, end="")
    print()


def cmd_rag_index_all(_args: argparse.Namespace) -> None:
    from indexing_service import CorpusIndexingService

    for msg in CorpusIndexingService.index_all_runs():
        print(msg, end="")
    print()


def cmd_rag_stats(_args: argparse.Namespace) -> None:
    try:
        from rag.db import get_corpus_stats
        from rag.embedding import get_collection_info

        db_stats = get_corpus_stats()
        qdrant_info = get_collection_info()
        _print_json({**db_stats, "vectors_count": qdrant_info.get("points_count", 0)})
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


def cmd_rag_infra_start(_args: argparse.Namespace) -> None:
    from rag_infra_manager import start_and_init_rag

    success, msg = start_and_init_rag()
    print(msg)
    sys.exit(0 if success else 1)


def cmd_rag_infra_stop(_args: argparse.Namespace) -> None:
    from rag_infra_manager import stop_rag_infrastructure

    success, msg = stop_rag_infrastructure()
    print(msg)
    sys.exit(0 if success else 1)


def cmd_rag_infra_status(_args: argparse.Namespace) -> None:
    from rag_infra_manager import get_rag_service_status

    _print_json(get_rag_service_status())


# ── Diagnostics commands ──────────────────────────────────────────────────────


def cmd_diagnostics_health(_args: argparse.Namespace) -> None:
    from settings_manager import load_settings
    from system_diagnostics import check_backing_services_data

    settings = load_settings()
    port = settings.get("docker_port", 8000)
    data = check_backing_services_data({}, vllm_port=port)

    overall = "HEALTHY" if data["all_healthy"] else "DEGRADED"
    print(f"Overall: {overall}\n")

    for name, info in data["services"].items():
        status = "✓ UP" if info["is_up"] else "✗ DOWN"
        latency = f"{info['latency']:.1f}ms" if info["is_up"] else "—"
        extra = f" ({info['extra_info']})" if info.get("extra_info") else ""
        print(f"  {name:>10s}: {status}  {latency}{extra}")


def cmd_diagnostics_gpu(_args: argparse.Namespace) -> None:
    from system_diagnostics import get_gpu_metrics_data

    data = get_gpu_metrics_data()
    if not data.get("cuda_available"):
        print("No CUDA GPU detected.")
        return
    print(f"GPU: {data['gpu_name']}")
    print(f"VRAM: {data['vram_used']:.0f} / {data['vram_total']:.0f} MB ({data['vram_pct']:.1f}%)")
    print(f"Free: {data['vram_free']:.0f} MB | Reclaimable: {data['vram_reclaimable']:.0f} MB")
    if data.get("processes"):
        print(f"\nGPU Processes ({len(data['processes'])}):")
        for p in data["processes"]:
            print(f"  PID {p['pid']:>6d}  {p['vram']:>6d} MB  {p['display_name']}")


def cmd_diagnostics_report(_args: argparse.Namespace) -> None:
    from settings_manager import load_settings
    from system_diagnostics import generate_diagnostic_report_file

    settings = load_settings()
    port = settings.get("docker_port", 8000)
    path = generate_diagnostic_report_file(port)
    print(f"Report saved: {path}")


# ── Settings commands ─────────────────────────────────────────────────────────


def cmd_settings_show(_args: argparse.Namespace) -> None:
    from settings_manager import load_settings

    settings = load_settings()
    if settings.get("hf_token"):
        settings["hf_token"] = "********"
    _print_json(settings)


def cmd_settings_set(args: argparse.Namespace) -> None:
    from settings_manager import load_settings, save_settings

    settings = load_settings()
    key = args.key
    value = args.value

    if key not in settings:
        print(f"Warning: '{key}' is not a known setting. Adding it anyway.")

    # Auto-cast to appropriate type based on existing value
    existing = settings.get(key)
    if isinstance(existing, bool):
        value = value.lower() in ("true", "1", "yes")
    elif isinstance(existing, int):
        value = int(value)
    elif isinstance(existing, float):
        value = float(value)

    settings[key] = value
    result = save_settings(settings)
    print(result)


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="kirag",
        description="KIRAG — Medicolegal RAG Workstation CLI",
    )
    subparsers = parser.add_subparsers(dest="command", help="Command group")

    # ── pipeline ──
    pipeline_parser = subparsers.add_parser("pipeline", help="OCR pipeline operations")
    pipeline_sub = pipeline_parser.add_subparsers(dest="subcommand")

    pipeline_sub.add_parser("runs", help="List completed OCR runs")

    p_status = pipeline_sub.add_parser("status", help="Check run status")
    p_status.add_argument("run_id", help="Run ID to check")

    p_stop = pipeline_sub.add_parser("stop", help="Stop a running pipeline")
    p_stop.add_argument("run_id", help="Run ID to stop")

    # ── docker ──
    docker_parser = subparsers.add_parser("docker", help="vLLM container management")
    docker_sub = docker_parser.add_subparsers(dest="subcommand")

    docker_sub.add_parser("status", help="Container status")
    docker_sub.add_parser("start", help="Start container")
    docker_sub.add_parser("stop", help="Stop container")
    docker_sub.add_parser("shutdown", help="Shutdown and remove container")

    d_create = docker_sub.add_parser("create", help="Create/recreate container")
    d_create.add_argument("--model", default="allenai/olmOCR-2-7B-1025-FP8")
    d_create.add_argument("--port", type=int, default=8000)
    d_create.add_argument("--gpu-mem", type=float, default=0.8)
    d_create.add_argument("--max-model-len", type=int, default=15360)
    d_create.add_argument("--tensor-parallel-size", "-tp", type=int, default=1)
    d_create.add_argument("--hf-token", default=None)

    # ── rag ──
    rag_parser = subparsers.add_parser("rag", help="RAG operations")
    rag_sub = rag_parser.add_subparsers(dest="subcommand")

    r_query = rag_sub.add_parser("query", help="Query the RAG system")
    r_query.add_argument("query", help="Natural language query")
    r_query.add_argument("--mode", default="free_qa", help="Analysis mode")
    r_query.add_argument("--case", default=None, help="Case/run ID filter")
    r_query.add_argument("--top-k", type=int, default=None)
    r_query.add_argument("--model", default=None)
    r_query.add_argument("--server-url", default=None)

    r_index = rag_sub.add_parser("index", help="Index a specific run")
    r_index.add_argument("run_dir", help="Path to run directory")

    rag_sub.add_parser("index-all", help="Index all available runs")
    rag_sub.add_parser("stats", help="Show corpus statistics")

    # RAG infrastructure sub-subcommands
    r_infra = rag_sub.add_parser("infra", help="RAG infrastructure management")
    r_infra_sub = r_infra.add_subparsers(dest="infra_cmd")
    r_infra_sub.add_parser("start", help="Start RAG infrastructure")
    r_infra_sub.add_parser("stop", help="Stop RAG infrastructure")
    r_infra_sub.add_parser("status", help="Infrastructure status")

    # ── diagnostics ──
    diag_parser = subparsers.add_parser("diagnostics", help="System diagnostics")
    diag_sub = diag_parser.add_subparsers(dest="subcommand")

    diag_sub.add_parser("health", help="Full health check")
    diag_sub.add_parser("gpu", help="GPU metrics")
    diag_sub.add_parser("report", help="Generate diagnostic report")

    # ── settings ──
    settings_parser = subparsers.add_parser("settings", help="Configuration management")
    settings_sub = settings_parser.add_subparsers(dest="subcommand")

    settings_sub.add_parser("show", help="Show current settings")

    s_set = settings_sub.add_parser("set", help="Update a setting")
    s_set.add_argument("key", help="Setting key")
    s_set.add_argument("value", help="New value")

    args = parser.parse_args()

    # ── Dispatch ──
    dispatch = {
        ("pipeline", "runs"): cmd_pipeline_runs,
        ("pipeline", "status"): cmd_pipeline_status,
        ("pipeline", "stop"): cmd_pipeline_stop,
        ("docker", "status"): cmd_docker_status,
        ("docker", "start"): cmd_docker_start,
        ("docker", "stop"): cmd_docker_stop,
        ("docker", "create"): cmd_docker_create,
        ("docker", "shutdown"): cmd_docker_shutdown,
        ("rag", "query"): cmd_rag_query,
        ("rag", "index"): cmd_rag_index,
        ("rag", "index-all"): cmd_rag_index_all,
        ("rag", "stats"): cmd_rag_stats,
        ("diagnostics", "health"): cmd_diagnostics_health,
        ("diagnostics", "gpu"): cmd_diagnostics_gpu,
        ("diagnostics", "report"): cmd_diagnostics_report,
        ("settings", "show"): cmd_settings_show,
        ("settings", "set"): cmd_settings_set,
    }

    key = (args.command, getattr(args, "subcommand", None))

    # Handle RAG infra sub-subcommands
    if args.command == "rag" and getattr(args, "subcommand", None) == "infra":
        infra_dispatch = {
            "start": cmd_rag_infra_start,
            "stop": cmd_rag_infra_stop,
            "status": cmd_rag_infra_status,
        }
        infra_cmd = getattr(args, "infra_cmd", None)
        if infra_cmd in infra_dispatch:
            infra_dispatch[infra_cmd](args)
            return
        rag_parser.print_help()
        return

    handler = dispatch.get(key)
    if handler:
        handler(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
