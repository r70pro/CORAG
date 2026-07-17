from typing import Dict, List, Set, Optional, Any

def make_progress_bar_html(completed: int, total: int, elapsed_secs: float = 0) -> str:
    pct = int((completed / total) * 100) if total > 0 else 0
    # ETA calculation
    eta_str = ""
    if completed > 0 and elapsed_secs > 0 and completed < total:
        rate = completed / elapsed_secs
        remaining = (total - completed) / rate
        if remaining < 60:
            eta_str = f"{int(remaining)}s remaining"
        elif remaining < 3600:
            eta_str = f"{int(remaining // 60)}m {int(remaining % 60)}s remaining"
        else:
            eta_str = f"{int(remaining // 3600)}h {int((remaining % 3600) // 60)}m remaining"
    elif completed >= total and total > 0:
        eta_str = "Complete"
    
    elapsed_str = ""
    if elapsed_secs > 0:
        if elapsed_secs < 60:
            elapsed_str = f"{int(elapsed_secs)}s elapsed"
        elif elapsed_secs < 3600:
            elapsed_str = f"{int(elapsed_secs // 60)}m {int(elapsed_secs % 60)}s elapsed"
        else:
            elapsed_str = f"{int(elapsed_secs // 3600)}h {int((elapsed_secs % 3600) // 60)}m elapsed"
    
    time_info = ""
    if elapsed_str and eta_str:
        time_info = f"<div style='display:flex; justify-content:space-between; font-size:0.8rem; color:#94a3b8; margin-top:4px;'><span>{elapsed_str}</span><span>{eta_str}</span></div>"
    elif elapsed_str:
        time_info = f"<div style='font-size:0.8rem; color:#94a3b8; margin-top:4px;'>{elapsed_str}</div>"
    
    return f"""<div style='width:100%;'>
        <div style='display:flex; justify-content:space-between; margin-bottom:4px;'>
            <span style='font-size:0.9rem; color:#e2e8f0; font-weight:600;'>{completed}/{total} Pages</span>
            <span style='font-size:0.9rem; color:#818cf8; font-weight:600;'>{pct}%</span>
        </div>
        <div style='width:100%; background:#1e293b; border-radius:8px; height:12px; overflow:hidden;'>
            <div style='width:{pct}%; height:100%; background:linear-gradient(90deg, #6366f1, #3b82f6); border-radius:8px; transition:width 0.4s ease;'></div>
        </div>
        {time_info}
    </div>"""


def make_file_status_html(file_mapping: Dict[int, str], file_page_counts: Dict[int, int], completed_files_set: Set[int], failed_files_set: Optional[Set[int]] = None) -> str:
    if failed_files_set is None:
        failed_files_set = set()
    
    rows = ""
    for idx in sorted(file_mapping.keys()):
        name = file_mapping[idx]
        pages = file_page_counts.get(idx, "?")
        if idx in failed_files_set:
            status = "<span style='color:#fca5a5;'>✗ Failed</span>"
        elif idx in completed_files_set:
            status = "<span style='color:#34d399;'>✓ Done</span>"
        else:
            status = "<span style='color:#94a3b8;'>⏳ Pending</span>"
        rows += f"<tr style='border-bottom:1px solid rgba(255,255,255,0.05);'><td style='padding:6px 10px; color:#e2e8f0; font-size:0.85rem;'>{name}</td><td style='padding:6px 10px; color:#94a3b8; text-align:center; font-size:0.85rem;'>{pages}</td><td style='padding:6px 10px; text-align:center; font-size:0.85rem;'>{status}</td></tr>"
    
    return f"""<table style='width:100%; border-collapse:collapse;'>
        <thead><tr style='border-bottom:1px solid rgba(255,255,255,0.1);'>
            <th style='padding:6px 10px; color:#94a3b8; text-align:left; font-size:0.8rem; text-transform:uppercase; letter-spacing:0.05em;'>File</th>
            <th style='padding:6px 10px; color:#94a3b8; text-align:center; font-size:0.8rem; text-transform:uppercase; letter-spacing:0.05em;'>Pages</th>
            <th style='padding:6px 10px; color:#94a3b8; text-align:center; font-size:0.8rem; text-transform:uppercase; letter-spacing:0.05em;'>Status</th>
        </tr></thead>
        <tbody>{rows}</tbody>
    </table>"""


def make_upload_manifest_html(file_mapping: Dict[int, str], file_page_counts: Dict[int, int], file_sizes: Dict[int, int]) -> str:
    rows = ""
    total_pages = 0
    total_size = 0
    for idx in sorted(file_mapping.keys()):
        name = file_mapping[idx]
        pages = file_page_counts.get(idx, "?")
        size_bytes = file_sizes.get(idx, 0)
        if isinstance(pages, int):
            total_pages += pages
        total_size += size_bytes
        
        if size_bytes < 1024:
            size_str = f"{size_bytes} B"
        elif size_bytes < 1024 * 1024:
            size_str = f"{size_bytes / 1024:.1f} KB"
        else:
            size_str = f"{size_bytes / (1024 * 1024):.1f} MB"
        
        rows += f"<tr style='border-bottom:1px solid rgba(255,255,255,0.05);'><td style='padding:5px 10px; color:#e2e8f0; font-size:0.85rem;'>{name}</td><td style='padding:5px 10px; color:#94a3b8; text-align:center; font-size:0.85rem;'>{pages}</td><td style='padding:5px 10px; color:#94a3b8; text-align:center; font-size:0.85rem;'>{size_str}</td></tr>"
    
    if total_size < 1024 * 1024:
        total_size_str = f"{total_size / 1024:.1f} KB"
    else:
        total_size_str = f"{total_size / (1024 * 1024):.1f} MB"
    
    rows += f"<tr style='border-top:1px solid rgba(255,255,255,0.1);'><td style='padding:5px 10px; color:#818cf8; font-size:0.85rem; font-weight:600;'>Total ({len(file_mapping)} files)</td><td style='padding:5px 10px; color:#818cf8; text-align:center; font-size:0.85rem; font-weight:600;'>{total_pages}</td><td style='padding:5px 10px; color:#818cf8; text-align:center; font-size:0.85rem; font-weight:600;'>{total_size_str}</td></tr>"
    
    return f"""<table style='width:100%; border-collapse:collapse;'>
        <thead><tr style='border-bottom:1px solid rgba(255,255,255,0.1);'>
            <th style='padding:5px 10px; color:#94a3b8; text-align:left; font-size:0.8rem; text-transform:uppercase; letter-spacing:0.05em;'>File</th>
            <th style='padding:5px 10px; color:#94a3b8; text-align:center; font-size:0.8rem; text-transform:uppercase; letter-spacing:0.05em;'>Pages</th>
            <th style='padding:5px 10px; color:#94a3b8; text-align:center; font-size:0.8rem; text-transform:uppercase; letter-spacing:0.05em;'>Size</th>
        </tr></thead>
        <tbody>{rows}</tbody>
    </table>"""


def get_simulated_sparkline(is_up: bool = True, latency_history: Optional[List[float]] = None) -> str:
    import random
    if not is_up:
        return """<svg class='sparkline-svg sparkline-red' viewBox='0 0 60 20'><polyline points='0,10 10,10 20,10 30,10 40,10 50,10 60,10'/></svg>"""
    
    if not latency_history:
        points = [random.randint(5, 15) for _ in range(8)]
    else:
        min_val = min(latency_history) if latency_history else 0
        max_val = max(latency_history) if latency_history else 1
        val_range = max_val - min_val if max_val != min_val else 1
        points = [20 - int((v - min_val) / val_range * 15 + 2) for v in latency_history]
        
    points_str = " ".join(f"{i*8},{v}" for i, v in enumerate(points))
    return f"""<svg class='sparkline-svg' viewBox='0 0 60 20'><polyline points='{points_str}'/></svg>"""


def make_backing_services_html(data: Dict[str, Any]) -> str:
    html_parts = []
    service_names = {
        "postgres": "PostgreSQL",
        "redis": "Redis",
        "minio": "MinIO",
        "qdrant": "Qdrant",
        "vllm": "vLLM"
    }
    
    for s, info in data["services"].items():
        is_up = info["is_up"]
        latency = info["latency"]
        extra_info = info["extra_info"]
        latency_history = info["latency_history"]
        
        model_badge = ""
        extra_badge = ""
        if is_up:
            sparkline = get_simulated_sparkline(True, latency_history)
            badge = "<span class='badge-success' style='font-size:0.75rem; padding: 2px 6px;'>UP</span>"
            latency_str = f"{latency:.1f} ms"
            if s == "vllm":
                if extra_info:
                    model_badge = f"<span style='font-size:0.75rem; color:#94a3b8; font-weight: normal; font-family: monospace; background: rgba(255, 255, 255, 0.05); padding: 2px 6px; border-radius: 4px; margin-left: 8px;'>{extra_info}</span>"
        else:
            sparkline = get_simulated_sparkline(False)
            badge = "<span class='badge-stopped' style='font-size:0.75rem; padding: 2px 6px;'>DOWN</span>"
            latency_str = "N/A"
            if s == "vllm" and data["vllm_progress"]:
                progress = data["vllm_progress"]
                badge = "<span class='badge-running' style='font-size:0.75rem; padding: 2px 6px; animation: pulse 2s infinite;'>LOADING</span>"
                latency_str = f"Progress: {progress['pct']}%"
                extra_badge = f"""
                <div style='margin-top: 10px; width: 100%; background: rgba(59, 130, 246, 0.08); border: 1px solid rgba(59, 130, 246, 0.2); border-radius: 8px; padding: 10px; box-sizing: border-box;'>
                    <div style='display:flex; justify-content:space-between; font-size:0.75rem; font-weight:600; color:#60a5fa; margin-bottom: 4px;'>
                        <span>Model Loading...</span>
                        <span>{progress['shards_loaded']}/{progress['shards_total']} shards</span>
                    </div>
                    <div style='display:flex; justify-content:space-between; font-size:0.7rem; color:#94a3b8; margin-top:2px;'>
                        <span>ETA: {progress['eta']}</span>
                        <span>{progress['pct']}%</span>
                    </div>
                    <div style='background: rgba(255,255,255,0.05); height: 6px; border-radius: 3px; overflow: hidden; margin-top: 6px;'>
                        <div style='width: {progress['pct']}%; height: 100%; background: linear-gradient(90deg, #60a5fa, #3b82f6); transition: width 0.5s ease;'></div>
                    </div>
                </div>
                """
                
        html_parts.append(f"""
        <div class='diag-service-row' style='display: block;'>
            <div style='display: flex; align-items: center; justify-content: space-between; width: 100%;'>
                <div class='diag-service-name'>
                    <span>{service_names[s]}</span>
                    {badge}
                    {model_badge}
                </div>
                <div class='diag-service-latency'>
                    <span>{latency_str}</span>
                    {sparkline}
                </div>
            </div>
            {extra_badge}
        </div>
        """)
        
    return "".join(html_parts)


def make_system_health_badge_html(data: Dict[str, Any]) -> str:
    service_names = {
        "postgres": "PostgreSQL",
        "redis": "Redis",
        "minio": "MinIO",
        "qdrant": "Qdrant",
        "vllm": "vLLM"
    }
    fixes = {
        "postgres": "Start PostgreSQL service/container.",
        "redis": "Start Redis service/container.",
        "minio": "Start MinIO service/container.",
        "qdrant": "Start Qdrant service/container.",
        "vllm": "Start vLLM service/container."
    }
    
    all_healthy = data["all_healthy"]
    vllm_model = data["vllm_model"]
    failed_services = data["failed_services"]
    vllm_progress = data["vllm_progress"]
    
    if all_healthy:
        if not vllm_model or vllm_model in ["None Loaded", "Unknown"]:
            suitability = "No model loaded"
            suit_color = "#94a3b8"
        elif "olmocr" in vllm_model.lower():
            suitability = "Best suited for PDF conversion"
            suit_color = "#34d399"
        else:
            suitability = "Best suited for RAG processing"
            suit_color = "#60a5fa"
            
        display_model = vllm_model if vllm_model else "None Loaded"
        return f"""
        <div style='display: flex; flex-direction: column; align-items: center; gap: 4px; text-align: center;'>
            <span class='badge-success' style='padding: 6px 12px; font-weight: 700;'>✓ System Healthy</span>
            <div style='font-size: 0.75rem; color: #94a3b8; line-height: 1.2;'>
                Model: <span style='font-family: monospace; color: #e2e8f0; font-weight: 600;'>{display_model}</span><br>
                <span style='color: {suit_color}; font-weight: 600;'>● {suitability}</span>
            </div>
        </div>
        """
    else:
        degraded_names = [service_names[s] for s in failed_services]
        degraded_str = ", ".join(degraded_names)
        fix_instructions = [fixes[s] for s in failed_services]
        fix_str = " ".join(fix_instructions)
        
        if failed_services == ["vllm"] and vllm_progress:
            progress = vllm_progress
            return f"""
            <div style='display: flex; flex-direction: column; align-items: center; gap: 4px; text-align: center;'>
                <span class='badge-running' style='padding: 6px 12px; font-weight: 700; animation: pulse 2s infinite;'>⚡ Model Loading</span>
                <div style='font-size: 0.75rem; color: #93c5fd; line-height: 1.2;'>
                    Progress: <span style='font-weight:600; color:#e2e8f0;'>{progress['pct']}%</span> ({progress['shards_loaded']}/{progress['shards_total']})<br>
                    <span style='color: #93c5fd;'>ETA: {progress['eta']}</span>
                </div>
            </div>
            """
        else:
            return f"""
            <div style='display: flex; flex-direction: column; align-items: center; gap: 4px; text-align: center;'>
                <span class='badge-failed' style='padding: 6px 12px; font-weight: 700;'>✗ System Degraded</span>
                <div style='font-size: 0.75rem; color: #fca5a5; line-height: 1.2;'>
                    <span style='font-weight: 600;'>Offline:</span> {degraded_str}<br>
                    <span style='color: #e2e8f0;'>Fix: {fix_str}</span>
                </div>
            </div>
            """


def make_gpu_metrics_html(data: Dict[str, Any]) -> str:
    if not data["cuda_available"]:
        return """
        <div style='background: rgba(17, 24, 39, 0.5); border: 1px solid rgba(255, 255, 255, 0.05); border-radius: 10px; padding: 18px; margin-top: 10px;'>
            <div style='display:flex; justify-content:space-between; align-items:center; margin-bottom:12px;'>
                <span class='badge-idle' style='padding:4px 10px;'>⚠ CUDA Unavailable</span>
            </div>
            <div style='font-weight:600; font-size:1.05rem; color:#e2e8f0;'>Running on Host CPU</div>
            <div style='color:#94a3b8; font-size:0.85rem; margin-top:4px;'>VRAM Usage</div>
            
            <div class='vram-progress-container'>
                <div style='display:flex; justify-content:space-between; font-size:0.85rem; font-family:"JetBrains Mono", monospace; color:#94a3b8; margin-bottom:4px;'>
                    <span>0.0%</span>
                    <span>0 MB / 0 MB</span>
                </div>
                <div class='vram-bar-outer'>
                    <div class='vram-bar-inner' style='width: 0%;'></div>
                </div>
            </div>
        </div>
        """
        
    gpu_name = data["gpu_name"]
    vram_used = data["vram_used"]
    vram_total = data["vram_total"]
    vram_pct = data["vram_pct"]
    vram_free = data["vram_free"]
    vram_reclaimable = data["vram_reclaimable"]
    vram_potential_free = data["vram_potential_free"]
    processes = data["processes"]
    
    rows_html = ""
    if not processes:
        rows_html = """
        <tr>
            <td colspan='3' style='padding: 12px; text-align: center; color: #94a3b8;'>No active GPU processes detected.</td>
        </tr>
        """
    else:
        for rp in processes:
            rows_html += f"""
            <tr style='border-bottom: 1px solid rgba(255,255,255,0.05);'>
                <td style='padding: 8px 10px; max-width: 180px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;'>
                    <span style='color: #e2e8f0; font-weight: 600;' title="{rp['cmdline']}">{rp['display_name']}</span><br>
                    <span style='color: #64748b; font-size: 0.7rem;'>PID: {rp['pid']}</span>
                </td>
                <td style='padding: 8px 10px; font-family: "JetBrains Mono", monospace; color: #e2e8f0;'>
                    {rp['vram']:,.0f} MB
                </td>
                <td style='padding: 8px 10px;'>
                    <span style='padding: 2px 6px; border-radius: 4px; font-size: 0.7rem; font-weight: 600; display: inline-block; {rp['type_badge_style']}'>
                        {rp['type_text']}
                    </span>
                    <div style='font-size: 0.65rem; color: {rp['action_color']}; margin-top: 2px;'>{rp['action_text']}</div>
                </td>
            </tr>
            """

    return f"""
    <div style='background: rgba(17, 24, 39, 0.5); border: 1px solid rgba(255, 255, 255, 0.05); border-radius: 12px; padding: 18px; margin-top: 10px;'>
        <div style='display:flex; justify-content:space-between; align-items:center; margin-bottom:12px;'>
            <span class='badge-success' style='padding:4px 10px;'>✓ CUDA Available - 1 GPU</span>
        </div>
        <div style='font-weight:600; font-size:1.05rem; color:#e2e8f0;'>GPU 0: {gpu_name}</div>
        
        <div style='color:#94a3b8; font-size:0.85rem; margin-top:12px; margin-bottom:4px;'>Overall VRAM Usage</div>
        <div class='vram-progress-container'>
            <div style='display:flex; justify-content:space-between; font-size:0.85rem; font-family:"JetBrains Mono", monospace; color:#34d399; margin-bottom:4px;'>
                <span>{vram_pct:.1f}%</span>
                <span>{vram_used:,.0f} MB / {vram_total:,.0f} MB</span>
            </div>
            <div class='vram-bar-outer' style='background: rgba(255,255,255,0.05); height: 8px; border-radius: 4px; overflow: hidden;'>
                <div class='vram-bar-inner' style='width: {vram_pct:.1f}%; height: 100%; background: linear-gradient(90deg, #34d399, #10b981); transition: width 0.5s ease;'></div>
            </div>
        </div>
        
        <div style='display: grid; grid-template-columns: 1fr 1fr; gap: 8px; margin-top: 16px; margin-bottom: 16px;'>
            <div style='background: rgba(30, 41, 59, 0.4); border: 1px solid rgba(255, 255, 255, 0.05); border-radius: 8px; padding: 10px; text-align: center;'>
                <div style='font-size: 0.75rem; color: #94a3b8; text-transform: uppercase;'>Free VRAM</div>
                <div style='font-size: 1.15rem; font-weight: 700; color: #34d399; margin-top: 4px;'>{vram_free:,.0f} MB</div>
            </div>
            <div style='background: rgba(30, 41, 59, 0.4); border: 1px solid rgba(255, 255, 255, 0.05); border-radius: 8px; padding: 10px; text-align: center;'>
                <div style='font-size: 0.75rem; color: #94a3b8; text-transform: uppercase;'>Reclaimable</div>
                <div style='font-size: 1.15rem; font-weight: 700; color: #fbbf24; margin-top: 4px;'>{vram_reclaimable:,.0f} MB</div>
            </div>
            <div style='grid-column: span 2; background: rgba(16, 185, 129, 0.1); border: 1px solid rgba(16, 185, 129, 0.2); border-radius: 8px; padding: 10px; text-align: center;'>
                <div style='font-size: 0.75rem; color: #a7f3d0; text-transform: uppercase;'>Max Potential Free VRAM</div>
                <div style='font-size: 1.25rem; font-weight: 800; color: #34d399; margin-top: 4px;'>{vram_potential_free:,.0f} MB</div>
                <div style='font-size: 0.7rem; color: #6ee7b7; margin-top: 2px;'>If non-essential apps & containers are stopped</div>
            </div>
        </div>
        
        <div style='margin-top: 16px;'>
            <div style='font-size: 0.9rem; font-weight: 600; color: #c7d2fe; margin-bottom: 8px;'>Active GPU Processes</div>
            
            <div style='max-height: 500px; overflow-y: auto; border: 1px solid rgba(255, 255, 255, 0.05); border-radius: 8px;'>
                <table style='width: 100%; border-collapse: collapse; font-size: 0.8rem; text-align: left; background: rgba(15, 23, 42, 0.3);'>
                    <thead>
                        <tr style='border-bottom: 1px solid rgba(255, 255, 255, 0.1); background: rgba(15, 23, 42, 0.6); color: #94a3b8;'>
                            <th style='padding: 8px 10px;'>Process / PID</th>
                            <th style='padding: 8px 10px;'>VRAM</th>
                            <th style='padding: 8px 10px;'>Type / Action</th>
                        </tr>
                    </thead>
                    <tbody>
                        {rows_html}
                    </tbody>
                </table>
            </div>
        </div>
    </div>
    """


def make_case_dashboard_html(runs: List[Dict[str, Any]]) -> str:
    import os
    if not runs:
        return (
            "<div class='dashboard-empty'>"
            "<div style='font-size:2.5rem; margin-bottom:12px;'>📂</div>"
            "<div>No indexed cases yet.</div>"
            "<div style='font-size:0.9rem; margin-top:8px; color:#4b5563;'>"
            "Upload and index documents using the Analysis tab to see them here.</div>"
            "</div>"
        )

    cards = []
    for run in runs:
        run_dir = run.get("run_dir", "")
        run_name = os.path.basename(run_dir) if run_dir else run.get("run_id", "unknown")
        docs = run.get("total_documents", 0)
        chunks = run.get("total_chunks", 0)
        authors = run.get("unique_authors", 0)
        earliest = run.get("earliest_date", None)
        latest = run.get("latest_date", None)
        indexed_at = run.get("indexed_at", None)

        date_range = "—"
        if earliest and latest:
            date_range = f"{earliest} → {latest}"
        elif earliest:
            date_range = f"{earliest} → ..."
        elif latest:
            date_range = f"... → {latest}"

        indexed_str = ""
        if indexed_at:
            try:
                indexed_str = indexed_at.strftime("%Y-%m-%d %H:%M")
            except Exception:
                indexed_str = str(indexed_at)[:16]

        run_id = run.get("run_id", "")
        
        # Fetch rich case metadata
        from rag.metadata_helper import get_case_metadata
        meta = get_case_metadata(run_id)
        
        client_display = ", ".join(meta["names"]) if meta["names"] else "Unknown Client"
        dob_display = meta["dob"] if meta["dob"] != "—" else "Not Extracted"
        
        if meta["injuries"]:
            injury_display = "<ul style='margin: 0; padding-left: 14px; font-size: 0.8rem;'>" + "".join([f"<li style='margin-bottom: 2px;'>{inj}</li>" for inj in meta["injuries"]]) + "</ul>"
        else:
            injury_display = "<span style='color: #9ca3af; font-style: italic; font-size: 0.8rem;'>No specific injury or diagnosis found.</span>"

        card = f"""
        <div class="case-card" onclick="window.toggleCaseSelection(this, event, '{run_id}')" style="cursor: pointer;">
            <div class="case-card-header" style="margin-bottom: 8px;">
                <div style="display: flex; flex-direction: column; gap: 4px; width: 100%;">
                    <div style="font-size: 0.75rem; color: #9ca3af; font-family: monospace; word-break: break-all; opacity: 0.85;">📁 {run_name}</div>
                    <div class="case-card-title" style="margin: 4px 0 0 0; font-size: 1.15rem; color: #f3f4f6; font-weight: 700; display: flex; align-items: center; gap: 6px;">
                        <span>👤</span> {client_display}
                    </div>
                </div>
                <input type="checkbox" class="case-select-checkbox" data-run-id="{run_id}" onclick="event.stopPropagation(); window.toggleCaseSelection(this, event, '{run_id}');" />
            </div>
            
            <div style="margin: 6px 0; font-size: 0.85rem; color: #d1d5db; display: flex; align-items: center; gap: 6px;">
                <span style="color: #818cf8; font-weight: 600;">📅 DOB:</span> <span>{dob_display}</span>
            </div>

            <div style="margin: 10px 0; padding: 8px 12px; background: rgba(99, 102, 241, 0.05); border: 1px solid rgba(99, 102, 241, 0.15); border-radius: 8px;">
                <div style="font-size: 0.7rem; color: #a5b4fc; text-transform: uppercase; letter-spacing: 0.05em; font-weight: 700; margin-bottom: 4px; display: flex; align-items: center; gap: 4px;">
                    <span>🤕</span> Injury / Diagnosis
                </div>
                <div style="color: #e0e7ff; line-height: 1.4;">
                    {injury_display}
                </div>
            </div>

            <div class="case-card-stats" style="margin-top: 12px; border-top: 1px solid rgba(255, 255, 255, 0.06); padding-top: 10px; opacity: 0.9;">
                <span>Documents: <span class="stat-val">{docs}</span></span>
                <span>Chunks: <span class="stat-val">{chunks}</span></span>
                <span>Authors: <span class="stat-val">{authors}</span></span>
                <span>Date Range: <span class="stat-val" style="font-size: 0.75rem;">{date_range}</span></span>
            </div>
            
            <div style="font-size:0.75rem; color:#6b7280; margin-top:10px; display: flex; justify-content: space-between; align-items: center;">
                <span class="badge-success" style="font-size:0.7rem; padding: 2px 6px; border-radius: 4px; background: rgba(16, 185, 129, 0.08); border: 1px solid rgba(16, 185, 129, 0.15); color: #34d399;">✓ Indexed</span>
                <span>{indexed_str}</span>
            </div>
        </div>
        """
        cards.append(card)

    return f"<div class='case-dashboard-grid'>{''.join(cards)}</div>"


def make_case_banner_html(active_case_label: Optional[str]) -> str:
    if not active_case_label or "All Cases" in str(active_case_label):
        return (
            "<div class='active-case-banner'>"
            "<span class='banner-icon'>🌐</span>"
            "<span><span class='banner-label'>Active Case:</span> "
            "<span class='banner-value'>All Cases — querying entire corpus</span></span>"
            "</div>"
        )
    name = str(active_case_label)
    return (
        "<div class='active-case-banner'>"
        "<span class='banner-icon'>📂</span>"
        "<span><span class='banner-label'>Active Case:</span> "
        f"<span class='banner-value'>{name}</span></span>"
        "</div>"
    )
