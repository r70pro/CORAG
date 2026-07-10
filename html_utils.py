def make_progress_bar_html(completed, total, elapsed_secs=0):
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


def make_file_status_html(file_mapping, file_page_counts, completed_files_set, failed_files_set=None):
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


def make_upload_manifest_html(file_mapping, file_page_counts, file_sizes):
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
