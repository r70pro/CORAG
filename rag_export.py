"""
RAG Export — Generate downloadable files from chat analysis sessions.

Provides:
- Markdown export of full chat sessions
- Plain text export for paste into Word/email
- CSV export of timeline-mode table output
"""

import os
import re
import csv
import datetime

EXPORT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "workspace", "exports")


def _ensure_export_dir():
    """Create the exports directory if needed."""
    os.makedirs(EXPORT_DIR, exist_ok=True)


def _make_export_filename(prefix, ext):
    """Generate a timestamped export filename."""
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{ts}.{ext}"


def _extract_case_name(active_case):
    """Extract a human-readable case name from the active case label."""
    if not active_case or active_case == "All Cases":
        return "all_cases"
    # Strip common prefixes and clean for filenames
    name = active_case.replace("workspace/", "").replace("run_", "")
    name = re.sub(r'[^\w\-]', '_', name)
    return name[:60]


def _history_to_pairs(history):
    """Convert Gradio chatbot history (list of dicts) to (role, content) pairs."""
    pairs = []
    if not history:
        return pairs
    for msg in history:
        if isinstance(msg, dict):
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
        elif isinstance(msg, (list, tuple)) and len(msg) >= 2:
            role = "user" if msg[0] is not None else "assistant"
            content = msg[0] if msg[0] else msg[1]
        else:
            continue
        pairs.append((role, str(content)))
    return pairs


def export_chat_markdown(history, mode="Free Q&A", active_case="All Cases"):
    """Export a chat session as a Markdown file.

    Args:
        history: Gradio chatbot history (list of message dicts).
        mode: Current analysis mode name.
        active_case: Active case label.

    Returns:
        File path to the generated .md file, or None on error.
    """
    if not history:
        return None

    _ensure_export_dir()
    case_name = _extract_case_name(active_case)
    filename = _make_export_filename(f"analysis_{case_name}", "md")
    filepath = os.path.join(EXPORT_DIR, filename)

    pairs = _history_to_pairs(history)

    lines = [
        "# RAG Analysis Export",
        "",
        f"**Mode:** {mode}",
        f"**Case:** {active_case}",
        f"**Exported:** {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"**Messages:** {len(pairs)}",
        "",
        "---",
        "",
    ]

    for role, content in pairs:
        if role == "user":
            lines.append("## 👤 User Query")
            lines.append("")
            lines.append(content)
            lines.append("")
        else:
            lines.append("## 🤖 Analysis Response")
            lines.append("")
            lines.append(content)
            lines.append("")
            lines.append("---")
            lines.append("")

    with open(filepath, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    return filepath


def export_chat_text(history, mode="Free Q&A", active_case="All Cases"):
    """Export a chat session as a plain text file.

    Args:
        history: Gradio chatbot history (list of message dicts).
        mode: Current analysis mode name.
        active_case: Active case label.

    Returns:
        File path to the generated .txt file, or None on error.
    """
    if not history:
        return None

    _ensure_export_dir()
    case_name = _extract_case_name(active_case)
    filename = _make_export_filename(f"analysis_{case_name}", "txt")
    filepath = os.path.join(EXPORT_DIR, filename)

    pairs = _history_to_pairs(history)

    lines = [
        "RAG ANALYSIS EXPORT",
        f"{'=' * 60}",
        f"Mode:      {mode}",
        f"Case:      {active_case}",
        f"Exported:  {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Messages:  {len(pairs)}",
        f"{'=' * 60}",
        "",
    ]

    for role, content in pairs:
        if role == "user":
            lines.append("USER QUERY:")
            lines.append(f"{'-' * 40}")
            lines.append(content)
            lines.append("")
        else:
            lines.append("ANALYSIS RESPONSE:")
            lines.append(f"{'-' * 40}")
            # Strip markdown formatting for plain text
            plain = content
            # Remove markdown headers
            plain = re.sub(r'^#{1,6}\s+', '', plain, flags=re.MULTILINE)
            # Remove bold/italic markers
            plain = re.sub(r'\*{1,3}([^*]+)\*{1,3}', r'\1', plain)
            # Remove markdown links but keep text
            plain = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', plain)
            lines.append(plain)
            lines.append("")
            lines.append(f"{'=' * 60}")
            lines.append("")

    with open(filepath, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    return filepath


def export_timeline_csv(history, active_case="All Cases"):
    """Extract timeline table from chat history and export as CSV.

    Parses Markdown tables from assistant responses in timeline mode.

    Args:
        history: Gradio chatbot history (list of message dicts).
        active_case: Active case label.

    Returns:
        File path to the generated .csv file, or None on error/no tables.
    """
    if not history:
        return None

    _ensure_export_dir()
    case_name = _extract_case_name(active_case)
    filename = _make_export_filename(f"timeline_{case_name}", "csv")
    filepath = os.path.join(EXPORT_DIR, filename)

    # Extract all markdown tables from assistant messages
    table_rows = []
    headers = None

    pairs = _history_to_pairs(history)

    for role, content in pairs:
        if role != "assistant" or not content:
            continue

        # Find markdown table lines: lines starting with |
        lines = content.split("\n")
        in_table = False
        for line in lines:
            stripped = line.strip()
            if stripped.startswith("|") and stripped.endswith("|"):
                # Skip separator lines (e.g., |---|---|---|)
                if re.match(r'^\|[\s\-:|]+\|$', stripped):
                    continue
                # Parse cells
                cells = [c.strip() for c in stripped.split("|")[1:-1]]
                if not headers:
                    headers = cells
                    in_table = True
                else:
                    table_rows.append(cells)
                    in_table = True
            elif in_table and not stripped.startswith("|"):
                in_table = False

    if not headers and not table_rows:
        return None

    with open(filepath, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        for row in table_rows:
            writer.writerow(row)

    return filepath
