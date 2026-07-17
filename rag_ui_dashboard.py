import os
import gradio as gr
from rag_ui_state import log_to_rag

def get_rag_ui_fn(name, fallback):
    import sys
    rag_ui = sys.modules.get('rag_ui')
    return getattr(rag_ui, name, fallback)

def _build_dashboard_html():
    """Build HTML card grid for the Case Dashboard tab.

    Returns:
        HTML string with styled case cards.
    """
    try:
        from rag.db import get_runs_with_stats
        runs = get_runs_with_stats()
    except Exception as e:
        return f"<div class='dashboard-empty'>⚠️ Cannot load cases: {e}</div>"

    from html_utils import make_case_dashboard_html
    return make_case_dashboard_html(runs)


def _get_indexed_run_choices():
    """Get indexed runs as dropdown choices for the Active Case Selector.

    Returns:
        List of (display_label, run_id) tuples, plus an 'All Cases' option.
    """
    choices = [("🌐 All Cases (no filter)", "")]
    try:
        from rag.db import get_runs_with_stats
        runs = get_runs_with_stats()
        for run in runs:
            run_dir = run.get("run_dir", "")
            run_name = os.path.basename(run_dir) if run_dir else "unknown"
            run_id = run.get("run_id", "")
            docs = run.get("total_documents", 0)
            chunks = run.get("total_chunks", 0)
            label = f"📁 {run_name} ({docs} docs, {chunks} chunks)"
            choices.append((label, run_id))
    except Exception:
        pass
    return choices


def _refresh_active_case_after_upload():
    import rag_ui_state
    get_choices = get_rag_ui_fn('_get_indexed_run_choices', _get_indexed_run_choices)
    choices = get_choices()
    val = rag_ui_state.LAST_CREATED_RUN_ID if rag_ui_state.LAST_CREATED_RUN_ID else ""
    return gr.update(choices=choices, value=val)


def _get_case_banner_html(active_case_label):
    """Generate the active case indicator banner HTML."""
    from html_utils import make_case_banner_html
    return make_case_banner_html(active_case_label)


def build_case_dashboard_ui():
    """Build the Case Dashboard UI components.
    
    Returns:
        Dict of component references.
    """
    build_html = get_rag_ui_fn('_build_dashboard_html', _build_dashboard_html)
    
    with gr.Row():
        gr.HTML(
            "<h2 class='inline-case-title'>📊 Indexed Cases</h2>"
        )
    with gr.Row():
        dashboard_refresh_btn = gr.Button("🔄 Refresh Dashboard", variant="secondary", size="sm")
        dashboard_delete_selector = gr.Dropdown(
            label="Select case to delete",
            choices=[],
            interactive=True,
            scale=3,
        )
        dashboard_delete_btn = gr.Button("🗑️ Delete Case", variant="stop", size="sm")
    dashboard_html = gr.HTML(value=build_html())
    dashboard_status = gr.Markdown("")

    def _refresh_dashboard():
        build_html_fn = get_rag_ui_fn('_build_dashboard_html', _build_dashboard_html)
        get_choices_fn = get_rag_ui_fn('_get_indexed_run_choices', _get_indexed_run_choices)
        html = build_html_fn()
        choices = get_choices_fn()
        # Build delete selector choices (skip "All Cases")
        del_choices = [(lbl, rid) for lbl, rid in choices if rid]
        return html, gr.update(choices=del_choices, value=None), ""

    dashboard_refresh_btn.click(
        _refresh_dashboard,
        outputs=[dashboard_html, dashboard_delete_selector, dashboard_status],
    )

    def _delete_case(run_id):
        build_html_fn = get_rag_ui_fn('_build_dashboard_html', _build_dashboard_html)
        get_choices_fn = get_rag_ui_fn('_get_indexed_run_choices', _get_indexed_run_choices)
        if not run_id:
            return build_html_fn(), gr.update(), "⚠️ No case selected."
        try:
            from rag.db import delete_run_data
            delete_run_data(run_id)
            log_to_rag(f"Deleted case data from PostgreSQL: {run_id[:12]}...")
        except Exception as e:
            log_to_rag(f"DB delete warning: {e}")

        try:
            from rag.embedding import delete_run_vectors
            delete_run_vectors(run_id)
            log_to_rag(f"Deleted vectors from Qdrant: {run_id[:12]}...")
        except Exception as e:
            log_to_rag(f"Vector delete warning: {e}")

        try:
            from rag.storage import delete_run_objects
            delete_run_objects(run_id)
            log_to_rag(f"Deleted blobs from MinIO: {run_id[:12]}...")
        except Exception as e:
            log_to_rag(f"Storage delete warning: {e}")

        try:
            from rag.cache import invalidate_query_cache
            invalidate_query_cache()
        except Exception:
            pass

        html = build_html_fn()
        choices = get_choices_fn()
        del_choices = [(lbl, rid) for lbl, rid in choices if rid]
        return html, gr.update(choices=del_choices, value=None), "✅ Case deleted successfully."

    dashboard_delete_btn.click(
        _delete_case,
        inputs=[dashboard_delete_selector],
        outputs=[dashboard_html, dashboard_delete_selector, dashboard_status],
    )

    return {
        "dashboard_html": dashboard_html,
        "dashboard_delete_selector": dashboard_delete_selector,
        "dashboard_status": dashboard_status,
        "refresh_btn": dashboard_refresh_btn,
        "refresh_fn": _refresh_dashboard,
        "delete_fn": _delete_case,
    }
