import logging
import os

import gradio as gr

from rag_ui_state import log_to_rag

logger = logging.getLogger(__name__)


def get_rag_ui_fn(name, fallback):
    import sys

    rag_ui = sys.modules.get("rag_ui")
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

    # Batch-fetch all case metadata in a single set of queries (avoids N+1).
    cases_metadata = {}
    try:
        from rag.metadata_helper import get_all_cases_metadata

        run_ids = [r.get("run_id") for r in runs if r.get("run_id")]
        cases_metadata = get_all_cases_metadata(run_ids)
    except Exception as e:
        logger.warning(f"Warning: could not pre-load case metadata: {e}")

    return make_case_dashboard_html(runs, cases_metadata)


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

    get_choices = get_rag_ui_fn("_get_indexed_run_choices", _get_indexed_run_choices)
    choices = get_choices()
    val = rag_ui_state.LAST_CREATED_RUN_ID if rag_ui_state.LAST_CREATED_RUN_ID else ""
    return gr.update(choices=choices, value=val)


def _get_case_banner_html(active_case_label):
    """Generate the active case indicator banner HTML."""
    from html_utils import make_case_banner_html

    return make_case_banner_html(active_case_label)


def _update_delete_button_label(selected_ids_str):
    if not selected_ids_str:
        return gr.update(value="🗑️ Delete Selected")
    ids = [i for i in selected_ids_str.split(",") if i.strip()]
    if not ids:
        return gr.update(value="🗑️ Delete Selected")
    return gr.update(value=f"🗑️ Delete Selected ({len(ids)})")


def build_case_dashboard_ui():
    """Build the Case Dashboard UI components.

    Returns:
        Dict of component references.
    """
    build_html = get_rag_ui_fn("_build_dashboard_html", _build_dashboard_html)

    with gr.Row():
        gr.HTML("<h2 class='inline-case-title'>📊 Indexed Cases</h2>")
    with gr.Row():
        dashboard_refresh_btn = gr.Button("🔄 Refresh Dashboard", variant="secondary", size="sm")
        dashboard_select_all_btn = gr.Button("☑️ Select All", variant="secondary", size="sm")
        dashboard_deselect_all_btn = gr.Button("⬜ Clear Selection", variant="secondary", size="sm")
        dashboard_delete_selected_btn = gr.Button("🗑️ Delete Selected", variant="stop", size="sm")
        dashboard_delete_all_btn = gr.Button("🚨 Delete All Cases", variant="stop", size="sm")

        # Keep hidden dropdown for backward compatibility with existing tests/wrappers
        dashboard_delete_selector = gr.Dropdown(
            label="Select case to delete",
            choices=[],
            interactive=True,
            visible=False,
        )
        # Keep hidden button for backward compatibility with tests
        dashboard_delete_btn = gr.Button("🗑️ Delete Case", variant="stop", visible=False)

    selected_cases_input = gr.Textbox(elem_id="selected-cases-input", visible=True, value="")
    dashboard_html = gr.HTML(value=build_html())
    dashboard_status = gr.Markdown("")

    selected_cases_input.change(
        _update_delete_button_label,
        inputs=[selected_cases_input],
        outputs=[dashboard_delete_selected_btn],
    )

    dashboard_select_all_btn.click(
        None,
        js="""() => {
            const checkboxes = document.querySelectorAll('.case-select-checkbox');
            const selectedIds = [];
            checkboxes.forEach(cb => {
                cb.checked = true;
                const card = cb.closest('.case-card');
                if (card) card.classList.add('selected');
                const rid = cb.getAttribute('data-run-id');
                if (rid) selectedIds.push(rid);
            });
            const txtEl = document.querySelector('#selected-cases-input textarea, #selected-cases-input input');
            if (txtEl) {
                txtEl.value = selectedIds.join(',');
                txtEl.dispatchEvent(new Event('input', { bubbles: true }));
            }
        }""",
    )

    dashboard_deselect_all_btn.click(
        None,
        js="""() => {
            const checkboxes = document.querySelectorAll('.case-select-checkbox');
            checkboxes.forEach(cb => {
                cb.checked = false;
                const card = cb.closest('.case-card');
                if (card) card.classList.remove('selected');
            });
            const txtEl = document.querySelector('#selected-cases-input textarea, #selected-cases-input input');
            if (txtEl) {
                txtEl.value = '';
                txtEl.dispatchEvent(new Event('input', { bubbles: true }));
            }
        }""",
    )

    def _refresh_dashboard():
        build_html_fn = get_rag_ui_fn("_build_dashboard_html", _build_dashboard_html)
        get_choices_fn = get_rag_ui_fn("_get_indexed_run_choices", _get_indexed_run_choices)
        html = build_html_fn()
        choices = get_choices_fn()
        # Build delete selector choices (skip "All Cases")
        del_choices = [(lbl, rid) for lbl, rid in choices if rid]
        return html, gr.update(choices=del_choices, value=None), ""

    # When refresh clicked, we run the Python refresh fn and also run JS to clear the frontend selection state
    dashboard_refresh_btn.click(
        _refresh_dashboard,
        outputs=[dashboard_html, dashboard_delete_selector, dashboard_status],
    )
    dashboard_refresh_btn.click(
        None,
        js="""() => {
            const checkboxes = document.querySelectorAll('.case-select-checkbox');
            checkboxes.forEach(cb => {
                cb.checked = false;
                const card = cb.closest('.case-card');
                if (card) card.classList.remove('selected');
            });
            const txtEl = document.querySelector('#selected-cases-input textarea, #selected-cases-input input');
            if (txtEl) {
                txtEl.value = '';
                txtEl.dispatchEvent(new Event('input', { bubbles: true }));
            }
        }""",
    )

    def _delete_case(run_id):
        build_html_fn = get_rag_ui_fn("_build_dashboard_html", _build_dashboard_html)
        get_choices_fn = get_rag_ui_fn("_get_indexed_run_choices", _get_indexed_run_choices)
        if not run_id:
            return build_html_fn(), gr.update(), "⚠️ No case selected."

        from settings_manager import delete_run_directory

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
            delete_run_directory(run_id)
            log_to_rag(f"Deleted run folder from disk: {run_id[:12]}...")
        except Exception as e:
            log_to_rag(f"Disk directory delete warning: {e}")

        try:
            from rag.cache import invalidate_query_cache

            invalidate_query_cache()
        except Exception:
            pass

        html = build_html_fn()
        choices = get_choices_fn()
        del_choices = [(lbl, rid) for lbl, rid in choices if rid]
        return html, gr.update(choices=del_choices, value=None), "✅ Case deleted successfully."

    # Keep compatibility click binding for tests
    dashboard_delete_btn.click(
        _delete_case,
        inputs=[dashboard_delete_selector],
        outputs=[dashboard_html, dashboard_delete_selector, dashboard_status],
    )

    def _delete_selected_cases(selected_ids_str):
        build_html_fn = get_rag_ui_fn("_build_dashboard_html", _build_dashboard_html)
        get_choices_fn = get_rag_ui_fn("_get_indexed_run_choices", _get_indexed_run_choices)
        if not selected_ids_str:
            return build_html_fn(), gr.update(), "", "⚠️ No case selected."

        run_ids = [rid.strip() for rid in selected_ids_str.split(",") if rid.strip()]
        if not run_ids:
            return build_html_fn(), gr.update(), "", "⚠️ No case selected."

        from settings_manager import delete_run_directory

        deleted_count = 0
        for run_id in run_ids:
            try:
                from rag.db import delete_run_data

                delete_run_data(run_id)
                log_to_rag(f"Deleted case data from PostgreSQL: {run_id[:12]}...")
                deleted_count += 1
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
                delete_run_directory(run_id)
                log_to_rag(f"Deleted run folder from disk: {run_id[:12]}...")
            except Exception as e:
                log_to_rag(f"Disk directory delete warning: {e}")

        try:
            from rag.cache import invalidate_query_cache

            invalidate_query_cache()
        except Exception:
            pass

        html = build_html_fn()
        choices = get_choices_fn()
        del_choices = [(lbl, rid) for lbl, rid in choices if rid]
        return (
            html,
            gr.update(choices=del_choices, value=None),
            "",
            f"✅ Selected case(s) ({deleted_count}) deleted successfully.",
        )

    dashboard_delete_selected_btn.click(
        _delete_selected_cases,
        inputs=[selected_cases_input],
        outputs=[dashboard_html, dashboard_delete_selector, selected_cases_input, dashboard_status],
    )

    def _delete_all_cases():
        build_html_fn = get_rag_ui_fn("_build_dashboard_html", _build_dashboard_html)
        get_choices_fn = get_rag_ui_fn("_get_indexed_run_choices", _get_indexed_run_choices)

        try:
            from rag.db import get_all_runs, get_runs_with_stats

            runs = get_all_runs()
            if not runs:
                runs = get_runs_with_stats()
            run_ids = [run.get("run_id") for run in runs if run.get("run_id")]
        except Exception as e:
            log_to_rag(f"Failed to fetch runs for delete all: {e}")
            return build_html_fn(), gr.update(), "", f"⚠️ Error fetching cases: {e}"

        if not run_ids:
            return build_html_fn(), gr.update(), "", "⚠️ No cases to delete."

        from settings_manager import delete_run_directory

        deleted_count = 0
        for run_id in run_ids:
            try:
                from rag.db import delete_run_data

                delete_run_data(run_id)
                log_to_rag(f"Deleted case data from PostgreSQL: {run_id[:12]}...")
                deleted_count += 1
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
                delete_run_directory(run_id)
                log_to_rag(f"Deleted run folder from disk: {run_id[:12]}...")
            except Exception as e:
                log_to_rag(f"Disk directory delete warning: {e}")

        try:
            from rag.cache import invalidate_query_cache

            invalidate_query_cache()
        except Exception:
            pass

        html = build_html_fn()
        choices = get_choices_fn()
        del_choices = [(lbl, rid) for lbl, rid in choices if rid]
        return (
            html,
            gr.update(choices=del_choices, value=None),
            "",
            f"✅ All cases ({deleted_count}) deleted successfully.",
        )

    dashboard_delete_all_btn.click(
        _delete_all_cases,
        outputs=[dashboard_html, dashboard_delete_selector, selected_cases_input, dashboard_status],
    )

    return {
        "dashboard_html": dashboard_html,
        "dashboard_delete_selector": dashboard_delete_selector,
        "dashboard_status": dashboard_status,
        "refresh_btn": dashboard_refresh_btn,
        "refresh_fn": _refresh_dashboard,
        "delete_fn": _delete_case,
        "selected_cases_input": selected_cases_input,
        "delete_selected_btn": dashboard_delete_selected_btn,
        "delete_all_btn": dashboard_delete_all_btn,
        "delete_selected_fn": _delete_selected_cases,
        "delete_all_fn": _delete_all_cases,
    }
