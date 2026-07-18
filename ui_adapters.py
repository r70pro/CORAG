"""
UI Adapters — translate plain backend data into Gradio component updates.

This module is the **only place** outside the core UI files (app.py, rag_ui.py)
where ``gradio`` is imported.  Backend modules (pipeline_manager, pdf_manager,
etc.) return plain Python dicts/strings; the functions here convert them into
``gr.update()`` objects for the Gradio event system.
"""

import gradio as gr

from pipeline_manager import PipelineResult

# ---------------------------------------------------------------------------
# Index constants for PipelineResult tuple positions
# ---------------------------------------------------------------------------
_IDX_LOG = 0
_IDX_STATUS_BADGE = 1
_IDX_PROGRESS = 2
_IDX_COMPLETED = 3
_IDX_FAILED = 4
_IDX_FILE_SELECTOR = 5
_IDX_ZIP = 6
_IDX_INDIVIDUAL = 7
_IDX_START_BTN = 8
_IDX_RUN_ID = 9
_IDX_FILE_STATUS = 10
_IDX_MANIFEST = 11
_IDX_STOP_BTN = 12


def _dict_to_update(val):
    """Convert a plain dict of kwargs to a gr.update(), or pass through."""
    if isinstance(val, dict):
        return gr.update(**val)
    return val


def pipeline_result_to_gradio(result: PipelineResult) -> tuple:
    """Convert a PipelineResult (plain data) into a tuple of gr.update() objects.

    The returned tuple has the same positional layout expected by the Gradio
    callback output list in ``app.py``.
    """
    return (
        result[_IDX_LOG],  # log text (str)
        _dict_to_update(result[_IDX_STATUS_BADGE]),  # status badge HTML
        result[_IDX_PROGRESS],  # progress bar HTML
        _dict_to_update(result[_IDX_COMPLETED]),  # completed pages card
        _dict_to_update(result[_IDX_FAILED]),  # failed pages card
        _dict_to_update(result[_IDX_FILE_SELECTOR]),  # file dropdown
        result[_IDX_ZIP],  # zip download path
        result[_IDX_INDIVIDUAL],  # individual download
        _dict_to_update(result[_IDX_START_BTN]),  # start button
        result[_IDX_RUN_ID],  # run_id string
        result[_IDX_FILE_STATUS],  # file status HTML
        result[_IDX_MANIFEST],  # upload manifest HTML
        _dict_to_update(result[_IDX_STOP_BTN]),  # stop button
    )


def file_selection_to_gradio(result: tuple) -> tuple:
    """Convert a pdf_manager.on_file_selected() result to Gradio updates.

    Input tuple: (pdf_path, total_pages, page_ranges, full_markdown,
                   slider_dict, download_path)
    Output: same but with slider_dict converted to gr.update().
    """
    return (
        result[0],  # pdf_path
        result[1],  # total_pages
        result[2],  # page_ranges
        result[3],  # full_markdown
        _dict_to_update(result[4]),  # slider → gr.update(maximum=..., value=..., interactive=...)
        result[5],  # download_path
    )
