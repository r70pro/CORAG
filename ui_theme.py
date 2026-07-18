import os

import gradio as gr

# Load custom CSS from external file
CSS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "theme.css")
with open(CSS_FILE, encoding="utf-8") as f:
    custom_css = f.read()

dark_theme = gr.themes.Base(
    primary_hue="indigo",
    secondary_hue="blue",
    neutral_hue="slate",
).set(
    body_background_fill="#090d16",
    body_background_fill_dark="#090d16",
    block_background_fill="rgba(17, 24, 39, 0.7)",
    block_background_fill_dark="rgba(17, 24, 39, 0.7)",
    block_border_color="rgba(255, 255, 255, 0.08)",
    block_border_color_dark="rgba(255, 255, 255, 0.08)",
    block_label_text_color="#e2e8f0",
    block_label_text_color_dark="#e2e8f0",
    block_title_text_color="#e2e8f0",
    block_title_text_color_dark="#e2e8f0",
    body_text_color="#e2e8f0",
    body_text_color_dark="#e2e8f0",
    body_text_color_subdued="#9ca3af",
    body_text_color_subdued_dark="#9ca3af",
    input_background_fill="#1e293b",
    input_background_fill_dark="#1e293b",
    input_border_color="rgba(255, 255, 255, 0.1)",
    input_border_color_dark="rgba(255, 255, 255, 0.1)",
    button_primary_background_fill="linear-gradient(135deg, #6366f1, #3b82f6)",
    button_primary_background_fill_dark="linear-gradient(135deg, #6366f1, #3b82f6)",
    button_primary_text_color="#ffffff",
    button_primary_text_color_dark="#ffffff",
    button_secondary_background_fill="rgba(30, 41, 59, 0.8)",
    button_secondary_background_fill_dark="rgba(30, 41, 59, 0.8)",
    button_secondary_text_color="#e2e8f0",
    button_secondary_text_color_dark="#e2e8f0",
    border_color_accent="rgba(99, 102, 241, 0.5)",
    border_color_accent_dark="rgba(99, 102, 241, 0.5)",
    shadow_drop="0 4px 24px rgba(0,0,0,0.4)",
    shadow_drop_lg="0 8px 32px rgba(0,0,0,0.5)",
)
