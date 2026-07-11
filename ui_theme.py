import gradio as gr

custom_css = """
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono&display=swap');

body {
    background-color: #090d16 !important;
    font-family: 'Outfit', sans-serif !important;
}

.gradio-container {
    background-color: #090d16 !important;
    max-width: 96% !important;
    width: 96% !important;
    margin: 0 auto !important;
    padding: 0 24px !important;
}

.sidebar-panel {
    max-width: 320px !important;
    min-width: 280px !important;
}

/* Prevent table text/filenames from overflowing and causing horizontal scroll */
table td, table th {
    word-break: break-all !important;
    white-space: normal !important;
}

.glass-panel {
    background: rgba(17, 24, 39, 0.7) !important;
    backdrop-filter: blur(16px) !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    border-radius: 16px !important;
    box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.5) !important;
    padding: 20px !important;
    margin-bottom: 16px !important;
}

.gradient-title {
    background: linear-gradient(135deg, #818cf8, #3b82f6, #60a5fa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-weight: 800 !important;
    font-size: 2.2rem !important;
    text-align: left;
    margin-bottom: 2px !important;
    line-height: 1.2 !important;
}

.gradient-subtitle {
    color: #9ca3af !important;
    text-align: center;
    font-size: 1.1rem !important;
    margin-bottom: 30px !important;
}

.log-console textarea, .log-console code {
    background-color: #020617 !important;
    color: #38bdf8 !important;
    font-family: 'JetBrains Mono', monospace !important;
    border: 1px solid #1e293b !important;
    font-size: 0.85rem !important;
}

.stat-card {
    background: rgba(30, 41, 59, 0.5) !important;
    border: 1px solid rgba(255, 255, 255, 0.05) !important;
    border-radius: 12px !important;
    padding: 15px !important;
    text-align: center;
}

.stat-value {
    font-size: 1.8rem !important;
    font-weight: 700 !important;
    color: #f3f4f6 !important;
}

.stat-label {
    font-size: 0.85rem !important;
    color: #9ca3af !important;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-top: 5px;
}

.badge-idle { background-color: #1e293b; color: #94a3b8; padding: 4px 10px; border-radius: 9999px; font-weight: 600; font-size: 0.85rem; }
.badge-running { background-color: #1e3a8a; color: #60a5fa; padding: 4px 10px; border-radius: 9999px; font-weight: 600; font-size: 0.85rem; animation: pulse 2s infinite; }
.badge-success { background-color: #064e3b; color: #34d399; padding: 4px 10px; border-radius: 9999px; font-weight: 600; font-size: 0.85rem; }
.badge-stopped { background-color: #7f1d1d; color: #fca5a5; padding: 4px 10px; border-radius: 9999px; font-weight: 600; font-size: 0.85rem; }
.badge-failed { background-color: #7f1d1d; color: #fca5a5; padding: 4px 10px; border-radius: 9999px; font-weight: 600; font-size: 0.85rem; }

@keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.6; }
}

.preview-container {
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    border-radius: 12px !important;
    padding: 15px !important;
    background: rgba(15, 23, 42, 0.3) !important;
    min-height: 350px;
}

/* Height containment: rendered preview scrolls internally */
.preview-scroll .prose,
.preview-scroll .md,
.preview-scroll .markdown-body,
.preview-scroll > div > div {
    max-height: 70vh !important;
    overflow-y: auto !important;
}

/* Height containment: raw markdown code editor */
.raw-md-scroll .cm-editor {
    max-height: 70vh !important;
}

/* Height containment: log viewer capped at 250px */
.log-console .cm-editor {
    max-height: 250px !important;
}

/* Compact download file components */
.compact-download {
    min-height: 0 !important;
}
.compact-download .file-preview,
.compact-download .upload-button,
.compact-download .wrap {
    min-height: 0 !important;
    padding: 6px 10px !important;
}

.status-container {
    padding: 10px 15px !important;
    margin: 0 !important;
    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: center;
}

.section-divider {
    border: none;
    border-top: 1px solid rgba(255, 255, 255, 0.06);
    margin: 8px 0 16px 0;
}

.section-header {
    background: linear-gradient(90deg, rgba(99,102,241,0.15), transparent) !important;
    border-left: 3px solid #818cf8 !important;
    padding: 10px 16px !important;
    border-radius: 0 12px 12px 0 !important;
    margin-bottom: 12px !important;
}

.section-header h3 {
    margin: 0 !important;
    color: #c7d2fe !important;
    font-size: 1.05rem !important;
    font-weight: 600 !important;
}

@media (max-width: 1200px) {
    .gradio-container {
        max-width: 100% !important;
        padding: 0 12px !important;
    }
}

input, textarea, select, .wrap input {
    color: #e2e8f0 !important;
    background-color: #1e293b !important;
}

.code-wrap, .cm-editor, .cm-content {
    background-color: #020617 !important;
    color: #38bdf8 !important;
}

.accordion {
    border-color: rgba(255, 255, 255, 0.06) !important;
}

.file-preview {
    background: rgba(15, 23, 42, 0.5) !important;
}

/* Ensure all file upload/download containers are dark with high contrast text */
.gradio-file,
.upload-container,
[data-testid="file-upload"],
.file-preview,
.compact-download,
.compact-download .file-preview,
.compact-download .upload-button,
.compact-download .wrap {
    background-color: #1e293b !important;
    background: #1e293b !important;
    color: #e2e8f0 !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
}

.gradio-file *,
.upload-container *,
.file-preview *,
.compact-download * {
    color: #e2e8f0 !important;
}

/* File status table scrollable container */
.file-status-wrap {
    max-height: 200px;
    overflow-y: auto;
}

/* 3-Window Dashboard Style */
#pdf-scroll-container,
#raw-scroll-container,
#preview-scroll-container {
    height: 70vh !important;
    max-height: 70vh !important;
    overflow-y: auto !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    border-radius: 12px !important;
    background-color: #020617 !important;
}

#pdf-scroll-container {
    background-color: #000000 !important;
}

#raw-scroll-container {
    font-family: 'JetBrains Mono', monospace !important;
    color: #38bdf8 !important;
    padding: 15px !important;
    white-space: pre-wrap !important;
    font-size: 0.85rem !important;
    line-height: 1.5 !important;
}

#preview-scroll-container {
    background: rgba(15, 23, 42, 0.3) !important;
    padding: 20px !important;
}

/* Styled scrollbars for premium look and WCAG 1.4.11 compliance */
::-webkit-scrollbar {
    width: 8px;
    height: 8px;
}
::-webkit-scrollbar-track {
    background: rgba(255, 255, 255, 0.03);
}
::-webkit-scrollbar-thumb {
    background: rgba(255, 255, 255, 0.25);
    border-radius: 4px;
}
::-webkit-scrollbar-thumb:hover {
    background: rgba(129, 140, 248, 0.6);
}

/* Visible Focus Indicator (WCAG 2.2 SC 2.4.7) */
*:focus-visible {
    outline: 2px solid #818cf8 !important;
    outline-offset: 2px !important;
    box-shadow: 0 0 0 4px rgba(129, 140, 248, 0.4) !important;
}
button:focus-visible, 
input:focus-visible, 
textarea:focus-visible, 
select:focus-visible,
[role="button"]:focus-visible,
[role="combobox"]:focus-visible,
[contenteditable="true"]:focus-visible,
.label-wrap:focus-visible,
.accordion:focus-visible,
.tab-nav button:focus-visible {
    outline: 2px solid #818cf8 !important;
    outline-offset: 2px !important;
    box-shadow: 0 0 0 4px rgba(129, 140, 248, 0.4) !important;
}

/* RAG Analysis Section */
.analysis-chatbot {
    background: rgba(15, 23, 42, 0.5) !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    border-radius: 12px !important;
}

.analysis-chatbot .message {
    font-size: 0.95rem !important;
    line-height: 1.6 !important;
}

.analysis-chatbot .bot {
    background: rgba(30, 41, 59, 0.7) !important;
}

.analysis-chatbot .user {
    background: rgba(99, 102, 241, 0.15) !important;
}

.mode-hint {
    font-size: 0.85rem !important;
    color: #94a3b8 !important;
}

.mode-hint p {
    margin: 0 !important;
    font-style: italic;
}
"""

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
