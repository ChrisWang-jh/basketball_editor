"""Shared soft surfaces and controls for the two workspaces."""

from __future__ import annotations

import inspect
from html import escape
from typing import Any

CSS = r"""
:root { --be-paper:#faf7f0; --be-ink:#443f37; --be-yellow:#f2dcaa; --be-muted:#827b6d; --be-line:#e8e1d5; --be-surface:#fffefb; }
body, .gradio-container { background:var(--be-paper) !important; color:var(--be-ink) !important; }
.gradio-container { width:100% !important; max-width:1120px !important; margin:auto !important; padding:clamp(18px,2.5vw,30px) clamp(15px,2.6vw,30px) 30px !important;
 font-family:'Trebuchet MS','PingFang SC','Microsoft YaHei',sans-serif !important; }
.gradio-container main.app, .gradio-container .html-container { padding:0 !important; }
.gradio-container footer:not(.be-footer) { display:none !important; }
.gradio-container .prose, .gradio-container label, .gradio-container input, .gradio-container textarea { color:var(--be-ink) !important; }
.gradio-container .prose p { color:var(--be-muted); line-height:1.7; }
.be-header { display:flex; justify-content:space-between; gap:20px; align-items:center; margin-bottom:20px; }
.be-brand { display:flex; align-items:center; gap:10px; font-size:25px; font-weight:700; letter-spacing:-.7px; }
.be-logo { width:42px; height:42px; display:grid; place-items:center; color:#7a6540; background:#f4e7c8; border-radius:15px; }
.be-tag { color:var(--be-muted); font-size:12px; }
#be-pager { align-items:center !important; flex-wrap:nowrap !important; gap:14px !important; max-width:380px; margin:0 auto 10px !important; }
#be-pager button { border-radius:99px !important; min-height:40px; font-size:13px; padding:8px 16px; }
#be-page-count { flex:1; min-width:60px !important; }
.be-page-count { display:flex; justify-content:center; align-items:baseline; gap:6px; color:#a69b86; font-size:12px; font-variant-numeric:tabular-nums; }
.be-page-count b { color:#6b6253; font-size:19px; font-weight:600; }
.be-page-heading { padding:8px 0 18px; }
.be-page-heading h1 { font-size:28px; font-weight:700; line-height:1.4; margin:0 0 8px; letter-spacing:-.5px; }
.be-page-heading p { margin:0; color:var(--be-muted); font-size:13px; line-height:1.8; }
.be-workspace-row { gap:24px !important; align-items:flex-start !important; }
.be-panel { background:var(--be-surface) !important; border:1px solid var(--be-line) !important; border-radius:22px !important; padding:22px !important;
 box-shadow:0 8px 28px #78684a08 !important; overflow:visible !important; gap:14px !important; }
.be-panel .block { background:transparent !important; }
.be-section { font-size:16px; font-weight:700; margin:0; line-height:1.6; }
.be-section small { display:block; color:var(--be-muted); font-size:12px; font-weight:400; margin-top:5px; }
.gradio-container button { font-family:inherit !important; }
.gradio-container button.primary, .gradio-container button.secondary { border:1px solid var(--be-line) !important; border-radius:12px !important;
 background:#fbf9f4 !important; color:var(--be-ink) !important; box-shadow:none !important; font-weight:600 !important; transition:background .18s,border-color .18s !important; }
.gradio-container button.primary { background:var(--be-yellow) !important; border-color:#e9d29d !important; min-height:46px; }
.gradio-container button.secondary:hover:not(:disabled) { background:#f1ece2 !important; border-color:#d9cdb8 !important; }
.gradio-container button.primary:hover:not(:disabled) { background:#eacf96 !important; }
#be-stitcher button.primary { background:#e0e9d9 !important; border-color:#d3dec9 !important; }
#be-stitcher button.primary:hover:not(:disabled) { background:#d3e0c9 !important; }
.gradio-container button:disabled { opacity:.42 !important; }
.gradio-container button:focus-visible, .gradio-container input:focus-visible, .gradio-container label:focus-within { outline:2px solid #9a9776 !important; outline-offset:3px; }
.gradio-container .form { background:transparent !important; border-color:var(--be-line) !important; border-radius:12px !important; }
.gradio-container .block > .wrap { border-color:var(--be-line); }
.be-subtle .prose, .be-subtle p { font-size:12px !important; line-height:1.6 !important; }
.be-accordion { border:1px solid var(--be-line) !important; border-radius:12px !important; background:transparent !important; padding:0 !important; }
.be-accordion > .label-wrap { padding:12px !important; }
.be-accordion > div:not(.wrap) { padding:0 12px 12px; }
#be-frame { border:1px solid var(--be-line) !important; border-radius:14px !important; overflow:hidden !important; background:#f0ede6 !important; }
#be-frame img { object-fit:contain !important; }
#be-frame-caption p { font:12px monospace; color:var(--be-muted); }
#be-targets .wrap { gap:7px; }
#be-targets .wrap label { flex:1; justify-content:center; border:1px solid var(--be-line) !important; border-radius:12px !important; padding:10px 7px; margin:0; background:#fbf9f4 !important; }
#be-targets .wrap label.selected { border-color:#d7d9bd !important; background:#f0f0df !important; }
#be-targets input[type=radio] { position:absolute; opacity:0; width:1px; height:1px; }
#be-operation .wrap { gap:7px; }
#be-operation .wrap label { font-size:12px; padding:8px; border:1px solid var(--be-line); border-radius:10px; background:#fbf9f4; }
#be-stats { border-top:1px solid var(--be-line); border-bottom:1px solid var(--be-line); padding:12px 0; }
.be-readiness { display:flex; justify-content:space-between; font-size:12px; color:var(--be-muted); margin-bottom:7px; }
.be-target-status { display:flex; align-items:center; gap:8px; padding:6px 0; font-size:12px; }
.be-target-status strong { margin-left:auto; color:#a29887; font-size:11px; font-weight:400; }
.be-target-status.is-ready strong { color:#718362; }
.be-target-dot { width:7px; height:7px; border-radius:50%; background:#dfd8ca; }
.is-ready .be-target-dot { background:#9cae8b; }
.be-note { font-size:12px; line-height:1.8; color:var(--be-muted); margin:0; }
.be-frame-controls { gap:6px !important; }
.be-frame-controls button { min-height:32px !important; font-size:12px !important; padding:6px !important; }
#be-results, #be-stitch-result { margin-top:24px; }
#be-results .be-section, #be-stitch-result .be-section { margin-bottom:6px; }
#be-library-heading { align-items:center; gap:10px !important; }
#be-library-heading button { min-height:32px; padding:6px 12px; }
#be-clip-gallery { border:0 !important; background:transparent !important; }
#be-clip-gallery .grid-wrap { min-height:0 !important; max-height:240px !important; padding:3px; overflow-y:auto; }
#be-clip-gallery .grid-container { grid-auto-rows:auto; grid-template-rows:none; }
#be-clip-gallery .thumbnail-item { border-radius:12px !important; overflow:hidden; height:auto !important; aspect-ratio:16/9; }
#be-clip-gallery .thumbnail-item svg { width:32px; height:32px; padding:8px; border-radius:50%; background:#faf7f0; opacity:.85; }
#be-clip-gallery .caption-label { font-size:11px; overflow-wrap:anywhere; }
#be-selected-clips { padding:3px 0; }
#be-selected-clips p { font-size:13px; }
#be-clip-queue .wrap { flex-direction:column; gap:8px; }
#be-clip-queue .wrap label { margin:0; border:1px solid var(--be-line); border-radius:12px; padding:12px; font-size:12px; line-height:1.6; overflow-wrap:anywhere; }
#be-clip-queue .wrap label:not(.selected) { background:#fbfaf6 !important; }
#be-clip-queue .wrap label.selected { background:#eef2e8 !important; border-color:#d1dbc6 !important; }
.be-empty { text-align:center; border:1px dashed #ded7ca; border-radius:16px; padding:38px 18px; color:#aaa08e; }
.be-empty svg { display:block; margin:0 auto 14px; }
.be-empty h3 { color:#736957; font-size:15px; font-weight:600; margin:0 0 8px; }
.be-empty p { font-size:12px; line-height:1.8; margin:0; }
.be-footer { padding-top:18px; margin-top:24px; display:flex; justify-content:space-between; gap:10px; color:#a29887; font-size:11px; }
#be-editor, #be-stitcher { animation:be-page-in .2s ease-out; }
@keyframes be-page-in { from { opacity:.5; transform:translateY(5px); } to { opacity:1; transform:none; } }
@media (max-width:760px) {
 .be-header { margin-bottom:16px; }.be-brand { font-size:23px; }.be-logo { width:38px; height:38px; }.be-tag { display:none; }
 #be-pager { gap:12px !important; }#be-pager button { min-width:96px !important; padding:8px 12px; }
 .be-page-heading { padding:6px 0 14px; }.be-page-heading h1 { font-size:25px; }.be-page-heading p { font-size:12px; }
 .be-workspace-row { gap:18px !important; }.be-panel { padding:16px !important; min-width:0 !important; flex-basis:100% !important; border-radius:18px !important; }
 #be-frame { height:auto !important; min-height:0 !important; }
 #be-frame .image-container { height:auto !important; min-height:0 !important; aspect-ratio:16/9; }
 #be-clip-gallery .grid-container { grid-template-columns:repeat(2,minmax(0,1fr)) !important; }
 #be-clip-preview { height:auto !important; }#be-clip-preview .video-container { height:auto !important; aspect-ratio:16/9; }
 .be-footer { flex-wrap:wrap; font-size:10px; }
}
@media (prefers-reduced-motion:reduce) { #be-editor, #be-stitcher { animation:none; }.gradio-container button { transition:none !important; } }
"""

LOGO = '<svg width="32" height="32" viewBox="0 0 40 40" fill="none" stroke="currentColor" stroke-width="1.8" aria-hidden="true"><circle cx="20" cy="20" r="16"/><path d="M4 20h32M20 4v32M9 8c15 8 15 16 0 24M31 8C16 16 16 24 31 32"/></svg>'


def header(title: str = "BasketEdit", tag: str = "篮球视频工作台") -> str:
    return f'<header class="be-header"><div class="be-brand"><span class="be-logo">{LOGO}</span>{title}</div><span class="be-tag">{tag}</span></header>'


def page_heading(title: str, description: str) -> str:
    return f'<div class="be-page-heading"><h1>{escape(title)}</h1><p>{escape(description)}</p></div>'


def blocks(gr: Any, title: str) -> Any:
    theme = gr.themes.Base(
        primary_hue="amber",
        neutral_hue="stone",
        font=["Trebuchet MS", "Arial", "sans-serif"],
    )
    theme = theme.set(
        body_background_fill="#faf7f0",
        body_background_fill_dark="#faf7f0",
        body_text_color="#443f37",
        body_text_color_dark="#443f37",
        background_fill_primary="#fffefb",
        background_fill_primary_dark="#fffefb",
        background_fill_secondary="#f8f5ee",
        background_fill_secondary_dark="#f8f5ee",
        block_background_fill="#fffefb",
        block_background_fill_dark="#fffefb",
        panel_background_fill="#fffefb",
        panel_background_fill_dark="#fffefb",
        input_background_fill="#f8f5ee",
        input_background_fill_dark="#f8f5ee",
        block_label_text_color="#827b6d",
        block_label_text_color_dark="#827b6d",
        body_text_color_subdued="#827b6d",
        body_text_color_subdued_dark="#827b6d",
        input_border_color="#e8e1d5",
        input_border_color_dark="#e8e1d5",
        slider_color="#c5b17d",
        slider_color_dark="#c5b17d",
        block_radius="12px",
        input_radius="10px",
    )
    kwargs = {"title": title}
    # Gradio 6 moved theme/css from Blocks to launch; support both API shapes.
    if "css" in inspect.signature(gr.Blocks.__init__).parameters:
        kwargs.update(css=CSS, theme=theme)
    demo = gr.Blocks(**kwargs)
    demo.basketedit_launch_kwargs = (
        {} if "css" in kwargs else {"css": CSS, "theme": theme}
    )
    return demo
