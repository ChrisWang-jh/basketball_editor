"""Reusable clip library and ordered stitching workspace."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any

from .stitch import discover_attack_clips, stitch_clips
from .theme import blocks, header, page_heading


@dataclass
class ClipWorkspace:
    refresh: Callable
    inputs: list[Any]
    outputs: list[Any]
    edit_button: Any = None

    def refresh_on(self, trigger: Callable) -> Any:
        return trigger(
            self.refresh,
            inputs=self.inputs,
            outputs=self.outputs,
            concurrency_id="basketedit-clips",
            concurrency_limit=1,
            show_progress="hidden",
        )


def create_clip_workspace(
    output_root: Path, gr: Any, *, allow_edit=False
) -> ClipWorkspace:
    """Render a workspace whose selections survive video and mode changes."""
    output_root = output_root.expanduser().resolve()
    serial = {"concurrency_id": "basketedit-clips", "concurrency_limit": 1}

    def discover():
        if not output_root.exists():
            return []
        return [str(p) for p in discover_attack_clips(output_root)]

    def label(path):
        relative = Path(path).relative_to(output_root)
        source = relative.parent.name or "片段"
        match = re.fullmatch(r"(.+)_(\d{8})_(\d{6})(?:_\d{6})?", source)
        stamp = ""
        if match:
            source, date, time = match.groups()
            stamp = f" · {date[4:6]}-{date[6:8]} {time[:2]}:{time[2:4]}"
        return f"{source} · 片段 {relative.stem.removeprefix('attack_')}{stamp}"

    def summary(selected):
        if not selected:
            return "还没有选择片段。\n\n在左侧预览喜欢的片段，点击「加入集锦」。"
        return f"**已选 {len(selected)} 段** · 按下方顺序播放"

    def values(clips, selected, current, focus=None, *, update_gallery=True):
        selected = [p for p in dict.fromkeys(selected or []) if p in clips]
        current = current if current in clips else next(iter(clips), None)
        focus = focus if focus in selected else next(iter(selected), None)
        in_queue = current in selected
        index = selected.index(focus) if focus else -1
        return (
            clips,
            selected,
            current,
            gr.update(
                choices=[(label(p), p) for p in clips],
                value=current,
                interactive=bool(clips),
            ),
            gr.update(
                value=[
                    (p, f"{'✓ ' if p in selected else ''}{label(p)}") for p in clips
                ],
                selected_index=clips.index(current) if current else None,
            )
            if update_gallery
            else gr.skip(),
            gr.update(value=current, visible=bool(current)),
            f"共 **{len(clips)}** 个已裁剪片段 · 已选 **{len(selected)}** 段。"
            if clips
            else "暂无已裁剪片段",
            gr.update(
                value="移出集锦" if in_queue else "加入集锦",
                interactive=bool(current),
            ),
            summary(selected),
            gr.update(
                choices=[(f"{i}. {label(p)}", p) for i, p in enumerate(selected, 1)],
                value=focus,
                interactive=bool(selected),
                visible=bool(selected),
            ),
            gr.update(interactive=index > 0),
            gr.update(interactive=0 <= index < len(selected) - 1),
            gr.update(interactive=bool(focus)),
            gr.update(interactive=bool(selected)),
            gr.update(interactive=bool(selected)),
            gr.update(visible=bool(selected)),
            gr.update(visible=not clips),
            gr.update(visible=bool(clips)),
        )

    def refresh(selected, current, focus):
        return values(discover(), selected, current, focus)

    def browse(current, clips, selected, focus):
        return values(clips, selected, current, focus)

    def choose_thumbnail(clips, selected, focus, evt):
        index = int(evt.index)
        current = clips[index] if 0 <= index < len(clips) else None
        # Updating Gallery.selected_index emits another select event in Gradio.
        # Keep the user's browser selection; writing it back races rapid clicks.
        return values(clips, selected, current, focus, update_gallery=False)

    choose_thumbnail.__annotations__["evt"] = gr.SelectData

    def toggle(clips, selected, current, focus):
        selected = list(selected)
        if current in selected:
            selected.remove(current)
        elif current in clips:
            selected.append(current)
            focus = current
        return values(clips, selected, current, focus)

    def reorder(action, clips, selected, current, focus):
        selected = list(selected)
        if action == "clear":
            selected = []
        elif focus in selected:
            index = selected.index(focus)
            if action == "remove":
                selected.pop(index)
                focus = selected[min(index, len(selected) - 1)] if selected else None
            else:
                destination = index + (-1 if action == "up" else 1)
                if 0 <= destination < len(selected):
                    selected[index], selected[destination] = (
                        selected[destination],
                        selected[index],
                    )
        return values(clips, selected, current, focus)

    def concatenate(selected, progress=gr.Progress()):
        yield (
            "正在拼接，稍后即可预览和下载。",
            None,
            None,
            gr.update(visible=False),
            gr.update(value="正在拼接…", interactive=False),
        )
        try:
            if not selected:
                raise ValueError("请先选择需要拼接的片段。")
            available = set(discover())
            if any(p not in available for p in selected):
                raise ValueError("部分片段已不存在，请刷新片段列表后重试。")
            # Capture the order before the potentially long export.
            output = stitch_clips(
                [Path(p) for p in list(selected)], output_root, progress
            )
            yield (
                f"**拼接完成** · {len(selected)} 个片段，已按选择顺序生成集锦。",
                str(output),
                str(output),
                gr.update(visible=True),
                gr.update(value="生成集锦", interactive=bool(selected)),
            )
        except Exception as exc:
            yield (
                f"**拼接未完成** · {exc}",
                None,
                None,
                gr.update(visible=False),
                gr.update(value="生成集锦", interactive=bool(selected)),
            )

    initial = discover()
    current = next(iter(initial), None)
    clips_state = gr.State(initial)
    selected_state = gr.State([])
    current_state = gr.State(current)
    edit_button = None
    with gr.Row(equal_height=False, elem_classes=["be-workspace-row"]):
        with gr.Column(
            scale=6, min_width=340, elem_classes=["be-panel"], elem_id="be-library"
        ):
            with gr.Row(elem_id="be-library-heading"):
                gr.HTML(
                    '<div class="be-section">片段库<small>点选片段，先看看内容。</small></div>'
                )
                refresh_button = gr.Button("刷新片段", size="sm", scale=0, min_width=95)
            library_status = gr.Markdown(
                values(initial, [], current)[6],
                elem_id="be-library-status",
                elem_classes=["be-subtle"],
            )
            with gr.Column(
                visible=not bool(initial), elem_id="be-library-empty"
            ) as empty_panel:
                gr.HTML(
                    '<div class="be-empty"><svg width="48" height="40" viewBox="0 0 48 40" fill="none" stroke="currentColor" stroke-width="1.5" aria-hidden="true">'
                    '<rect x="3" y="5" width="42" height="30" rx="6"/><path d="m20 14 12 6-12 6z"/></svg>'
                    "<h3>还没有精彩片段</h3><p>先裁剪一段录像，生成的片段就会出现在这里。</p></div>"
                )
                if allow_edit:
                    edit_button = gr.Button("← 返回裁剪")
            with gr.Column(visible=bool(initial)) as library_body:
                gallery = gr.Gallery(
                    value=[(p, label(p)) for p in initial],
                    label="点击片段预览",
                    show_label=False,
                    container=False,
                    columns=3,
                    height="auto",
                    object_fit="cover",
                    allow_preview=False,
                    selected_index=0 if initial else None,
                    interactive=False,
                    show_download_button=False,
                    show_fullscreen_button=False,
                    elem_id="be-clip-gallery",
                )
                with gr.Accordion(
                    "按名称查找片段", open=False, elem_classes=["be-accordion"]
                ):
                    clip_selector = gr.Dropdown(
                        choices=[(label(p), p) for p in initial],
                        value=current,
                        label="搜索片段",
                        interactive=bool(initial),
                    )
                current_video = gr.Video(
                    value=current,
                    label="当前片段",
                    interactive=False,
                    height=260,
                    visible=bool(current),
                    elem_id="be-clip-preview",
                )
                toggle_button = gr.Button("加入集锦", interactive=bool(current))

        with gr.Column(
            scale=4, min_width=290, elem_classes=["be-panel"], elem_id="be-queue-panel"
        ):
            gr.HTML(
                '<div class="be-section">播放顺序<small>选中一段，可以上移、下移或移除。</small></div>'
            )
            selected_markdown = gr.Markdown(summary([]), elem_id="be-selected-clips")
            queue_selector = gr.Radio(
                choices=[],
                value=None,
                label="已选片段",
                show_label=False,
                container=False,
                interactive=False,
                visible=False,
                elem_id="be-clip-queue",
            )
            with gr.Row(
                visible=False, elem_classes=["be-frame-controls"]
            ) as queue_controls:
                up_button = gr.Button(
                    "↑ 上移", size="sm", interactive=False, min_width=55
                )
                down_button = gr.Button(
                    "↓ 下移", size="sm", interactive=False, min_width=55
                )
                remove_button = gr.Button(
                    "移除", size="sm", interactive=False, min_width=50
                )
                clear_button = gr.Button(
                    "清空", size="sm", interactive=False, min_width=50
                )
            stitch_button = gr.Button("生成集锦", variant="primary", interactive=False)
            status = gr.Markdown(
                "选好片段后，一键生成并下载。",
                elem_id="be-stitch-status",
                elem_classes=["be-subtle"],
            )

    with gr.Column(
        visible=False, elem_classes=["be-panel"], elem_id="be-stitch-result"
    ) as result_panel:
        gr.HTML(
            '<div class="be-section">已生成的集锦<small>调整选择或顺序后，可以重新生成。</small></div>'
        )
        result_video = gr.Video(label="集锦预览", interactive=False, height=360)
        result_file = gr.DownloadButton("下载集锦", value=None, variant="primary")

    outputs = [
        clips_state,
        selected_state,
        current_state,
        clip_selector,
        gallery,
        current_video,
        library_status,
        toggle_button,
        selected_markdown,
        queue_selector,
        up_button,
        down_button,
        remove_button,
        clear_button,
        stitch_button,
        queue_controls,
        empty_panel,
        library_body,
    ]
    state_inputs = [clips_state, selected_state, current_state, queue_selector]
    workspace = ClipWorkspace(
        refresh, [selected_state, current_state, queue_selector], outputs, edit_button
    )
    workspace.refresh_on(refresh_button.click)
    clip_selector.input(
        browse,
        [clip_selector, clips_state, selected_state, queue_selector],
        outputs,
        **serial,
    )
    gallery.select(
        choose_thumbnail,
        [clips_state, selected_state, queue_selector],
        outputs,
        trigger_mode="always_last",
        show_progress="hidden",
        **serial,
    )
    toggle_button.click(toggle, state_inputs, outputs, **serial)
    queue_selector.input(
        lambda clips, selected, current, focus: values(clips, selected, current, focus),
        state_inputs,
        outputs,
        **serial,
    )
    for button, action in (
        (up_button, "up"),
        (down_button, "down"),
        (remove_button, "remove"),
        (clear_button, "clear"),
    ):
        button.click(
            lambda clips, selected, current, focus, action=action: reorder(
                action, clips, selected, current, focus
            ),
            state_inputs,
            outputs,
            **serial,
        )
    stitch_button.click(
        concatenate,
        selected_state,
        [status, result_video, result_file, result_panel, stitch_button],
        **serial,
    )
    return workspace


def create_stitch_interface(output_root: Path, gr: Any) -> Any:
    """Standalone entry shares the same library as the main editor."""
    with blocks(gr, "BasketEdit · 精彩拼接") as demo:
        gr.HTML(header())
        gr.HTML(page_heading("拼接集锦", "挑选已有片段，排好播放顺序，合成一支集锦。"))
        workspace = create_clip_workspace(output_root, gr)
        workspace.refresh_on(demo.load)
    return demo
