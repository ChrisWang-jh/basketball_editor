"""Attack 片段浏览、选择和拼接界面。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .stitch import discover_attack_clips, stitch_clips


def create_stitch_interface(output_root: Path, gr: Any) -> Any:
    """创建可按选择顺序拼接 attack 片段的 Gradio UI。"""
    output_root = output_root.expanduser().resolve()
    initial_clips = [str(path) for path in discover_attack_clips(output_root)]

    def selected_summary(selected: list[str]) -> str:
        if not selected:
            return "No clips selected. Clips will be stitched in selection order."
        lines = ["### Selected clips (stitch order)"]
        for index, path in enumerate(selected, 1):
            lines.append(f"{index}. `{Path(path).relative_to(output_root).as_posix()}`")
        return "\n\n".join(lines)

    def view(
        index: int, clips: list[str], selected: list[str]
    ) -> tuple[Any, str, Any, Any, Any]:
        if not clips:
            return (
                None,
                "No `attack*` videos were found recursively below the output directory.",
                gr.update(value="Select current", interactive=False),
                gr.update(interactive=False),
                gr.update(interactive=False),
            )
        index = int(index) % len(clips)
        current = clips[index]
        relative = Path(current).relative_to(output_root).as_posix()
        is_selected = current in selected
        return (
            current,
            (
                f"**Clip {index + 1}/{len(clips)}:** `{relative}`  \n"
                f"Status: **{'selected' if is_selected else 'not selected'}**"
            ),
            gr.update(
                value="Remove current" if is_selected else "Select current",
                interactive=True,
                variant="secondary" if is_selected else "primary",
            ),
            gr.update(interactive=True),
            gr.update(interactive=True),
        )

    def navigate(
        offset: int, index: int, clips: list[str], selected: list[str]
    ) -> tuple[int, Any, str, Any, Any, Any]:
        next_index = (int(index) + offset) % len(clips) if clips else 0
        return (next_index, *view(next_index, clips, selected))

    def toggle(
        index: int, clips: list[str], selected: list[str]
    ) -> tuple[list[str], str, Any, str]:
        if not clips:
            return selected, selected_summary(selected), gr.update(), "No clips found."
        current = clips[int(index) % len(clips)]
        updated = list(selected)
        if current in updated:
            updated.remove(current)
        else:
            updated.append(current)
        _, metadata, button, _, _ = view(index, clips, updated)
        return updated, selected_summary(updated), button, metadata

    def refresh(
        selected: list[str],
    ) -> tuple[list[str], int, list[str], Any, str, Any, Any, Any, str, str]:
        clips = [str(path) for path in discover_attack_clips(output_root)]
        clip_set = set(clips)
        selected = [path for path in selected if path in clip_set]
        video, metadata, toggle_button, previous, following = view(0, clips, selected)
        return (
            clips,
            0,
            selected,
            video,
            metadata,
            toggle_button,
            previous,
            following,
            selected_summary(selected),
            f"Found {len(clips)} attack clips.",
        )

    def concatenate(
        selected: list[str],
        progress: Any = gr.Progress(),  # noqa: B008
    ) -> tuple[str, Any, Any]:
        try:
            output = stitch_clips(
                [Path(path) for path in selected], output_root, progress
            )
            return (
                f"### Stitch complete\n\n{len(selected)} clips → `{output}`",
                str(output),
                str(output),
            )
        except Exception as exc:  # noqa: BLE001 - surface failures in the UI.
            return f"### Stitch failed\n\n{type(exc).__name__}: {exc}", None, None

    initial_video, initial_meta, _, _, _ = view(0, initial_clips, [])
    has_initial_clips = bool(initial_clips)
    with gr.Blocks(title="Attack Clip Stitcher") as demo:
        gr.Markdown(
            "# Attack Clip Stitcher\n\n"
            f"Recursively browsing `attack*` videos below `{output_root}`. "
            "Click clips in the desired final order."
        )
        clips_state = gr.State(initial_clips)
        index_state = gr.State(0)
        selected_state = gr.State([])
        current_video = gr.Video(
            value=initial_video, label="Current attack clip", interactive=False
        )
        clip_meta = gr.Markdown(initial_meta)
        with gr.Row():
            previous_button = gr.Button("← Previous", interactive=has_initial_clips)
            toggle_button = gr.Button(
                "Select current", variant="primary", interactive=has_initial_clips
            )
            next_button = gr.Button("Next →", interactive=has_initial_clips)
        selected_markdown = gr.Markdown(selected_summary([]))
        with gr.Row():
            refresh_button = gr.Button("Refresh clip list")
            stitch_button = gr.Button("Stitch selected clips", variant="primary")
        status = gr.Markdown(f"Found {len(initial_clips)} attack clips.")
        result_video = gr.Video(label="Stitched result", interactive=False)
        result_file = gr.File(label="Download stitched video")

        navigation_outputs = [
            index_state,
            current_video,
            clip_meta,
            toggle_button,
            previous_button,
            next_button,
        ]
        previous_button.click(
            lambda index, clips, selected: navigate(-1, index, clips, selected),
            inputs=[index_state, clips_state, selected_state],
            outputs=navigation_outputs,
        )
        next_button.click(
            lambda index, clips, selected: navigate(1, index, clips, selected),
            inputs=[index_state, clips_state, selected_state],
            outputs=navigation_outputs,
        )
        toggle_button.click(
            toggle,
            inputs=[index_state, clips_state, selected_state],
            outputs=[selected_state, selected_markdown, toggle_button, clip_meta],
        )
        refresh_button.click(
            refresh,
            inputs=[selected_state],
            outputs=[
                clips_state,
                index_state,
                selected_state,
                current_video,
                clip_meta,
                toggle_button,
                previous_button,
                next_button,
                selected_markdown,
                status,
            ],
        )
        stitch_button.click(
            concatenate,
            inputs=[selected_state],
            outputs=[status, result_video, result_file],
        )
    return demo
