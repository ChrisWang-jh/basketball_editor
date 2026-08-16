"""Gradio 视频标注界面。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .models import VideoInfo
from .pipeline import run_pipeline
from .sam2 import Sam2Preview
from .video import (
    annotation_summary,
    draw_annotations_and_masks,
    read_frame,
)


def create_interface(
    info: VideoInfo,
    checkpoint: Path,
    model_config: str,
    device: str,
    sam2_repo: Path | None,
    np: Any,
    cv2: Any,
    gr: Any,
) -> Any:
    """创建标注 UI；点击提示点后即时显示当前帧的 SAM2 掩码。"""
    initial_frame = read_frame(info, 0, cv2)
    preview = Sam2Preview(checkpoint, model_config, device, sam2_repo, np)

    def render_frame(frame_idx: float, annotations: list[dict[str, Any]]) -> Any:
        frame_idx = round(frame_idx)
        frame = read_frame(info, frame_idx, cv2)
        masks = preview.predict(frame, frame_idx, annotations)
        return draw_annotations_and_masks(frame, frame_idx, annotations, masks, np, cv2)

    def change_frame(
        frame_idx: float, annotations: list[dict[str, Any]]
    ) -> tuple[Any, str]:
        return render_frame(frame_idx, annotations), annotation_summary(annotations)

    def click_image(
        target: str,
        point_type: str,
        annotations: list[dict[str, Any]],
        frame_idx: float,
        evt: Any,
    ) -> tuple[Any, list[dict[str, Any]], str]:
        x, y = evt.index
        frame_idx = round(frame_idx)
        updated = [
            *annotations,
            {
                "target": target,
                "frame_idx": frame_idx,
                "x": float(x),
                "y": float(y),
                "label": 1 if point_type == "positive" else 0,
            },
        ]
        return (
            render_frame(frame_idx, updated),
            updated,
            annotation_summary(updated),
        )

    # Gradio 通过具体的事件类型注解识别这个无需放入 inputs 的参数。
    # gr 是运行时注入的，因此在函数定义后绑定真实类型。
    click_image.__annotations__["evt"] = gr.SelectData

    def undo(
        annotations: list[dict[str, Any]], frame_idx: float
    ) -> tuple[Any, list[dict[str, Any]], str]:
        updated = annotations[:-1] if annotations else []
        return (
            render_frame(frame_idx, updated),
            updated,
            annotation_summary(updated),
        )

    def clear_all(frame_idx: float) -> tuple[Any, list[dict[str, Any]], str]:
        return read_frame(info, frame_idx, cv2), [], annotation_summary([])

    def start_analysis(
        annotations: list[dict[str, Any]],
        progress: Any = gr.Progress(),  # noqa: B008
    ) -> tuple[str, list[str]]:
        try:
            preview.release()
            return run_pipeline(
                info,
                annotations,
                checkpoint,
                model_config,
                device,
                sam2_repo,
                np,
                cv2,
                progress,
            )
        except Exception as exc:  # noqa: BLE001 - UI must return failures to the page.
            return f"### Analysis failed\n\n{type(exc).__name__}: {exc}", []

    with gr.Blocks(title="Basketball Attack Clip Extractor") as demo:
        gr.Markdown(
            f"# Basketball Attack Clip Extractor\n\n"
            f"video: `{info.path}`　FPS: {info.fps:.3f}　"
            f"frame count: {info.frame_count}　duration: {info.duration:.1f} s\n\n"
            "Select a target and frame, then click directly on the image. "
            "SAM2 will preview the segmentation mask on annotated frames. "
            "Add ball points on 3-10 clear frames."
        )
        annotation_state = gr.State([])
        with gr.Row():
            with gr.Column(scale=4):
                image = gr.Image(
                    value=initial_frame,
                    label="Click video frame",
                    interactive=False,
                )
                frame_slider = gr.Slider(
                    0,
                    info.frame_count - 1,
                    value=0,
                    step=1,
                    label="current frame",
                )
            with gr.Column(scale=2):
                target_selector = gr.Radio(
                    ["player", "hoop", "ball"], value="player", label="target"
                )
                point_type = gr.Radio(
                    ["positive", "negative"], value="positive", label="point type"
                )
                stats = gr.Markdown(annotation_summary([]))
                with gr.Row():
                    undo_button = gr.Button("undo last point")
                    clear_button = gr.Button("clear all points")
                start_button = gr.Button(
                    "Start tracking and clipping", variant="primary"
                )
        run_status = gr.Markdown("Analysis has not started.")
        result_files = gr.File(label="result files", file_count="multiple")

        frame_slider.change(
            change_frame,
            inputs=[frame_slider, annotation_state],
            outputs=[image, stats],
        )
        image.select(
            click_image,
            inputs=[target_selector, point_type, annotation_state, frame_slider],
            outputs=[image, annotation_state, stats],
        )
        undo_button.click(
            undo,
            inputs=[annotation_state, frame_slider],
            outputs=[image, annotation_state, stats],
        )
        clear_button.click(
            clear_all,
            inputs=[frame_slider],
            outputs=[image, annotation_state, stats],
        )
        start_button.click(
            start_analysis,
            inputs=[annotation_state],
            outputs=[run_status, result_files],
        )
    return demo
