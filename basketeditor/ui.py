"""Gradio 视频标注界面。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .pipeline import run_pipeline
from .sam2 import Sam2Preview
from .video import (
    annotation_summary,
    draw_annotations_and_masks,
    read_frame,
    read_video_info,
)


def create_interface(
    video_paths: list[Path],
    video_root: Path,
    checkpoint: Path,
    model_config: str,
    device: str,
    sam2_repo: Path | None,
    np: Any,
    cv2: Any,
    gr: Any,
) -> Any:
    """创建可切换多个视频的标注 UI。"""
    if not video_paths:
        raise ValueError("At least one video is required.")
    paths = [str(path.resolve()) for path in video_paths]
    path_indexes = {path: index for index, path in enumerate(paths)}
    info_cache: dict[str, Any] = {}

    def get_info(video_path: str) -> Any:
        if video_path not in path_indexes:
            raise ValueError(f"Unknown video: {video_path}")
        if video_path not in info_cache:
            info_cache[video_path] = read_video_info(Path(video_path), cv2)
        return info_cache[video_path]

    def get_annotations(
        annotation_map: dict[str, list[dict[str, Any]]] | None, video_path: str
    ) -> list[dict[str, Any]]:
        return list((annotation_map or {}).get(video_path, []))

    def set_annotations(
        annotation_map: dict[str, list[dict[str, Any]]] | None,
        video_path: str,
        annotations: list[dict[str, Any]],
    ) -> dict[str, list[dict[str, Any]]]:
        updated = dict(annotation_map or {})
        updated[video_path] = annotations
        return updated

    def video_summary(video_path: str) -> str:
        current = get_info(video_path)
        position = path_indexes[video_path] + 1
        return (
            f"**Video {position}/{len(paths)}:** `{current.path}`  \n"
            f"FPS: {current.fps:.3f}　frames: {current.frame_count}　"
            f"duration: {current.duration:.1f} s"
        )

    initial_path = paths[0]
    info = get_info(initial_path)
    initial_frame = read_frame(info, 0, cv2)
    preview = Sam2Preview(checkpoint, model_config, device, sam2_repo, np)

    def render_frame(
        video_path: str,
        frame_idx: float,
        annotation_map: dict[str, list[dict[str, Any]]] | None,
    ) -> Any:
        current = get_info(video_path)
        frame_idx = round(frame_idx)
        annotations = get_annotations(annotation_map, video_path)
        frame = read_frame(current, frame_idx, cv2)
        masks = preview.predict(frame, frame_idx, annotations, source_key=video_path)
        return draw_annotations_and_masks(frame, frame_idx, annotations, masks, np, cv2)

    def change_frame(
        video_path: str,
        frame_idx: float,
        annotation_map: dict[str, list[dict[str, Any]]] | None,
    ) -> tuple[Any, str]:
        annotations = get_annotations(annotation_map, video_path)
        return (
            render_frame(video_path, frame_idx, annotation_map),
            annotation_summary(annotations),
        )

    def switch_video(
        video_path: str,
        annotation_map: dict[str, list[dict[str, Any]]] | None,
    ) -> tuple[str, Any, Any, str, str, None]:
        current = get_info(video_path)
        annotations = get_annotations(annotation_map, video_path)
        preview.reset_frame()
        return (
            video_summary(video_path),
            read_frame(current, 0, cv2),
            gr.update(minimum=0, maximum=current.frame_count - 1, value=0),
            annotation_summary(annotations),
            "Analysis has not started for this video.",
            None,
        )

    def move_video(
        offset: int,
        video_path: str,
        annotation_map: dict[str, list[dict[str, Any]]] | None,
    ) -> tuple[Any, str, Any, Any, str, str, None]:
        index = (path_indexes[video_path] + offset) % len(paths)
        next_path = paths[index]
        return (gr.update(value=next_path), *switch_video(next_path, annotation_map))

    def click_image(
        target: str,
        point_type: str,
        annotation_map: dict[str, list[dict[str, Any]]] | None,
        video_path: str,
        frame_idx: float,
        evt: Any,
    ) -> tuple[Any, dict[str, list[dict[str, Any]]], str]:
        x, y = evt.index
        frame_idx = round(frame_idx)
        annotations = get_annotations(annotation_map, video_path)
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
        updated_map = set_annotations(annotation_map, video_path, updated)
        return (
            render_frame(video_path, frame_idx, updated_map),
            updated_map,
            annotation_summary(updated),
        )

    # Gradio 通过具体的事件类型注解识别这个无需放入 inputs 的参数。
    # gr 是运行时注入的，因此在函数定义后绑定真实类型。
    click_image.__annotations__["evt"] = gr.SelectData

    def undo(
        annotation_map: dict[str, list[dict[str, Any]]] | None,
        video_path: str,
        frame_idx: float,
    ) -> tuple[Any, dict[str, list[dict[str, Any]]], str]:
        annotations = get_annotations(annotation_map, video_path)
        updated = annotations[:-1] if annotations else []
        updated_map = set_annotations(annotation_map, video_path, updated)
        return (
            render_frame(video_path, frame_idx, updated_map),
            updated_map,
            annotation_summary(updated),
        )

    def clear_all(
        annotation_map: dict[str, list[dict[str, Any]]] | None,
        video_path: str,
        frame_idx: float,
    ) -> tuple[Any, dict[str, list[dict[str, Any]]], str]:
        updated_map = set_annotations(annotation_map, video_path, [])
        preview.reset_frame()
        return (
            read_frame(get_info(video_path), frame_idx, cv2),
            updated_map,
            annotation_summary([]),
        )

    def start_analysis(
        video_path: str,
        annotation_map: dict[str, list[dict[str, Any]]] | None,
        progress: Any = gr.Progress(),  # noqa: B008
    ) -> tuple[str, list[str]]:
        try:
            preview.release()
            return run_pipeline(
                get_info(video_path),
                get_annotations(annotation_map, video_path),
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
            "# Basketball Attack Clip Extractor\n\n"
            "Select a target and frame, then click directly on the image. "
            "SAM2 will preview the segmentation mask on annotated frames. "
            "Annotations are retained separately when switching videos."
        )
        annotation_state = gr.State({})
        choices = [
            (path.relative_to(video_root.resolve()).as_posix(), str(path.resolve()))
            for path in video_paths
        ]
        with gr.Row():
            previous_video = gr.Button("← Previous", scale=1)
            video_selector = gr.Dropdown(
                choices=choices,
                value=initial_path,
                label="Video",
                interactive=True,
                scale=5,
            )
            next_video = gr.Button("Next →", scale=1)
        video_meta = gr.Markdown(video_summary(initial_path))
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

        switch_outputs = [
            video_meta,
            image,
            frame_slider,
            stats,
            run_status,
            result_files,
        ]
        video_selector.change(
            switch_video,
            inputs=[video_selector, annotation_state],
            outputs=switch_outputs,
        )
        navigation_outputs = [video_selector, *switch_outputs]
        previous_video.click(
            lambda path, state: move_video(-1, path, state),
            inputs=[video_selector, annotation_state],
            outputs=navigation_outputs,
        )
        next_video.click(
            lambda path, state: move_video(1, path, state),
            inputs=[video_selector, annotation_state],
            outputs=navigation_outputs,
        )

        frame_slider.change(
            change_frame,
            inputs=[video_selector, frame_slider, annotation_state],
            outputs=[image, stats],
        )
        image.select(
            click_image,
            inputs=[
                target_selector,
                point_type,
                annotation_state,
                video_selector,
                frame_slider,
            ],
            outputs=[image, annotation_state, stats],
        )
        undo_button.click(
            undo,
            inputs=[annotation_state, video_selector, frame_slider],
            outputs=[image, annotation_state, stats],
        )
        clear_button.click(
            clear_all,
            inputs=[annotation_state, video_selector, frame_slider],
            outputs=[image, annotation_state, stats],
        )
        start_button.click(
            start_analysis,
            inputs=[video_selector, annotation_state],
            outputs=[run_status, result_files],
        )
    return demo
