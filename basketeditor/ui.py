"""视频标注、追踪和回放工作台。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .annotations import load_annotations, save_annotations
from .models import REPO_DIR, ReIDOptions, TrackingOptions
from .pipeline import run_pipeline
from .sam2 import Sam2Preview
from .stitch_ui import create_clip_workspace
from .theme import blocks, header, page_heading
from .video import (
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
    output_root: Path | None = None,
    reid_options: ReIDOptions | None = None,
) -> Any:
    if not video_paths:
        raise ValueError("At least one video is required.")
    paths = [str(p.resolve()) for p in video_paths]
    indexes = {p: i for i, p in enumerate(paths)}
    info_cache = {}
    output_root = output_root or REPO_DIR / "outputs"
    reid_options = reid_options or ReIDOptions()
    preview = Sam2Preview(checkpoint, model_config, device, sam2_repo, np)
    # All callbacks that touch the shared predictor use the same concurrency group.
    serial = {"concurrency_id": "basketedit-model", "concurrency_limit": 1}

    def get_info(path):
        if path not in indexes:
            raise ValueError("未知视频。")
        if path not in info_cache:
            info_cache[path] = read_video_info(Path(path), cv2)
        return info_cache[path]

    def get_annotations(state, path):
        if path in (state or {}):
            return list(state[path])
        try:
            return load_annotations(output_root, get_info(path))
        except (ValueError, OSError, KeyError) as exc:
            gr.Warning(f"未载入旧标注：{exc}")
            return []

    def commit(state, path, annotations):
        updated = {**(state or {}), path: annotations}
        try:
            save_annotations(output_root, get_info(path), annotations)
        except OSError as exc:
            gr.Warning(f"标注已保留在当前会话，但磁盘保存失败：{exc}")
        return updated

    def metadata(path):
        info = get_info(path)
        return f"{info.duration:.1f} 秒　·　{info.width} × {info.height}　·　{info.fps:.2f} FPS"

    def ready(annotations):
        return {p["target"] for p in annotations if p["label"] == 1} >= {
            "player",
            "ball",
            "hoop",
        }

    def start_state(annotations):
        complete = ready(annotations)
        return gr.update(
            interactive=complete,
            value="开始裁剪" if complete else "先标记三个目标",
        )

    def annotation_progress(annotations):
        rows = []
        completed = 0
        for target, name in (("player", "人物"), ("ball", "篮球"), ("hoop", "篮圈")):
            frames = {
                p["frame_idx"]
                for p in annotations
                if p["target"] == target and p["label"] == 1
            }
            completed += bool(frames)
            status = f"已标记 · {len(frames)} 帧" if frames else "待标记"
            rows.append(
                f'<div class="be-target-status {"is-ready" if frames else ""}">'
                f'<span class="be-target-dot"></span><span>{name}</span>'
                f"<strong>{status}</strong></div>"
            )
        return (
            f'<div class="be-readiness"><span>标记进度</span><b>{completed} / 3</b></div>'
            + "".join(rows)
        )

    def hint(target, operation):
        if operation == "negative":
            return "点一下不需要的区域，排除误选部分。"
        return {
            "player": "点一下球员身体，选中你想追踪的人。",
            "ball": "点在篮球上；画面模糊时，可以换一帧。",
            "hoop": "点在篮圈上；还没出现时，拖到后面的画面。",
        }[target]

    def render(path, index, state, show_preview=True):
        info = get_info(path)
        index = max(0, min(info.frame_count - 1, round(index)))
        annotations = get_annotations(state, path)
        frame = read_frame(info, index, cv2)
        masks = {}
        if show_preview:
            try:
                masks = preview.predict(frame, index, annotations, source_key=path)
            except Exception as exc:
                # A failed model preview must never discard the user's click.
                preview.reset_frame()
                gr.Warning(f"分割预览暂不可用，标注仍已保留：{exc}")
        annotated = sorted({p["frame_idx"] for p in annotations})
        choices = [(f"第 {i} 帧 · {i / info.fps:.2f}s", i) for i in annotated]
        return (
            draw_annotations_and_masks(frame, index, annotations, masks, np, cv2),
            index,
            path,
            annotation_progress(annotations),
            f"{index / info.fps:05.2f}s / {info.duration:.2f}s　 ·　第 {index} 帧",
            gr.update(choices=choices, value=None),
            start_state(annotations),
        )

    initial_path = paths[0]
    info = get_info(initial_path)
    initial_annotations = get_annotations({}, initial_path)
    initial_state = {initial_path: initial_annotations}
    initial_view = render(initial_path, 0, initial_state, False)

    def change_frame(path, index, state, enabled):
        return (
            *render(path, index, state, enabled),
            "标记自动保存，可随时切换工作区。",
        )

    def step_frame(offset, path, index, state, enabled):
        index = max(0, min(get_info(path).frame_count - 1, round(index) + offset))
        return (index, *change_frame(path, index, state, enabled))

    def switch_video(path, state, enabled):
        annotations = get_annotations(state, path)
        state = {**(state or {}), path: annotations}
        preview.reset_frame()
        current = get_info(path)
        return (
            metadata(path),
            gr.update(minimum=0, maximum=max(1, current.frame_count - 1), value=0),
            state,
            *change_frame(path, 0, state, enabled),
            "",
            [],
            None,
            gr.update(choices=[], value=None),
            [],
            "",
            gr.update(visible=False),
        )

    def click_image(target, operation, state, path, index, enabled, evt):
        if operation not in {"positive", "negative"}:
            raise gr.Error("请选择前景点或排除点。")
        x, y = (float(v) for v in evt.index)
        annotations = get_annotations(state, path)
        point = {
            "target": target,
            "frame_idx": int(index),
            "x": x,
            "y": y,
            "label": int(operation == "positive"),
        }
        state = commit(state, path, [*annotations, point])
        view = render(path, index, state, enabled)
        return view[0], state, view[3], "标注已自动保存。", view[5], view[6]

    click_image.__annotations__["evt"] = gr.SelectData

    def remove_points(state, path, index, target, enabled, all_current=False):
        annotations = get_annotations(state, path)
        matching = [
            i
            for i, p in enumerate(annotations)
            if p["frame_idx"] == index and p["target"] == target
        ]
        remove = set(matching if all_current else matching[-1:])
        annotations = [p for i, p in enumerate(annotations) if i not in remove]
        state = commit(state, path, annotations)
        view = render(path, index, state, enabled)
        return view[0], state, view[3], "当前帧的标注已更新。", view[5], view[6]

    def analyze(path, state, hoop_mode, progress=gr.Progress()):
        yield (
            "正在裁剪，完成后会在下方显示片段。",
            [],
            None,
            gr.update(choices=[], value=None),
            [],
            "",
            gr.update(visible=False),
            gr.update(value="正在裁剪…", interactive=False),
        )
        try:
            preview.release()
            status, files = run_pipeline(
                get_info(path),
                get_annotations(state, path),
                checkpoint,
                model_config,
                device,
                sam2_repo,
                np,
                cv2,
                progress,
                output_root=output_root,
                options=TrackingOptions(hoop_mode=hoop_mode),
                reid_options=reid_options,
            )
            event_path = next(Path(p) for p in files if Path(p).name == "events.json")
            events = json.loads(event_path.read_text())["events"]
            rows = [
                [
                    i,
                    f"{event['release_sec']:.2f}s",
                    "疑似命中" if event["outcome"] == "likely_made" else "到筐进攻",
                    f"{event['clip_start_sec']:.2f}–{event['clip_end_sec']:.2f}s",
                ]
                for i, event in enumerate(events, 1)
            ]
            videos = [
                (
                    "追踪回放"
                    if Path(p).name.startswith("debug")
                    else f"进攻 {Path(p).stem.split('_')[-1]}",
                    p,
                )
                for p in files
                if p.endswith(".mp4")
            ]
            yield (
                f"**裁剪完成** · 已生成 {len(events)} 个片段。"
                if events
                else "**分析完成** · 暂未找到进攻片段，可查看追踪回放并补充标记。",
                files,
                videos[0][1] if videos else None,
                gr.update(choices=videos, value=videos[0][1] if videos else None),
                rows,
                status,
                gr.update(visible=True),
                start_state(get_annotations(state, path)),
            )
        except Exception as exc:
            yield (
                f"**裁剪未完成**　{exc}\n\n标记已保存，可以调整后重试。",
                [],
                None,
                gr.update(choices=[], value=None),
                [],
                "",
                gr.update(visible=False),
                start_state(get_annotations(state, path)),
            )

    with blocks(gr, "BasketEdit · 篮球视频工作台") as demo:
        gr.HTML(header())
        annotation_state = gr.State(initial_state)
        displayed_frame = gr.State(0)
        displayed_video = gr.State(initial_path)

        def page_indicator(page):
            return (
                f'<div class="be-page-count" role="status" aria-live="polite" aria-label="第 {page} 页，共 2 页">'
                f"<b>0{page}</b><span> / 02</span></div>"
            )

        with gr.Row(elem_id="be-pager"):
            previous_page = gr.Button(
                "← 上一页", interactive=False, scale=0, min_width=104
            )
            page_count = gr.HTML(page_indicator(1), elem_id="be-page-count")
            next_page = gr.Button("下一页 →", scale=0, min_width=104)

        with gr.Column(elem_id="be-editor") as editor:
            gr.HTML(
                page_heading("裁剪视频", "选一段录像，标记三个目标，留下精彩进攻。")
            )
            with gr.Row(equal_height=False, elem_classes=["be-workspace-row"]):
                with gr.Column(scale=7, min_width=340, elem_classes=["be-panel"]):
                    video_selector = gr.Dropdown(
                        choices=[
                            (
                                str(Path(p).relative_to(video_root.resolve()))
                                if Path(p).is_relative_to(video_root.resolve())
                                else Path(p).name,
                                p,
                            )
                            for p in paths
                        ],
                        value=initial_path,
                        label="选择视频",
                        interactive=True,
                        elem_id="be-source",
                    )
                    video_meta = gr.Markdown(
                        metadata(initial_path), elem_classes=["be-subtle"]
                    )
                    image = gr.Image(
                        value=initial_view[0],
                        label="点击画面标记目标",
                        show_label=False,
                        interactive=False,
                        type="numpy",
                        format="png",
                        height=370,
                        show_download_button=False,
                        elem_id="be-frame",
                    )
                    frame_caption = gr.Markdown(
                        initial_view[4], elem_id="be-frame-caption"
                    )
                    frame_slider = gr.Slider(
                        0,
                        max(1, info.frame_count - 1),
                        value=0,
                        step=1,
                        label="拖动定位画面",
                        show_label=False,
                    )
                    with gr.Row(elem_classes=["be-frame-controls"]):
                        prev_second = gr.Button("−1 秒", size="sm", min_width=55)
                        prev_frame = gr.Button("← 1 帧", size="sm", min_width=55)
                        next_frame = gr.Button("1 帧 →", size="sm", min_width=55)
                        next_second = gr.Button("+1 秒", size="sm", min_width=55)
                    with gr.Accordion(
                        "查看已标记的画面", open=False, elem_classes=["be-accordion"]
                    ):
                        with gr.Row():
                            jump_selector = gr.Dropdown(
                                choices=initial_view[5].get("choices", []),
                                label="已标记的帧",
                                interactive=True,
                                scale=4,
                            )
                            jump_button = gr.Button("跳转 ↗", scale=1, min_width=65)
                    gr.HTML(
                        '<p class="be-note">三个目标可以在不同帧标记，篮圈晚些出现也没关系。</p>'
                    )

                with gr.Column(
                    scale=3,
                    min_width=290,
                    elem_classes=["be-panel"],
                    elem_id="be-controls",
                ):
                    gr.HTML(
                        '<div class="be-section">标记三个目标<small>选择一个目标，再点击左侧画面。</small></div>'
                    )
                    target_selector = gr.Radio(
                        [("人物", "player"), ("篮球", "ball"), ("篮圈", "hoop")],
                        value="player",
                        label="追踪目标",
                        show_label=False,
                        container=False,
                        elem_id="be-targets",
                    )
                    annotation_hint = gr.Markdown(
                        hint("player", "positive"), elem_classes=["be-subtle"]
                    )
                    operation = gr.Radio(
                        [("选中目标 +", "positive"), ("排除区域 −", "negative")],
                        value="positive",
                        label="标记方式",
                        show_label=False,
                        container=False,
                        elem_id="be-operation",
                    )
                    with gr.Row():
                        undo_button = gr.Button("撤销", size="sm", min_width=70)
                        clear_button = gr.Button(
                            "清除此帧标记", size="sm", min_width=120
                        )
                    stats = gr.HTML(initial_view[3], elem_id="be-stats")
                    annotation_status = gr.Markdown(
                        "标记自动保存，可随时切换工作区。", elem_classes=["be-subtle"]
                    )
                    with gr.Accordion(
                        "更多设置",
                        open=False,
                        elem_classes=["be-accordion"],
                        elem_id="be-settings",
                    ):
                        hoop_mode = gr.Dropdown(
                            [
                                ("自动稳定", "auto"),
                                ("移动机位", "moving"),
                                ("固定机位", "fixed"),
                            ],
                            value="auto",
                            label="篮圈稳定方式",
                            interactive=True,
                        )
                        gr.HTML(
                            '<p class="be-note">只有镜头与篮圈都不移动时，才选择固定机位。</p>'
                        )
                        preview_enabled = gr.Checkbox(
                            value=True, label="显示即时分割预览"
                        )
                        gr.HTML(
                            '<p class="be-note">已开启身份记忆与自动找回。</p>'
                            if reid_options.enabled
                            else '<p class="be-note">当前为 SAM2 对照模式，自动身份找回已关闭。</p>'
                        )
                    start_button = gr.Button(
                        "开始裁剪" if ready(initial_annotations) else "先标记三个目标",
                        variant="primary",
                        interactive=ready(initial_annotations),
                        elem_id="be-start",
                    )
                    run_status = gr.Markdown(
                        "", elem_id="be-run-status", elem_classes=["be-subtle"]
                    )

            with gr.Column(
                visible=False, elem_classes=["be-panel"], elem_id="be-results"
            ) as results_panel:
                with gr.Row():
                    gr.HTML(
                        '<div class="be-section">裁剪结果<small>先看回放，再把喜欢的片段加入集锦。</small></div>'
                    )
                    result_to_stitch = gr.Button(
                        "下一步：拼接 →", scale=0, min_width=145
                    )
                result_selector = gr.Dropdown(
                    choices=[], label="选择片段或追踪回放", interactive=True
                )
                result_video = gr.Video(label="片段预览", interactive=False, height=360)
                with gr.Accordion(
                    "下载文件与详细记录", open=False, elem_classes=["be-accordion"]
                ):
                    result_files = gr.File(
                        label="下载片段与报告", file_count="multiple"
                    )
                    event_table = gr.Dataframe(
                        headers=["#", "出手", "结果", "片段区间"],
                        datatype=["number", "str", "str", "str"],
                        value=[],
                        interactive=False,
                        label="进攻记录",
                        wrap=True,
                    )
                    result_details = gr.Markdown("")
                    gr.HTML(
                        '<p class="be-note">疑似命中需要结合回放确认；漏追时可以补充清晰画面的标记。</p>'
                    )

        with gr.Column(visible=False, elem_id="be-stitcher") as stitcher:
            gr.HTML(
                page_heading("拼接集锦", "挑选已有片段，排好播放顺序，合成一支集锦。")
            )
            library = create_clip_workspace(output_root, gr, allow_edit=True)
        gr.HTML(
            '<footer class="be-footer"><span>BasketEdit</span><span>每一次出手，都值得留下。</span></footer>'
        )

        def navigate(mode):
            return (
                gr.update(visible=mode == "edit"),
                gr.update(visible=mode == "stitch"),
                page_indicator(1 if mode == "edit" else 2),
                gr.update(interactive=mode == "stitch"),
                gr.update(interactive=mode == "edit"),
            )

        navigation_outputs = [editor, stitcher, page_count, previous_page, next_page]
        for button, mode in (
            (previous_page, "edit"),
            (next_page, "stitch"),
            (result_to_stitch, "stitch"),
            (library.edit_button, "edit"),
        ):
            event = button.click(
                lambda mode=mode: navigate(mode),
                outputs=navigation_outputs,
                queue=False,
            )
            event.then(
                fn=None, js="() => { window.scrollTo(0, 0); return []; }", queue=False
            )
            if mode == "stitch":
                library.refresh_on(event.then)

        frame_outputs = [
            image,
            displayed_frame,
            displayed_video,
            stats,
            frame_caption,
            jump_selector,
            start_button,
            annotation_status,
        ]
        switch_outputs = [
            video_meta,
            frame_slider,
            annotation_state,
            *frame_outputs,
            run_status,
            result_files,
            result_video,
            result_selector,
            event_table,
            result_details,
            results_panel,
        ]
        video_selector.input(
            switch_video,
            [video_selector, annotation_state, preview_enabled],
            switch_outputs,
            **serial,
        )
        frame_slider.release(
            change_frame,
            [video_selector, frame_slider, annotation_state, preview_enabled],
            frame_outputs,
            **serial,
        )
        for button, offset in (
            (prev_frame, -1),
            (next_frame, 1),
            (prev_second, -30),
            (next_second, 30),
        ):

            def step(path, index, state, enabled, offset=offset):
                delta = (
                    offset
                    if abs(offset) == 1
                    else round(get_info(path).fps) * (1 if offset > 0 else -1)
                )
                return step_frame(delta, path, index, state, enabled)

            button.click(
                step,
                [displayed_video, displayed_frame, annotation_state, preview_enabled],
                [frame_slider, *frame_outputs],
                **serial,
            )
        jump_button.click(
            lambda path, index, state, enabled: (
                int(index or 0),
                *change_frame(path, int(index or 0), state, enabled),
            ),
            [video_selector, jump_selector, annotation_state, preview_enabled],
            [frame_slider, *frame_outputs],
            **serial,
        )
        annotation_outputs = [
            image,
            annotation_state,
            stats,
            annotation_status,
            jump_selector,
            start_button,
        ]
        image.select(
            click_image,
            [
                target_selector,
                operation,
                annotation_state,
                displayed_video,
                displayed_frame,
                preview_enabled,
            ],
            annotation_outputs,
            **serial,
        )
        undo_button.click(
            remove_points,
            [
                annotation_state,
                displayed_video,
                displayed_frame,
                target_selector,
                preview_enabled,
            ],
            annotation_outputs,
            **serial,
        )
        clear_button.click(
            lambda state, path, index, target, enabled: remove_points(
                state, path, index, target, enabled, True
            ),
            [
                annotation_state,
                displayed_video,
                displayed_frame,
                target_selector,
                preview_enabled,
            ],
            annotation_outputs,
            **serial,
        )
        for control in (operation, target_selector):
            control.input(
                hint,
                [target_selector, operation],
                annotation_hint,
                queue=False,
            )
        preview_enabled.change(
            change_frame,
            [displayed_video, displayed_frame, annotation_state, preview_enabled],
            frame_outputs,
            **serial,
        )
        analysis_event = start_button.click(
            analyze,
            [video_selector, annotation_state, hoop_mode],
            [
                run_status,
                result_files,
                result_video,
                result_selector,
                event_table,
                result_details,
                results_panel,
                start_button,
            ],
            **serial,
        )
        library.refresh_on(analysis_event.then)
        library.refresh_on(demo.load)
        result_selector.input(lambda value: value, result_selector, result_video)
        demo.load(
            lambda: switch_video(initial_path, {}, False),
            outputs=switch_outputs,
            show_progress="hidden",
            **serial,
        )
    return demo
