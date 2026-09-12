"""把视频、SAM2、轨迹规则和结果导出串成完整分析流程。"""

from __future__ import annotations

import json
import shutil
import tempfile
from dataclasses import asdict, replace
from datetime import datetime
from pathlib import Path
from typing import Any

from .annotations import validate_annotations
from .models import REPO_DIR, RULES, ReIDOptions, TrackingOptions, TrackPoint, VideoInfo
from .rules import build_frame_states, detect_attacks
from .sam2 import run_sam2_tracking
from .tracking import prepare_tracks, tracking_diagnostics
from .video import (
    cut_clips,
    export_debug_video,
    extract_tracking_frames,
    read_frame_timestamps,
)


def _track_point_to_dict(point: TrackPoint) -> dict[str, Any]:
    value = asdict(point)
    if value["bbox"] is not None:
        value["bbox"] = list(value["bbox"])
    if value["center"] is not None:
        value["center"] = list(value["center"])
    return value


def _write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def run_pipeline(
    info: VideoInfo,
    annotations: list[dict[str, Any]],
    checkpoint: Path,
    model_config: str,
    device: str,
    sam2_repo: Path | None,
    np: Any,
    cv2: Any,
    progress: Any = None,
    output_root: Path | None = None,
    options: TrackingOptions | None = None,
    reid_options: ReIDOptions | None = None,
) -> tuple[str, list[str]]:
    """执行抽帧、追踪、规则分析、调试视频生成和片段裁剪。"""
    info = replace(info)
    annotations = validate_annotations(annotations, info)
    options = options or TrackingOptions()
    reid_options = reid_options or ReIDOptions()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"SAM2 checkpoint not found: {checkpoint}")
    for binary in ("ffmpeg", "ffprobe"):
        if shutil.which(binary) is None:
            raise RuntimeError(f"请先安装 {binary}。")

    timestamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S_%f")
    output_root = output_root or REPO_DIR / "outputs"
    output_dir = output_root / f"{Path(info.path).stem}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=False)
    tracks_file = output_dir / "tracks.json"
    events_file = output_dir / "events.json"
    debug_video = output_dir / "debug_tracking.mp4"
    annotations_file = output_dir / "annotations.json"
    report_file = output_dir / "tracking_report.json"
    identity_file = output_dir / "identity_memory.npz"
    identity_diagnostics = {}
    _write_json(
        annotations_file, {"video_info": asdict(info), "annotations": annotations}
    )

    if progress:
        progress(0.05, desc="正在读取视频画面")
    with tempfile.TemporaryDirectory(prefix="sam2_frames_", dir=output_dir) as temp:
        frames_dir = Path(temp) / "frames"
        masks_dir = Path(temp) / "masks"
        frames_dir.mkdir()
        scene_cuts = extract_tracking_frames(info, frames_dir, cv2)
        annotations = validate_annotations(annotations, info)
        read_frame_timestamps(info)
        if progress:
            progress(
                0.20,
                desc="正在分别追踪人物、篮球和篮圈",
            )
        tracks = run_sam2_tracking(
            frames_dir,
            info,
            annotations,
            checkpoint,
            model_config,
            device,
            sam2_repo,
            np,
            masks_dir,
            cv2,
            scene_cuts,
            progress,
            reid_options=reid_options,
            diagnostics=identity_diagnostics,
            identity_file=identity_file if reid_options.enabled else None,
        )

        if progress:
            progress(0.75, desc="正在恢复轨迹并识别进攻")
        raw_tracks = tracks
        tracks = prepare_tracks(raw_tracks, info, options, scene_cuts)
        frame_states = build_frame_states(tracks, info, scene_cuts)
        events = detect_attacks(frame_states, info)
        diagnostics = tracking_diagnostics(tracks, info)
        _write_json(
            report_file,
            {
                "options": asdict(options),
                "scene_cuts": scene_cuts,
                "targets": diagnostics,
                "identity": identity_diagnostics,
                "identity_options": {
                    k: str(v) if isinstance(v, Path) else v
                    for k, v in asdict(reid_options).items()
                },
                "note": "Identity templates do not expire. LOST/PENDING frames are not observations; recovery requires appearance evidence. likely_made is 2D trajectory evidence, not a verified score.",
            },
        )

        # Preserve usable intermediate results even if video encoding fails later.
        _write_json(
            tracks_file,
            {
                "video_info": asdict(info),
                "tracks": {
                    t: [_track_point_to_dict(p) for p in pts]
                    for t, pts in tracks.items()
                },
                "raw_tracks": {
                    t: [_track_point_to_dict(p) for p in pts]
                    for t, pts in raw_tracks.items()
                },
            },
        )

        _write_json(
            events_file,
            {
                "video_info": asdict(info),
                "rules": RULES,
                "events": [
                    {
                        **asdict(event),
                        "clip_start_sec": info.frame_time(event.clip_start_frame),
                        "clip_end_sec": info.frame_time(event.clip_end_frame, end=True),
                        "release_sec": info.frame_time(event.release_frame),
                    }
                    for event in events
                ],
            },
        )
        if progress:
            progress(0.82, desc="正在生成追踪回放")
        export_debug_video(
            info,
            frame_states,
            debug_video,
            masks_dir,
            np,
            cv2,
            observed_tracks=raw_tracks,
        )

    if progress:
        progress(0.93, desc="正在导出进攻片段")
    clips = cut_clips(info, events, output_dir)

    files = [
        str(events_file),
        str(tracks_file),
        str(debug_video),
        str(report_file),
        str(annotations_file),
        *([str(identity_file)] if identity_file.is_file() else []),
        *map(str, clips),
    ]
    made = sum(e.outcome == "likely_made" for e in events)
    gaps = sum(len(d["gaps"]) for d in diagnostics.values())
    recovered = sum(len(d["reacquired_frames"]) for d in diagnostics.values())
    status = (
        f"**分析完成 · {len(events)} 次进攻 · {made} 次疑似命中**\n\n"
        f"已自动找回 {recovered} 次；仍有 {gaps} 段缺失或身份待确认。分割回放仅显示已确认目标；可按报告帧号补标。疑似命中依据二维轨迹穿越篮圈判断。\n\n"
        f"结果已保存到 `{output_dir}`"
    )
    if progress:
        progress(1, desc="分析完成")
    return status, files
