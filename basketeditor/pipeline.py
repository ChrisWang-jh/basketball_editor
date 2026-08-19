"""把视频、SAM2、轨迹规则和结果导出串成完整分析流程。"""

from __future__ import annotations

import json
import tempfile
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

from .models import REPO_DIR, RULES, TARGET_IDS, TrackPoint, VideoInfo
from .rules import build_frame_states, detect_attacks
from .sam2 import run_sam2_tracking
from .video import cut_clips, export_debug_video, extract_tracking_frames


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
) -> tuple[str, list[str]]:
    """执行抽帧、追踪、规则分析、调试视频生成和片段裁剪。"""
    for target in TARGET_IDS:
        if not any(
            point["target"] == target and point["label"] == 1 for point in annotations
        ):
            raise ValueError(f"Please add at least one positive point for {target!r}.")

    timestamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S_%f")
    output_root = output_root or REPO_DIR / "outputs"
    output_dir = output_root / f"{Path(info.path).stem}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=False)
    tracks_file = output_dir / "tracks.json"
    events_file = output_dir / "events.json"
    debug_video = output_dir / "debug_tracking.mp4"

    if progress:
        progress(0.05, desc="Extracting frames for SAM2 tracking")
    with tempfile.TemporaryDirectory(prefix="sam2_frames_", dir=output_dir) as temp:
        frames_dir = Path(temp) / "frames"
        masks_dir = Path(temp) / "masks"
        frames_dir.mkdir()
        extract_tracking_frames(info, frames_dir, cv2)
        if progress:
            progress(
                0.20,
                desc="SAM2 is tracking the three targets bidirectionally",
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
        )

        if progress:
            progress(0.75, desc="Applying trajectory rules and detecting attacks")
        frame_states = build_frame_states(tracks)
        events = detect_attacks(frame_states, info)

        if progress:
            progress(0.82, desc="Generating H.264 debug tracking video")
        export_debug_video(info, frame_states, debug_video, masks_dir, np, cv2)

    _write_json(
        tracks_file,
        {
            "video_info": asdict(info),
            "tracks": {
                target: [_track_point_to_dict(point) for point in points]
                for target, points in tracks.items()
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
                    "clip_start_sec": event.clip_start_frame / info.fps,
                    "clip_end_sec": event.clip_end_frame / info.fps,
                }
                for event in events
            ],
        },
    )
    if progress:
        progress(0.93, desc="Exporting valid attack clips")
    clips = cut_clips(info, events, output_dir)

    files = [str(events_file), str(tracks_file), str(debug_video), *map(str, clips)]
    status = (
        f"### Analysis complete\n\nDetected **{len(events)}** valid attack.\n\n"
        f"Output directory: `{output_dir}`"
    )
    return status, files
