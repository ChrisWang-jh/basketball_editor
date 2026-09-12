"""视频读取、标注绘制、调试视频和片段导出。"""

from __future__ import annotations

import json
import math
import shutil
import subprocess
import tempfile
import warnings
from itertools import pairwise
from pathlib import Path
from typing import Any

from .models import COLORS, AttackEvent, FrameState, TrackPoint, VideoInfo

VIDEO_EXTENSIONS = {
    ".avi",
    ".m4v",
    ".mkv",
    ".mov",
    ".mp4",
    ".mpeg",
    ".mpg",
    ".webm",
}


def discover_videos(directory: Path) -> list[Path]:
    """递归查找目录中的视频，并按相对路径稳定排序。"""
    directory = directory.expanduser().resolve()
    if not directory.is_dir():
        raise NotADirectoryError(f"Video directory not found: {directory}")
    videos = [
        path.resolve()
        for path in directory.rglob("*")
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
    ]
    videos.sort(key=lambda path: path.relative_to(directory).as_posix().casefold())
    if not videos:
        supported = ", ".join(sorted(VIDEO_EXTENSIONS))
        raise FileNotFoundError(
            f"No videos found below {directory}. Supported extensions: {supported}"
        )
    return videos


def read_video_info(video_path: Path, cv2: Any) -> VideoInfo:
    """读取视频元数据，不修改原始视频。"""
    if not video_path.is_file():
        raise FileNotFoundError(f"Video not found: {video_path}")
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    capture.release()
    if (
        not math.isfinite(fps)
        or fps <= 0
        or frame_count <= 0
        or width <= 0
        or height <= 0
    ):
        raise RuntimeError(
            "Invalid video metadata; verify that OpenCV supports this codec."
        )
    return VideoInfo(
        str(video_path.resolve()), fps, frame_count, width, height, frame_count / fps
    )


def read_frame(info: VideoInfo, frame_idx: float, cv2: Any) -> Any:
    """按帧号读取单帧，用于网页中的交互式点选。"""
    frame_idx = max(0, min(info.frame_count - 1, round(frame_idx)))
    capture = cv2.VideoCapture(info.path)
    capture.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, image = capture.read()
    capture.release()
    if not ok:
        raise RuntimeError(f"Could not read frame {frame_idx}.")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def draw_annotations(
    image: Any, frame_idx: int, annotations: list[dict[str, Any]], cv2: Any
) -> Any:
    """将当前帧的标注画回图像：前景点画圆，排除点画叉。"""
    result = image.copy()
    for point in annotations:
        if int(point["frame_idx"]) != int(frame_idx):
            continue
        x, y = round(float(point["x"])), round(float(point["y"]))
        color = COLORS[point["target"]]
        if "box" in point:
            x1, y1, x2, y2 = map(round, point["box"])
            cv2.rectangle(result, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
            continue
        if int(point["label"]) == 1:
            cv2.circle(result, (x, y), 8, color, -1, cv2.LINE_AA)
            cv2.circle(result, (x, y), 11, (255, 255, 255), 2, cv2.LINE_AA)
        else:
            cv2.line(result, (x - 9, y - 9), (x + 9, y + 9), color, 3, cv2.LINE_AA)
            cv2.line(result, (x - 9, y + 9), (x + 9, y - 9), color, 3, cv2.LINE_AA)
    return result


def overlay_masks(
    image: Any,
    masks: dict[str, Any],
    np: Any,
    cv2: Any,
    alpha: float = 0.35,
) -> Any:
    """以半透明颜色叠加 SAM2 当前帧分割结果，并绘制边界。"""
    result = image.copy()
    for target in ("player", "hoop", "ball"):
        mask = masks.get(target)
        if mask is None:
            continue
        mask_array = np.squeeze(np.asarray(mask)).astype(bool)
        if mask_array.ndim != 2 or mask_array.shape != result.shape[:2]:
            continue
        color = np.asarray(COLORS[target], dtype=np.float32)
        blended = (1.0 - alpha) * result[mask_array].astype(np.float32) + alpha * color
        result[mask_array] = np.clip(blended, 0, 255).astype(np.uint8)
        contours, _ = cv2.findContours(
            mask_array.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(result, contours, -1, COLORS[target], 2, cv2.LINE_AA)
    return result


def draw_annotations_and_masks(
    image: Any,
    frame_idx: int,
    annotations: list[dict[str, Any]],
    masks: dict[str, Any],
    np: Any,
    cv2: Any,
) -> Any:
    """先画 SAM2 掩码，再把正/负提示点画在最上层。"""
    return draw_annotations(
        overlay_masks(image, masks, np, cv2), frame_idx, annotations, cv2
    )


def extract_tracking_frames(info: VideoInfo, directory: Path, cv2: Any) -> list[int]:
    """为 SAM2 抽取按数字排序的临时帧。"""
    capture = cv2.VideoCapture(info.path)
    written = 0
    previous = None
    cuts = []
    try:
        while True:
            ok, image = capture.read()
            if not ok:
                break
            small = cv2.resize(image, (64, 36))
            histogram = cv2.calcHist(
                [cv2.cvtColor(small, cv2.COLOR_BGR2HSV)],
                [0, 1],
                None,
                [24, 16],
                [0, 180, 0, 256],
            )
            cv2.normalize(histogram, histogram)
            if previous is not None:
                difference = float(cv2.absdiff(previous[0], small).mean()) / 255
                histogram_change = cv2.compareHist(
                    previous[1], histogram, cv2.HISTCMP_BHATTACHARYYA
                )
                if difference > 0.25 and histogram_change > 0.55:
                    cuts.append(written)
            previous = small, histogram
            file_path = directory / f"{written:08d}.jpg"
            if not cv2.imwrite(str(file_path), image, [cv2.IMWRITE_JPEG_QUALITY, 95]):
                raise RuntimeError(
                    f"Could not write temporary tracking frame: {file_path}"
                )
            written += 1
    finally:
        capture.release()
    if written == 0:
        raise RuntimeError("视频未解码出任何画面。")
    # Container frame counts can be estimates, especially for VFR. The subsequent
    # ffprobe timestamp check verifies the decoded count against actual frames.
    info.frame_count = written
    info.duration = written / info.fps
    return cuts


def read_frame_timestamps(info: VideoInfo) -> None:
    """Use presentation timestamps for exact clipping, including variable-FPS input."""
    ffprobe = shutil.which("ffprobe")
    if ffprobe is None:
        raise RuntimeError("ffprobe was not found. Install FFmpeg including ffprobe.")
    result = subprocess.run(
        [
            ffprobe,
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "frame=best_effort_timestamp_time:format=duration",
            "-of",
            "json",
            info.path,
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode:
        raise RuntimeError(f"Could not read video timestamps: {result.stderr[-500:]}")
    data = json.loads(result.stdout)
    times = [
        float(frame["best_effort_timestamp_time"])
        for frame in data.get("frames", [])
        if "best_effort_timestamp_time" in frame
    ]
    if (
        len(times) != info.frame_count
        or not all(math.isfinite(t) for t in times)
        or any(a >= b for a, b in pairwise(times))
    ):
        raise RuntimeError("视频时间戳缺失或不递增，请先将视频转为固定帧率后重新标注。")
    origin = times[0]
    info.frame_times = [t - origin for t in times]
    last_delta = times[-1] - times[-2] if len(times) > 1 else 1 / info.fps
    info.duration = info.frame_times[-1] + last_delta


# A source must describe an accepted measurement, not merely have a plausible box.
# Keep this list in sync with the tracking backends; unknown states stay unverified.
_OBSERVED_SOURCES = {"sam2", "anchor", "tracked", "reid", "reacquired"}
_DERIVED_SOURCES = {"stabilized", "fixed"}


def _measurement(point: TrackPoint, frame_idx: int) -> bool:
    return (
        point.frame_idx == frame_idx
        and point.visible
        and not point.interpolated
        and point.source in _OBSERVED_SOURCES
    )


def _mask_for_observation(
    mask: Any,
    point: TrackPoint,
    image_shape: tuple[int, ...],
    target: str,
    np: Any,
) -> tuple[Any, str]:
    if mask is None:
        return None, "MASK UNAVAILABLE"
    array = np.asarray(mask)
    if array.ndim != 2 or array.shape != image_shape[:2]:
        warnings.warn(
            f"Ignoring {target} mask with shape {array.shape}; expected {image_shape[:2]}.",
            RuntimeWarning,
            stacklevel=2,
        )
        return None, "INVALID MASK"
    if not np.issubdtype(array.dtype, np.number) and array.dtype != np.bool_:
        warnings.warn(
            f"Ignoring non-numeric {target} mask.", RuntimeWarning, stacklevel=2
        )
        return None, "INVALID MASK"
    if not np.isfinite(array).all():
        warnings.warn(
            f"Ignoring non-finite {target} mask.", RuntimeWarning, stacklevel=2
        )
        return None, "INVALID MASK"
    selected = array > 0
    ys, xs = np.nonzero(selected)
    if not len(xs):
        return None, "EMPTY MASK"
    # Compare against the actual measurement, never a median-smoothed or fixed box.
    # A little edge tolerance allows inclusive/exclusive box conventions.
    if point.bbox is not None:
        x1, y1, x2, y2 = point.bbox
        margin = max(3.0, min(x2 - x1, y2 - y1) * 0.08)
        inside = (
            (xs >= x1 - margin)
            & (xs <= x2 + margin)
            & (ys >= y1 - margin)
            & (ys <= y2 + margin)
        )
        if float(inside.mean()) < 0.5:
            warnings.warn(
                f"Ignoring {target} mask that disagrees with its accepted observation.",
                RuntimeWarning,
                stacklevel=2,
            )
            return None, "MASK MISMATCH"
    return selected, "TRACKED"


def _dashed_box(image: Any, bbox: Any, color: Any, cv2: Any) -> None:
    if bbox is None or not all(math.isfinite(value) for value in bbox):
        return
    height, width = image.shape[:2]
    x1, y1, x2, y2 = bbox
    x1, x2 = sorted(max(0, min(width - 1, round(value))) for value in (x1, x2))
    y1, y2 = sorted(max(0, min(height - 1, round(value))) for value in (y1, y2))
    for start, end in (
        ((x1, y1), (x2, y1)),
        ((x2, y1), (x2, y2)),
        ((x2, y2), (x1, y2)),
        ((x1, y2), (x1, y1)),
    ):
        distance = math.dist(start, end)
        if not distance:
            continue
        for offset in range(0, math.ceil(distance), 14):
            a, b = offset / distance, min(distance, offset + 7) / distance
            p1 = tuple(round(s + (e - s) * a) for s, e in zip(start, end))
            p2 = tuple(round(s + (e - s) * b) for s, e in zip(start, end))
            cv2.line(image, p1, p2, color, 1, cv2.LINE_AA)


def render_debug_frame(
    image: Any,
    frame: FrameState,
    masks: dict[str, Any],
    np: Any,
    cv2: Any,
    observations: dict[str, TrackPoint] | None = None,
) -> Any:
    """绘制当前帧的真实实例分割；缺失和插值不能冒充像素级观测。

    ``image`` 使用 OpenCV 的 BGR 顺序。``observations`` 是生成磁盘 mask 的、
    已经通过身份核验的观测，允许计分轨迹平滑后仍显示原位置上的真实轮廓。
    """
    result = image.copy()
    statuses = {}
    points = {target: getattr(frame, target) for target in ("player", "hoop", "ball")}
    for target, point in points.items():
        observed = observations.get(target) if observations is not None else point
        color_bgr = COLORS[target][::-1]
        accepted = (
            point.visible
            and not point.interpolated
            and point.source in _OBSERVED_SOURCES | _DERIVED_SOURCES
            and observed is not None
            and _measurement(observed, frame.frame_idx)
        )
        if not point.visible:
            statuses[target] = "LOST"
            continue
        if point.interpolated or (point.source == "fixed" and not accepted):
            statuses[target] = "ESTIMATED"
            _dashed_box(result, point.bbox, color_bgr, cv2)
            continue
        if not accepted:
            statuses[target] = "UNVERIFIED"
            continue

        selected, status = _mask_for_observation(
            masks.get(target), observed, result.shape, target, np
        )
        statuses[target] = status
        if selected is None:
            continue
        color = np.asarray(color_bgr, dtype=np.float32)
        result[selected] = np.clip(
            result[selected].astype(np.float32) * 0.58 + color * 0.42, 0, 255
        ).astype(np.uint8)
        contours, _ = cv2.findContours(
            selected.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(result, contours, -1, (28, 25, 22), 4, cv2.LINE_AA)
        cv2.drawContours(result, contours, -1, color_bgr, 2, cv2.LINE_AA)
        ys, xs = np.nonzero(selected)
        cv2.putText(
            result,
            target,
            (int(xs.min()), max(14, int(ys.min()) - 7)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color_bgr,
            1,
            cv2.LINE_AA,
        )

    header = f"frame:{frame.frame_idx}  stage:{frame.stage}"
    if frame.scene_start:
        header += "  SCENE CUT"
    lines = [(header, (255, 255, 255))]
    lines.extend(
        (
            f"{target}: {statuses[target]}  {point.source}  mask:{point.confidence:.2f}"
            + (
                f"  id:{point.identity_score:.2f}"
                if point.identity_score is not None
                else ""
            ),
            COLORS[target][::-1],
        )
        for target, point in points.items()
    )
    height, width = result.shape[:2]
    scale = max(0.23, min(0.55, width / 1200))
    sizes = [
        cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)[0]
        for line, _ in lines
    ]
    line_height = max(13, max(size[1] for size in sizes) + 8)
    panel_width = min(width - 8, max(size[0] for size in sizes) + 16)
    panel_height = min(height - 8, line_height * len(lines) + 8)
    panel = result[4 : 4 + panel_height, 4 : 4 + panel_width]
    panel[:] = (panel.astype(np.float32) * 0.25).astype(np.uint8)
    for index, (line, color) in enumerate(lines):
        cv2.putText(
            result,
            line,
            (10, 4 + (index + 1) * line_height),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            color,
            1,
            cv2.LINE_AA,
        )
    return result


def export_debug_video(
    info: VideoInfo,
    frame_states: list[FrameState],
    output: Path,
    masks_dir: Path,
    np: Any,
    cv2: Any,
    observed_tracks: dict[str, list[TrackPoint]] | None = None,
) -> None:
    """导出真实分割轮廓和状态，保留源音轨及帧时间对应关系。"""
    if len(frame_states) != info.frame_count or any(
        frame.frame_idx != index for index, frame in enumerate(frame_states)
    ):
        raise ValueError("Debug frame states must cover every source frame in order.")
    if observed_tracks is not None and any(
        len(observed_tracks.get(target, [])) != info.frame_count
        for target in ("player", "hoop", "ball")
    ):
        raise ValueError("Debug observations must cover every source frame.")
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg was not found. Install it and add it to PATH.")
    capture = cv2.VideoCapture(info.path)
    if not capture.isOpened():
        raise RuntimeError(f"Could not open source video for debug export: {info.path}")

    temp_output = output.with_name(f".{output.stem}.encoding.mp4")
    frame_limit = (
        max(1, round(info.duration * info.fps))
        if info.frame_times
        else len(frame_states)
    )
    command = [
        ffmpeg,
        "-y",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-s:v",
        f"{info.width}x{info.height}",
        "-r",
        f"{info.fps:.10f}",
        "-i",
        "pipe:0",
        "-i",
        info.path,
        "-map",
        "0:v:0",
        "-map",
        "1:a:0?",
        "-t",
        f"{frame_limit / info.fps:.10f}",
        "-vf",
        "pad=ceil(iw/2)*2:ceil(ih/2)*2",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "21",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-movflags",
        "+faststart",
        str(temp_output),
    ]
    written = 0
    error_text = ""
    with tempfile.TemporaryFile(mode="w+b") as error_log:
        process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=error_log,
        )
        try:
            if process.stdin is None:
                raise RuntimeError("Could not open the FFmpeg input pipe.")
            for frame in frame_states:
                ok, image = capture.read()
                if not ok:
                    raise RuntimeError(
                        f"Source video ended while reading debug frame {frame.frame_idx}."
                    )
                masks = {}
                for target in ("player", "hoop", "ball"):
                    mask_path = masks_dir / target / f"{frame.frame_idx:08d}.png"
                    if mask_path.is_file():
                        masks[target] = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                        if masks[target] is None:
                            warnings.warn(
                                f"Could not decode tracking mask: {mask_path}",
                                RuntimeWarning,
                                stacklevel=2,
                            )
                observations = (
                    {
                        target: points[frame.frame_idx]
                        for target, points in observed_tracks.items()
                    }
                    if observed_tracks is not None
                    else None
                )
                rendered = render_debug_frame(
                    image, frame, masks, np, cv2, observations
                )
                # A variable-rate source can hold a frame longer than 1/fps.
                # Reproduce those holds at the debug video's fixed output rate so
                # its audio stays aligned with the original video timestamps.
                end = (
                    min(
                        frame_limit,
                        round(info.frame_time(frame.frame_idx, end=True) * info.fps),
                    )
                    if info.frame_times
                    else frame.frame_idx + 1
                )
                for _ in range(max(0, end - written)):
                    process.stdin.write(rendered.tobytes())
                    written += 1
            process.stdin.close()
            return_code = process.wait()
        except (BrokenPipeError, OSError) as exc:
            if process.stdin is not None and not process.stdin.closed:
                process.stdin.close()
            process.wait()
            temp_output.unlink(missing_ok=True)
            raise RuntimeError(
                "FFmpeg stopped while encoding the debug video."
            ) from exc
        except Exception:
            if process.stdin is not None and not process.stdin.closed:
                process.stdin.close()
            process.wait()
            temp_output.unlink(missing_ok=True)
            raise
        finally:
            capture.release()
            error_log.seek(0)
            error_text = error_log.read().decode("utf-8", errors="replace")

    if return_code != 0 or written != frame_limit:
        temp_output.unlink(missing_ok=True)
        raise RuntimeError(
            "Debug video export failed: " + (error_text[-1000:] or "unknown error")
        )
    if not temp_output.is_file() or temp_output.stat().st_size == 0:
        raise RuntimeError("FFmpeg did not create a complete debug video.")
    temp_output.replace(output)


def cut_clips(
    info: VideoInfo, events: list[AttackEvent], output_dir: Path
) -> list[Path]:
    """根据事件时间戳，调用 FFmpeg 从原视频裁剪片段。"""
    if events and shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg was not found. Install it and add it to PATH.")
    output_files: list[Path] = []
    for index, current in enumerate(events, 1):
        start_time = info.frame_time(current.clip_start_frame)
        end_time = info.frame_time(current.clip_end_frame, end=True)
        output = output_dir / f"attack_{index:03d}.mp4"
        partial = output.with_name(f".{output.stem}.encoding.mp4")
        command = [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-ss",
            f"{start_time:.6f}",
            "-i",
            info.path,
            "-t",
            f"{end_time - start_time:.6f}",
            "-map",
            "0:v:0",
            "-map",
            "0:a:0?",
            "-vf",
            "pad=ceil(iw/2)*2:ceil(ih/2)*2",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-movflags",
            "+faststart",
            str(partial),
        ]
        result = subprocess.run(command, capture_output=True, text=True, check=False)
        if result.returncode != 0:
            partial.unlink(missing_ok=True)
            raise RuntimeError(f"Clip {index} export failed: {result.stderr[-500:]}")
        if not partial.is_file() or partial.stat().st_size == 0:
            raise RuntimeError(f"Clip {index} export produced an empty file")
        partial.replace(output)
        output_files.append(output)
    return output_files
