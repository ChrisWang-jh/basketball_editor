"""视频读取、标注绘制、调试视频和片段导出。"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from .models import COLORS, AttackEvent, FrameState, VideoInfo

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
    if fps <= 0 or frame_count <= 0 or width <= 0 or height <= 0:
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


def annotation_summary(annotations: list[dict[str, Any]]) -> str:
    """汇总每个目标已经标注的帧数、前景点和排除点数量。"""
    lines = []
    for target in ("player", "hoop", "ball"):
        positive = [
            point
            for point in annotations
            if point["target"] == target and point["label"] == 1
        ]
        negative = [
            point
            for point in annotations
            if point["target"] == target and point["label"] == 0
        ]
        frame_count = len({point["frame_idx"] for point in positive})
        lines.append(
            f"{target}: {frame_count} frames / {len(positive)} positive / "
            f"{len(negative)} negative"
        )
    return "\n\n".join(lines)


def extract_tracking_frames(info: VideoInfo, directory: Path, cv2: Any) -> None:
    """为 SAM2 抽取按数字排序的临时帧。"""
    capture = cv2.VideoCapture(info.path)
    written = 0
    while True:
        ok, image = capture.read()
        if not ok:
            break
        file_path = directory / f"{written:08d}.jpg"
        if not cv2.imwrite(str(file_path), image, [cv2.IMWRITE_JPEG_QUALITY, 95]):
            capture.release()
            raise RuntimeError(f"Could not write temporary tracking frame: {file_path}")
        written += 1
    capture.release()
    if written != info.frame_count:
        raise RuntimeError(
            f"Video reports {info.frame_count} frames, but only extracted {written}."
        )


def export_debug_video(
    info: VideoInfo,
    frame_states: list[FrameState],
    output: Path,
    masks_dir: Path,
    np: Any,
    cv2: Any,
) -> None:
    """把 mask、框和状态画回视频，并编码为浏览器可播放的 H.264 MP4。"""
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg was not found. Install it and add it to PATH.")
    capture = cv2.VideoCapture(info.path)
    if not capture.isOpened():
        raise RuntimeError(f"Could not open source video for debug export: {info.path}")

    temp_output = output.with_name(f".{output.stem}.encoding.mp4")
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
        "-an",
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
                for target, point in (
                    ("player", frame.player),
                    ("hoop", frame.hoop),
                    ("ball", frame.ball),
                ):
                    mask_path = masks_dir / target / f"{frame.frame_idx:08d}.png"
                    mask = (
                        cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                        if mask_path.is_file()
                        else None
                    )
                    if mask is not None and mask.shape == image.shape[:2]:
                        selected = mask > 0
                        if bool(np.any(selected)):
                            color = np.asarray(COLORS[target], dtype=np.float32)
                            blended = (
                                image[selected].astype(np.float32) * 0.62 + color * 0.38
                            )
                            image[selected] = np.clip(blended, 0, 255).astype(np.uint8)
                            contours, _ = cv2.findContours(
                                selected.astype(np.uint8),
                                cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE,
                            )
                            cv2.drawContours(
                                image, contours, -1, COLORS[target], 2, cv2.LINE_AA
                            )
                    if point.visible and point.bbox is not None:
                        x1, y1, x2, y2 = [round(value) for value in point.bbox]
                        x1, x2 = sorted((max(0, x1), min(info.width - 1, x2)))
                        y1, y2 = sorted((max(0, y1), min(info.height - 1, y2)))
                        cv2.rectangle(
                            image, (x1, y1), (x2, y2), COLORS[target], 3, cv2.LINE_AA
                        )
                        suffix = " interpolated" if point.interpolated else ""
                        cv2.putText(
                            image,
                            f"{target}{suffix}",
                            (x1, max(24, y1 - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.65,
                            COLORS[target],
                            2,
                            cv2.LINE_AA,
                        )

                cv2.rectangle(image, (10, 8), (470, 76), (0, 0, 0), -1)
                cv2.putText(
                    image,
                    f"frame:{frame.frame_idx}  stage:{frame.stage}",
                    (20, 34),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                visibility = "  ".join(
                    f"{name}:{'on' if point.visible else 'missing'}"
                    for name, point in (
                        ("player", frame.player),
                        ("hoop", frame.hoop),
                        ("ball", frame.ball),
                    )
                )
                cv2.putText(
                    image,
                    visibility,
                    (20, 62),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )
                process.stdin.write(image.tobytes())
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

    if return_code != 0 or written != len(frame_states):
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
        start_time = current.clip_start_frame / info.fps
        end_time = min(info.duration, (current.clip_end_frame + 1) / info.fps)
        output = output_dir / f"attack_{index:03d}.mp4"
        command = [
            "ffmpeg",
            "-y",
            "-ss",
            f"{start_time:.6f}",
            "-i",
            info.path,
            "-t",
            f"{end_time - start_time:.6f}",
            "-c:v",
            "libx264",
            "-c:a",
            "aac",
            "-movflags",
            "+faststart",
            str(output),
        ]
        result = subprocess.run(command, capture_output=True, text=True, check=False)
        if result.returncode != 0:
            raise RuntimeError(f"Clip {index} export failed: {result.stderr[-500:]}")
        output_files.append(output)
    return output_files
