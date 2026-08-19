"""递归发现并可靠拼接 attack 视频片段。"""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from datetime import datetime
from fractions import Fraction
from pathlib import Path
from typing import Any

from .video import VIDEO_EXTENSIONS


def discover_attack_clips(output_root: Path) -> list[Path]:
    """递归返回文件名以 attack 开头的视频片段。"""
    output_root = output_root.expanduser().resolve()
    if not output_root.is_dir():
        raise NotADirectoryError(f"Output directory not found: {output_root}")
    clips = [
        path.resolve()
        for path in output_root.rglob("*")
        if path.is_file()
        and path.name.casefold().startswith("attack")
        and path.suffix.lower() in VIDEO_EXTENSIONS
    ]
    return sorted(
        clips, key=lambda path: path.relative_to(output_root).as_posix().casefold()
    )


def _probe(path: Path, ffprobe: str) -> dict[str, Any]:
    command = [
        ffprobe,
        "-v",
        "error",
        "-show_streams",
        "-show_format",
        "-of",
        "json",
        str(path),
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"Could not inspect {path.name}: {result.stderr[-500:]}")
    try:
        data = json.loads(result.stdout)
        video = next(
            stream for stream in data["streams"] if stream["codec_type"] == "video"
        )
    except (KeyError, StopIteration, TypeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"No readable video stream in {path}") from exc
    duration = float(
        data.get("format", {}).get("duration") or video.get("duration") or 0
    )
    if duration <= 0:
        raise RuntimeError(f"Invalid duration in {path}")
    return {
        "width": int(video["width"]),
        "height": int(video["height"]),
        "fps": video.get("avg_frame_rate") or video.get("r_frame_rate") or "30/1",
        "duration": duration,
        "has_audio": any(
            stream.get("codec_type") == "audio" for stream in data.get("streams", [])
        ),
    }


def _run_ffmpeg(command: list[str], description: str) -> None:
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"{description} failed: {result.stderr[-1000:]}")


def _concat_line(path: Path) -> str:
    escaped = str(path.resolve()).replace("'", "'\\''")
    return f"file '{escaped}'"


def stitch_clips(clips: list[Path], output_root: Path, progress: Any = None) -> Path:
    """按给定顺序标准化并拼接片段，输出浏览器兼容的 MP4。"""
    if not clips:
        raise ValueError("Please select at least one attack clip.")
    ffmpeg = shutil.which("ffmpeg")
    ffprobe = shutil.which("ffprobe")
    if ffmpeg is None or ffprobe is None:
        raise RuntimeError("Both ffmpeg and ffprobe must be installed and in PATH.")

    output_root = output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    resolved_clips = [Path(path).expanduser().resolve() for path in clips]
    for clip in resolved_clips:
        if not clip.is_file():
            raise FileNotFoundError(f"Selected clip no longer exists: {clip}")

    first = _probe(resolved_clips[0], ffprobe)
    width = int(first["width"]) + int(first["width"]) % 2
    height = int(first["height"]) + int(first["height"]) % 2
    try:
        fps_fraction = Fraction(str(first["fps"]))
        if fps_fraction <= 0:
            raise ValueError
    except (ValueError, ZeroDivisionError):
        fps_fraction = Fraction(30, 1)
    fps = f"{fps_fraction.numerator}/{fps_fraction.denominator}"
    timestamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S_%f")
    output = output_root / f"stitched_{timestamp}.mp4"
    partial = output_root / f".stitched_{timestamp}.encoding.mp4"

    with tempfile.TemporaryDirectory(prefix="stitch_", dir=output_root) as temp:
        temp_dir = Path(temp)
        normalized: list[Path] = []
        for index, clip in enumerate(resolved_clips):
            metadata = _probe(clip, ffprobe)
            normalized_path = temp_dir / f"segment_{index:05d}.mp4"
            video_filter = (
                f"scale={width}:{height}:force_original_aspect_ratio=decrease,"
                f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:color=black,"
                f"fps={fps},setsar=1"
            )
            command = [ffmpeg, "-y", "-loglevel", "error", "-i", str(clip)]
            if metadata["has_audio"]:
                command += [
                    "-map",
                    "0:v:0",
                    "-map",
                    "0:a:0",
                    "-vf",
                    video_filter,
                    "-af",
                    "aresample=async=1:first_pts=0,apad",
                ]
            else:
                command += [
                    "-f",
                    "lavfi",
                    "-t",
                    f"{metadata['duration']:.6f}",
                    "-i",
                    "anullsrc=channel_layout=stereo:sample_rate=48000",
                    "-map",
                    "0:v:0",
                    "-map",
                    "1:a:0",
                    "-vf",
                    video_filter,
                ]
            command += [
                "-c:v",
                "libx264",
                "-preset",
                "veryfast",
                "-crf",
                "20",
                "-pix_fmt",
                "yuv420p",
                "-c:a",
                "aac",
                "-ar",
                "48000",
                "-ac",
                "2",
                "-shortest",
                "-movflags",
                "+faststart",
                str(normalized_path),
            ]
            if progress:
                progress(
                    0.05 + 0.80 * index / len(resolved_clips),
                    desc=f"Normalizing clip {index + 1}/{len(resolved_clips)}",
                )
            _run_ffmpeg(command, f"Normalizing {clip.name}")
            normalized.append(normalized_path)

        concat_file = temp_dir / "concat.txt"
        concat_file.write_text(
            "\n".join(_concat_line(path) for path in normalized) + "\n",
            encoding="utf-8",
        )
        if progress:
            progress(0.90, desc="Concatenating selected clips")
        try:
            _run_ffmpeg(
                [
                    ffmpeg,
                    "-y",
                    "-loglevel",
                    "error",
                    "-f",
                    "concat",
                    "-safe",
                    "0",
                    "-i",
                    str(concat_file),
                    "-c",
                    "copy",
                    "-movflags",
                    "+faststart",
                    str(partial),
                ],
                "Concatenation",
            )
        except Exception:
            partial.unlink(missing_ok=True)
            raise

    if not partial.is_file() or partial.stat().st_size == 0:
        partial.unlink(missing_ok=True)
        raise RuntimeError("FFmpeg did not create the stitched video.")
    partial.replace(output)
    if progress:
        progress(1.0, desc="Done")
    return output
