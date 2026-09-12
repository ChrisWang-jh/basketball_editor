"""标注校验与按视频保存；磁盘文件不依赖浏览器会话。"""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any

from .models import TARGET_IDS, VideoInfo


def validate_annotations(
    annotations: list[dict[str, Any]], info: VideoInfo, require_all: bool = True
) -> list[dict[str, Any]]:
    if not isinstance(annotations, list):
        raise ValueError("标注必须是列表。")
    result = []
    for raw in annotations:
        if not isinstance(raw, dict) or raw.get("target") not in TARGET_IDS:
            raise ValueError("标注中存在未知目标。")
        point = dict(raw)
        frame = point.get("frame_idx")
        if (
            isinstance(frame, bool)
            or not isinstance(frame, (int, float))
            or not math.isfinite(frame)
            or int(frame) != frame
            or not 0 <= frame < info.frame_count
        ):
            raise ValueError("标注帧号超出视频范围。")
        point["frame_idx"] = int(frame)
        if point.get("label") not in (0, 1):
            raise ValueError("提示点类型必须是前景或排除点。")
        for key, limit in (("x", info.width), ("y", info.height)):
            value = point.get(key)
            if (
                not isinstance(value, (int, float))
                or not math.isfinite(value)
                or not 0 <= value < limit
            ):
                raise ValueError("标注坐标超出画面范围。")
        if "box" in point:
            box = point["box"]
            if (
                not isinstance(box, (list, tuple))
                or len(box) != 4
                or not all(
                    isinstance(v, (int, float)) and math.isfinite(v) for v in box
                )
            ):
                raise ValueError("框选区域格式错误。")
            x1, y1, x2, y2 = box
            if not (0 <= x1 < x2 < info.width and 0 <= y1 < y2 < info.height):
                raise ValueError("框选区域须位于画面内，且宽高均大于零。")
            if point["label"] != 1:
                raise ValueError("框选区域只能作为前景提示。")
        result.append(point)
    if require_all:
        names = {"player": "人物", "hoop": "篮圈", "ball": "篮球"}
        missing = [
            names[t]
            for t in TARGET_IDS
            if not any(p["target"] == t and p["label"] == 1 for p in result)
        ]
        if missing:
            raise ValueError(
                f"请先标注{'、'.join(missing)}；可以分别在不同帧标注，无需都出现在首帧。"
            )
    return result


def _fingerprint(info: VideoInfo) -> dict[str, Any]:
    stat = Path(info.path).stat()
    return {
        "path": info.path,
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "frames": info.frame_count,
        "width": info.width,
        "height": info.height,
    }


def annotation_path(root: Path, info: VideoInfo) -> Path:
    digest = hashlib.sha256(info.path.encode()).hexdigest()[:20]
    return root / "annotations" / f"{digest}.json"


def save_annotations(
    root: Path, info: VideoInfo, annotations: list[dict[str, Any]]
) -> Path:
    annotations = validate_annotations(annotations, info, require_all=False)
    path = annotation_path(root, info)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "schema_version": 1,
        "video": _fingerprint(info),
        "annotations": annotations,
    }
    fd, temporary = tempfile.mkstemp(
        prefix=".annotation_", suffix=".json", dir=path.parent
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(data, stream, ensure_ascii=False, indent=2)
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return path


def load_annotations(root: Path, info: VideoInfo) -> list[dict[str, Any]]:
    path = annotation_path(root, info)
    if not path.is_file():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("video") != _fingerprint(info):
        raise ValueError(f"{Path(info.path).name} 已发生变化，旧标注未自动套用。")
    return validate_annotations(data["annotations"], info, require_all=False)
