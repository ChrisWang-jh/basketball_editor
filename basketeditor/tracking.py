"""可见性、几何异常过滤和有限时长恢复；不将长缺失伪装成观测。"""

from __future__ import annotations

import math
from dataclasses import replace
from statistics import median

from .models import TARGET_IDS, TrackingOptions, TrackPoint, VideoInfo

IDENTITY_VETO = {"identity_rejected", "pending", "lost"}


def interpolate_short_gaps(
    points: list[TrackPoint], maximum: int, speed_limit: float
) -> list[TrackPoint]:
    result = [replace(p) for p in points]
    left = None
    for i, point in enumerate(result):
        if point.source in IDENTITY_VETO:
            left = None
            continue
        if not point.visible or point.center is None:
            continue
        if left is not None and 0 < i - left - 1 <= maximum:
            start = result[left]
            if math.dist(start.center, point.center) / (i - left) <= speed_limit:
                for j in range(left + 1, i):
                    ratio = (j - left) / (i - left)
                    center = tuple(
                        a + (b - a) * ratio for a, b in zip(start.center, point.center)
                    )
                    bbox = (
                        tuple(
                            a + (b - a) * ratio for a, b in zip(start.bbox, point.bbox)
                        )
                        if start.bbox and point.bbox
                        else None
                    )
                    result[j] = TrackPoint(
                        result[j].frame_idx,
                        bbox,
                        center,
                        True,
                        min(start.confidence, point.confidence) * 0.5,
                        True,
                        "interpolated",
                    )
        left = i
    return result


def _clean_segment(
    points: list[TrackPoint], target: str, info: VideoInfo, options: TrackingOptions
) -> list[TrackPoint]:
    result = [replace(p) for p in points]
    anchors = [p for p in points if p.anchored and p.visible and p.bbox]
    anchor_areas = [(p.bbox[2] - p.bbox[0]) * (p.bbox[3] - p.bbox[1]) for p in anchors]
    typical_area = median(anchor_areas) if anchor_areas else None
    diagonal = math.hypot(info.width, info.height)
    speed = diagonal * {"player": 1.0, "hoop": 0.8, "ball": 4.0}[target] / info.fps
    last = None
    for i, point in enumerate(result):
        if not point.visible or not point.center or not point.bbox:
            continue
        x1, y1, x2, y2 = point.bbox
        area = (x2 - x1) * (y2 - y1)
        valid = all(
            math.isfinite(v) for v in (*point.bbox, *point.center, point.confidence)
        )
        valid &= 0 <= x1 < x2 <= info.width and 0 <= y1 < y2 <= info.height
        valid &= point.confidence >= options.min_confidence
        if typical_area and not point.anchored:
            lower, upper = (0.04, 12) if target == "ball" else (0.08, 8)
            valid &= lower * typical_area <= area <= upper * typical_area
        # A one-frame jump is rejected; after an occlusion new observations can reacquire.
        if last is not None and not point.anchored and point.source != "reacquired":
            previous = result[last]
            gap = i - last
            if gap <= max(1, round(info.fps * 0.3)):
                valid &= (
                    math.dist(point.center, previous.center)
                    <= speed * gap + math.sqrt(max(area, 0)) * 0.5
                )
        if valid or (point.anchored and area > 0):
            last = i
        else:
            result[i] = TrackPoint(point.frame_idx, source="rejected")
    gap_seconds = getattr(options, f"{target}_gap_seconds")
    return interpolate_short_gaps(result, round(info.fps * gap_seconds), speed)


def _stabilize_hoop(
    points: list[TrackPoint], mode: str, fps: float
) -> list[TrackPoint]:
    if mode == "moving":
        return points
    result = [replace(p) for p in points]
    if mode == "fixed":
        anchor = next((p for p in points if p.anchored and p.visible and p.bbox), None)
        if anchor is None:
            return result
        # Fixed is an explicit user constraint, limited to this camera shot. Do not
        # create a hoop before its first observed presence in the shot.
        first = next(
            (
                i
                for i, p in enumerate(points)
                if p.visible
                and p.center
                and math.dist(p.center, anchor.center)
                <= max(anchor.bbox[2] - anchor.bbox[0], 8)
            ),
            len(points),
        )
        for i in range(first, len(points)):
            if points[i].source in IDENTITY_VETO:
                continue
            result[i] = replace(
                anchor,
                frame_idx=points[i].frame_idx,
                source="fixed",
                interpolated=not points[i].visible,
                anchored=points[i].anchored,
            )
        return result
    radius = max(1, round(fps * 0.07))
    for i, point in enumerate(points):
        if not point.visible or not point.bbox or point.anchored:
            continue
        # Symmetric median removes isolated jitter without EMA's time lag. Do not
        # blend across missing intervals; real pans remain free to move.
        neighbors = points[max(0, i - radius) : i + radius + 1]
        if not all(p.visible and p.bbox for p in neighbors):
            continue
        bbox = tuple(median(p.bbox[k] for p in neighbors) for k in range(4))
        result[i] = replace(
            point,
            bbox=bbox,
            center=((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2),
            source="stabilized",
        )
    return result


def prepare_tracks(
    tracks: dict[str, list[TrackPoint]],
    info: VideoInfo,
    options: TrackingOptions | None = None,
    scene_cuts: list[int] | None = None,
) -> dict[str, list[TrackPoint]]:
    options = options or TrackingOptions()
    boundaries = sorted({0, info.frame_count, *(scene_cuts or [])})
    result = {target: [] for target in TARGET_IDS}
    for target in TARGET_IDS:
        if len(tracks[target]) != info.frame_count:
            raise ValueError(
                f"{target} trajectory length does not match the decoded video"
            )
        for start, end in zip(boundaries, boundaries[1:]):
            segment = _clean_segment(tracks[target][start:end], target, info, options)
            if target == "hoop":
                segment = _stabilize_hoop(segment, options.hoop_mode, info.fps)
            result[target].extend(segment)
    return result


def tracking_diagnostics(tracks: dict[str, list[TrackPoint]], info: VideoInfo) -> dict:
    result = {}
    for target, points in tracks.items():
        gaps = []
        start = None
        for i in range(len(points) + 1):
            missing = i < len(points) and not points[i].visible
            if missing and start is None:
                start = i
            if not missing and start is not None:
                gaps.append(
                    {
                        "start_frame": start,
                        "end_frame": i - 1,
                        "seconds": (i - start) / info.fps,
                    }
                )
                start = None
        result[target] = {
            "observed_frames": sum(p.visible and not p.interpolated for p in points),
            "estimated_frames": sum(p.visible and p.interpolated for p in points),
            "missing_frames": sum(not p.visible for p in points),
            "rejected_frames": sum(p.source == "rejected" for p in points),
            "identity_rejected_frames": sum(
                p.source == "identity_rejected" for p in points
            ),
            "pending_frames": sum(p.source == "pending" for p in points),
            "reacquired_frames": [
                p.frame_idx for p in points if p.source == "reacquired"
            ],
            "gaps": gaps,
        }
    return result
