"""轨迹补帧、平滑、几何特征计算与进攻事件状态机。"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import replace

from .models import RULES, AttackEvent, FrameState, TrackPoint, VideoInfo


def linear_interpolate(tracks: list[TrackPoint], max_missing: int) -> list[TrackPoint]:
    """只对前后都有观测的短缺失区间进行线性插值。"""
    result = [replace(point) for point in tracks]
    position = 0
    while position < len(result):
        if result[position].visible and result[position].center is not None:
            position += 1
            continue
        gap_start = position
        while position < len(result) and not (
            result[position].visible and result[position].center is not None
        ):
            position += 1
        end = position
        if gap_start == 0 or end == len(result) or end - gap_start > max_missing:
            continue
        left, right = result[gap_start - 1], result[end]
        for idx in range(gap_start, end):
            ratio = (idx - left.frame_idx) / (right.frame_idx - left.frame_idx)
            center = tuple(
                left.center[i] + (right.center[i] - left.center[i]) * ratio
                for i in range(2)
            )
            bbox = None
            if left.bbox is not None and right.bbox is not None:
                bbox = tuple(
                    left.bbox[i] + (right.bbox[i] - left.bbox[i]) * ratio
                    for i in range(4)
                )
            result[idx] = TrackPoint(
                idx,
                bbox,
                center,
                True,
                min(left.confidence, right.confidence),
                True,
            )
    return result


def smooth_ball_track(tracks: list[TrackPoint], alpha: float) -> list[TrackPoint]:
    """对每一段连续可见的篮球轨迹做 EMA 平滑。"""
    result = [replace(point) for point in tracks]
    previous: tuple[float, float] | None = None
    for point in result:
        if not point.visible or point.center is None:
            previous = None
            continue
        if previous is not None:
            point.center = tuple(
                alpha * point.center[i] + (1 - alpha) * previous[i] for i in range(2)
            )
        previous = point.center
    return result


def point_to_box_distance(
    point: tuple[float, float] | None,
    box: tuple[float, float, float, float] | None,
) -> float | None:
    """计算点到矩形的最短欧氏距离；点在矩形内时距离为零。"""
    if point is None or box is None:
        return None
    x, y = point
    x1, y1, x2, y2 = box
    return math.hypot(
        max(x1 - x, 0.0, x - x2),
        max(y1 - y, 0.0, y - y2),
    )


def sliding_confirm(raw_values: list[bool], window: int, min_count: int) -> list[bool]:
    """通过滑动窗口确认事件，抑制单帧误检和轨迹抖动。"""
    queue: deque[bool] = deque()
    hit_count = 0
    result = []
    for value in raw_values:
        queue.append(value)
        hit_count += int(value)
        if len(queue) > window:
            hit_count -= int(queue.popleft())
        result.append(hit_count >= min_count)
    return result


def build_frame_states(
    tracks: dict[str, list[TrackPoint]],
) -> list[FrameState]:
    """将三条轨迹转换为归一化距离与是否控球/碰框时序信号。"""
    # 先将球的轨迹做插值+平滑处理，几乎保证每一帧都有篮球出现
    ball_track = smooth_ball_track(
        linear_interpolate(tracks["ball"], RULES["max_short_gap_frames"]),
        RULES["smoothing_alpha"],
    )
    # 开始逐帧分析
    frame_states: list[FrameState] = []
    for frame_idx, ball in enumerate(ball_track):
        player = tracks["player"][frame_idx]
        hoop = tracks["hoop"][frame_idx]
        status = FrameState(frame_idx, player, hoop, ball)
        player_height = (
            player.bbox[3] - player.bbox[1] if player.visible and player.bbox else 0
        )
        if player_height > 0 and ball.visible:
            # 计算：人-球距离，球-筐距离
            player_ball = point_to_box_distance(ball.center, player.bbox)
            ball_hoop = (point_to_box_distance(ball.center, hoop.bbox) if hoop.visible else None)
            # 更新：每一帧frame state
            status.player_ball_distance = (player_ball / player_height if player_ball is not None else None)
            status.ball_hoop_distance = (ball_hoop / player_height if ball_hoop is not None else None)
        # 单帧“持球”条件
        status.raw_possession = (
            status.player_ball_distance is not None
            and status.player_ball_distance < RULES["possession_distance"]
        )
        # 单帧“碰框”条件
        status.raw_rim_contact = (
            status.ball_hoop_distance is not None
            and status.ball_hoop_distance < RULES["rim_contact_distance"]
        )
        frame_states.append(status)

    possession = sliding_confirm(
        [frame.raw_possession for frame in frame_states],
        RULES["possession_window"],
        RULES["possession_min_frames"],
    )
    rim_contact = sliding_confirm(
        [frame.raw_rim_contact for frame in frame_states],
        RULES["rim_contact_window"],
        RULES["rim_contact_min_frames"],
    )
    for frame, possession_value, rim_contact_value in zip(
        frame_states, possession, rim_contact
    ):
        frame.confirmed_possession = possession_value
        frame.confirmed_rim_contact = rim_contact_value
    return frame_states


def detect_attacks(
    frame_states: list[FrameState], info: VideoInfo
) -> list[AttackEvent]:
    """用状态机识别控球、持续离手和随后碰框形成的有效进攻。"""
    stage = "idle"
    possession_start = release_frame = missing_start = None
    away_frames = 0
    cooldown_until = -1
    events: list[AttackEvent] = []
    release_confirm_frames = max(
        10, math.ceil(RULES["release_confirm_seconds"] * info.fps)
    )
    max_attack_frames = math.ceil(RULES["max_attack_seconds"] * info.fps)
    max_ball_missing_frames = math.ceil(RULES["max_ball_missing_seconds"] * info.fps)

    def reset() -> tuple[str, None, None, None, int]:
        return "idle", None, None, None, 0

    def append_event(frame_idx: int) -> None:
        pre_buffer = math.ceil(RULES["pre_buffer_seconds"] * info.fps)
        post_buffer = math.ceil(RULES["post_buffer_seconds"] * info.fps)
        events.append(
            AttackEvent(
                possession_start,
                release_frame,
                frame_idx,
                max(0, possession_start - pre_buffer),
                min(info.frame_count - 1, frame_idx + post_buffer),
            )
        )

    # 开始逐帧判断
    for frame in frame_states:
        # 如果球丢失时间过程，会导致ball_lost_too_long
        if frame.ball.visible:
            missing_start = None
        elif missing_start is None:
            missing_start = frame.frame_idx
        ball_lost_too_long = (
            missing_start is not None
            and frame.frame_idx - missing_start + 1 > max_ball_missing_frames
        )

        if frame.frame_idx < cooldown_until:
            frame.stage = "cooldown"
            continue
        if stage == "cooldown":
            stage, possession_start, release_frame, missing_start, away_frames = reset()

        if stage == "idle" and frame.confirmed_possession:
            stage, possession_start = "possession", frame.frame_idx
        # 持球状态下
        elif stage == "possession":
            # 球丢失跟踪太久就cooldown
            if ball_lost_too_long:
                stage, possession_start, release_frame, missing_start, away_frames = (
                    reset()
                )
            # 观察是否运球（进攻是否持续）
            elif (
                frame.player.visible and frame.ball.visible and not frame.raw_possession
            ):
                stage, release_frame, away_frames = (
                    "pending_release",
                    frame.frame_idx,
                    1,
                )
        # 观察是否运球（进攻是否持续）状态下
        elif stage == "pending_release":
            # 球丢失太久就cooldown
            if ball_lost_too_long:
                stage, possession_start, release_frame, missing_start, away_frames = (
                    reset()
                )
            # 回到持球状态
            elif frame.raw_possession:
                stage, release_frame, away_frames = "possession", None, 0
            # 球离开时间长，算做出手
            elif frame.player.visible and frame.ball.visible:
                away_frames += 1
                if away_frames >= release_confirm_frames:
                    stage = "attack_active"
        # 出手状态下
        elif stage == "attack_active":
            # 球丢失，release_frame丢失，出手时间长而没碰框，球又回来了，都重置
            if (
                ball_lost_too_long
                or release_frame is None
                or frame.frame_idx - release_frame > max_attack_frames
                or (frame.confirmed_possession and frame.raw_possession)
            ):
                stage, possession_start, release_frame, missing_start, away_frames = (
                    reset()
                )
            # 碰框了。cooldown
            elif frame.confirmed_rim_contact and possession_start is not None:
                append_event(frame.frame_idx)
                cooldown_until = frame.frame_idx + math.ceil(
                    RULES["cooldown_seconds"] * info.fps
                )
                stage = "cooldown"

        # 低帧率下，出手确认和碰框确认可能落在同一帧。
        if (
            stage == "attack_active"
            and frame.confirmed_rim_contact
            and possession_start is not None
            and release_frame is not None
        ):
            append_event(frame.frame_idx)
            cooldown_until = frame.frame_idx + math.ceil(
                RULES["cooldown_seconds"] * info.fps
            )
            stage = "cooldown"
        frame.stage = stage
    return merge_events(events)


def merge_events(events: list[AttackEvent]) -> list[AttackEvent]:
    """合并裁剪区间重叠的重复事件。"""
    result: list[AttackEvent] = []
    for current in sorted(events, key=lambda item: item.clip_start_frame):
        if not result or current.clip_start_frame > result[-1].clip_end_frame:
            result.append(current)
        else:
            result[-1].clip_end_frame = max(
                result[-1].clip_end_frame, current.clip_end_frame
            )
    return result
