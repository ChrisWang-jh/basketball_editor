"""FPS-aware possession, release and rim crossing rules with bounded occlusion recovery."""

from __future__ import annotations

import math
from collections import deque
from dataclasses import replace

from .models import RULES, AttackEvent, FrameState, TrackPoint, VideoInfo
from .tracking import interpolate_short_gaps


def linear_interpolate(tracks: list[TrackPoint], max_missing: int) -> list[TrackPoint]:
    """Compatibility helper: only bridge bounded gaps with two visible endpoints."""
    return interpolate_short_gaps(tracks, max_missing, float("inf"))


def smooth_ball_track(tracks: list[TrackPoint], alpha: float) -> list[TrackPoint]:
    """Symmetric smoothing for display only; event rules use the original geometry."""
    result = [replace(p) for p in tracks]
    for i in range(1, len(tracks) - 1):
        neighbors = tracks[i - 1 : i + 2]
        if all(p.visible and p.center for p in neighbors) and not tracks[i].anchored:
            center = tuple(
                alpha * tracks[i].center[k]
                + (1 - alpha) * (tracks[i - 1].center[k] + tracks[i + 1].center[k]) / 2
                for k in range(2)
            )
            dx, dy = (center[k] - tracks[i].center[k] for k in range(2))
            box = tracks[i].bbox
            result[i] = replace(
                tracks[i],
                center=center,
                bbox=(box[0] + dx, box[1] + dy, box[2] + dx, box[3] + dy)
                if box
                else None,
            )
    return result


def point_to_box_distance(
    point: tuple[float, float] | None, box: tuple[float, float, float, float] | None
) -> float | None:
    if point is None or box is None:
        return None
    x, y = point
    x1, y1, x2, y2 = box
    return math.hypot(max(x1 - x, 0.0, x - x2), max(y1 - y, 0.0, y - y2))


def sliding_confirm(raw_values: list[bool], window: int, min_count: int) -> list[bool]:
    queue: deque[bool] = deque()
    count = 0
    result = []
    for value in raw_values:
        queue.append(value)
        count += int(value)
        if len(queue) > window:
            count -= int(queue.popleft())
        result.append(count >= min_count)
    return result


def _observed(point: TrackPoint) -> bool:
    return point.visible and point.center is not None and not point.interpolated


def _crosses_rim(previous: FrameState, current: FrameState, fps: float) -> bool:
    gap = current.frame_idx - previous.frame_idx
    if gap <= 0 or gap > max(1, round(RULES["max_crossing_gap_seconds"] * fps)):
        return False
    if not all(
        _observed(p) for p in (previous.ball, current.ball, previous.hoop, current.hoop)
    ):
        return False
    if not previous.hoop.bbox or not current.hoop.bbox:
        return False
    # Relative coordinates cancel a moving camera/hoop. Pixel displacement alone
    # would incorrectly classify camera tilts as a falling basketball.
    px, py = (previous.ball.center[k] - previous.hoop.center[k] for k in range(2))
    cx, cy = (current.ball.center[k] - current.hoop.center[k] for k in range(2))
    width = (
        current.hoop.bbox[2]
        - current.hoop.bbox[0]
        + previous.hoop.bbox[2]
        - previous.hoop.bbox[0]
    ) / 2
    if width <= 0 or not (py < 0 <= cy) or cy - py < max(1, width * 0.05):
        return False
    # Bound extrapolation: only connect samples in the vicinity of this rim.
    if max(abs(py), abs(cy)) > width * 2.5 or abs(cx - px) > width * 2:
        return False
    ratio = -py / (cy - py)
    crossing_x = px + (cx - px) * ratio
    return abs(crossing_x) <= width * 0.45


def build_frame_states(
    tracks: dict[str, list[TrackPoint]],
    info: VideoInfo | None = None,
    scene_cuts: list[int] | None = None,
) -> list[FrameState]:
    """Use prepared tracks; hoop geometry never depends on player visibility."""
    count = len(tracks["ball"])
    if any(len(tracks[t]) != count for t in ("player", "hoop")):
        raise ValueError("Target trajectories must have equal lengths")
    fps = info.fps if info else 30.0
    cuts = {0, *(scene_cuts or [])}
    states = []
    last_pair = None
    possession_window: deque[bool] = deque(
        maxlen=max(1, round(fps * RULES["possession_window_seconds"]))
    )
    rim_window: deque[bool] = deque(
        maxlen=max(1, round(fps * RULES["rim_contact_window_seconds"]))
    )
    possession_min = max(1, round(fps * RULES["possession_confirm_seconds"]))
    rim_min = max(1, round(fps * RULES["rim_contact_confirm_seconds"]))
    for i in range(count):
        player, hoop, ball = (tracks[t][i] for t in ("player", "hoop", "ball"))
        state = FrameState(i, player, hoop, ball, scene_start=i in cuts)
        if i in cuts:
            possession_window.clear()
            rim_window.clear()
            last_pair = None
        if player.visible and player.bbox and ball.visible:
            height = player.bbox[3] - player.bbox[1]
            distance = point_to_box_distance(ball.center, player.bbox)
            if height > 0 and distance is not None:
                state.player_ball_distance = distance / height
        if hoop.visible and hoop.bbox and ball.visible:
            width = hoop.bbox[2] - hoop.bbox[0]
            distance = point_to_box_distance(ball.center, hoop.bbox)
            if width > 0 and distance is not None:
                state.ball_hoop_distance = distance / width
        state.raw_possession = (
            state.player_ball_distance is not None
            and state.player_ball_distance < RULES["possession_distance"]
        )
        # Estimated ball coordinates can keep a pending shot alive, but cannot
        # establish possession, contact or a made-shot candidate by themselves.
        state.raw_rim_contact = (
            state.ball_hoop_distance is not None
            and state.ball_hoop_distance < RULES["rim_contact_distance"]
            and _observed(ball)
        )
        possession_window.append(
            state.raw_possession and _observed(ball) and _observed(player)
        )
        rim_window.append(state.raw_rim_contact)
        state.confirmed_possession = sum(possession_window) >= possession_min
        state.confirmed_rim_contact = (
            state.raw_rim_contact and sum(rim_window) >= rim_min
        )
        if _observed(ball) and _observed(hoop):
            if last_pair is not None:
                state.downward_crossing = _crosses_rim(last_pair, state, fps)
            last_pair = state
        states.append(state)
    return states


def detect_attacks(
    frame_states: list[FrameState], info: VideoInfo
) -> list[AttackEvent]:
    """Remember a shot across brief absence and inspect the rim from release onward.

    A 2D downward crossing is a likely make, not a calibrated score. Rim proximity
    alone remains a separate outcome. Individual attacks are never merged merely
    because their exported time ranges overlap.
    """
    events: list[AttackEvent] = []
    stage = "idle"
    possession_start = None
    last_possession = None
    release = None
    last_ball = None
    last_player = None
    cooldown_until = -1
    contact_frame = None
    active_event = None
    fps = info.fps
    release_confirm = max(1, round(RULES["release_confirm_seconds"] * fps))
    min_flight = max(1, round(RULES["min_shot_seconds"] * fps))
    max_attack = max(1, round(RULES["max_attack_seconds"] * fps))
    max_missing = max(1, round(RULES["max_ball_missing_seconds"] * fps))
    scene_start = 0
    scene_ends = [
        s.frame_idx - 1 for s in frame_states if s.scene_start and s.frame_idx > 0
    ] + [info.frame_count - 1]
    scene_index = 0

    def reset() -> None:
        nonlocal stage, possession_start, last_possession, release, contact_frame
        stage, possession_start, last_possession, release, contact_frame = (
            "idle",
            None,
            None,
            None,
            None,
        )

    for frame in frame_states:
        i = frame.frame_idx
        if frame.scene_start:
            reset()
            last_ball = last_player = active_event = None
            cooldown_until = -1
            scene_start = i
            while scene_index < len(scene_ends) - 1 and scene_ends[scene_index] < i:
                scene_index += 1
        if _observed(frame.ball):
            last_ball = i
        if _observed(frame.player):
            last_player = frame.player
        # A contact followed by a downward crossing belongs to the same shot.
        if active_event is not None and i - active_event.rim_contact_frame <= round(
            0.6 * fps
        ):
            if frame.downward_crossing:
                active_event.outcome = "likely_made"
                active_event.evidence = "downward_rim_crossing"
        else:
            active_event = None
        if i < cooldown_until:
            frame.stage = "cooldown"
            continue
        if stage == "cooldown":
            reset()
        if last_ball is not None and i - last_ball > max_missing:
            reset()
        if (
            stage == "possession"
            and last_possession is not None
            and i - last_possession > max_missing
        ):
            reset()
        if release is not None and i - release > max_attack:
            reset()

        if stage == "idle":
            if (
                frame.confirmed_possession
                and frame.raw_possession
                and _observed(frame.ball)
            ):
                stage = "possession"
                possession_start = max(
                    scene_start,
                    i - max(1, round(fps * RULES["possession_confirm_seconds"])) + 1,
                )
                last_possession = i
        elif stage == "possession":
            if frame.downward_crossing:
                # A dunk/close layup can cross the rim while still inside the
                # shooter's extended bounding box. A spatial "away" is optional
                # when direct crossing evidence exists after confirmed possession.
                release = i
                stage = "pending_release"
            elif frame.raw_possession and _observed(frame.ball):
                last_possession = i
            elif _observed(frame.ball):
                # When the player leaves the image during flight, retain their last
                # box briefly to distinguish a release from a ball moving nearby.
                reference = frame.player if frame.player.visible else last_player
                away = False
                if (
                    reference
                    and reference.bbox
                    and i - reference.frame_idx <= max_missing
                ):
                    height = reference.bbox[3] - reference.bbox[1]
                    distance = point_to_box_distance(frame.ball.center, reference.bbox)
                    away = (
                        height > 0
                        and distance is not None
                        and distance / height >= RULES["possession_distance"]
                    )
                if away or frame.raw_rim_contact or frame.downward_crossing:
                    release = min(
                        i, (last_possession + 1) if last_possession is not None else i
                    )
                    stage = "pending_release"
        if stage in {"pending_release", "attack_active"} and release is not None:
            age = i - release + 1
            # Check contact BEFORE a possession reset or a release timer. This is
            # critical for layups and low-frame-rate shots.
            if frame.confirmed_rim_contact or frame.downward_crossing:
                contact_frame = i
            evidence_is_recent = contact_frame is not None and i - contact_frame <= max(
                1, round(0.15 * fps)
            )
            if (
                (age >= min_flight or frame.downward_crossing)
                and evidence_is_recent
                and possession_start is not None
            ):
                contact = contact_frame
                event = AttackEvent(
                    possession_start,
                    release,
                    contact,
                    max(
                        scene_start,
                        possession_start - math.ceil(RULES["pre_buffer_seconds"] * fps),
                    ),
                    min(
                        scene_ends[scene_index],
                        contact + math.ceil(RULES["post_buffer_seconds"] * fps),
                    ),
                    "likely_made" if frame.downward_crossing else "rim_contact",
                    "downward_rim_crossing"
                    if frame.downward_crossing
                    else "rim_proximity",
                )
                events.append(event)
                active_event = event
                cooldown_until = i + max(1, round(RULES["cooldown_seconds"] * fps))
                stage = "cooldown"
            elif (
                frame.raw_possession and _observed(frame.ball) and frame.player.visible
            ):
                stage, release, contact_frame = "possession", None, None
                last_possession = i
            elif age >= release_confirm:
                stage = "attack_active"
        frame.stage = stage
    return merge_events(events)


def merge_events(events: list[AttackEvent]) -> list[AttackEvent]:
    """Deduplicate identical releases only; overlapping clips are distinct attacks."""
    unique: dict[tuple[int, int], AttackEvent] = {}
    for event in events:
        key = (event.possession_start_frame, event.release_frame)
        if key not in unique or event.outcome == "likely_made":
            unique[key] = replace(event)
    return sorted(unique.values(), key=lambda e: e.release_frame)
