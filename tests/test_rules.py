from dataclasses import replace

import pytest

from basketeditor.models import AttackEvent, TrackPoint, VideoInfo
from basketeditor.rules import build_frame_states, detect_attacks, merge_events


def point(i, x, y, w=8, h=8, **kwargs):
    return TrackPoint(
        i, (x - w / 2, y - h / 2, x + w / 2, y + h / 2), (x, y), True, 0.95, **kwargs
    )


def shot(fps=30, player_disappears=False, hoop_late=False):
    n = fps * 2
    info = VideoInfo("synthetic.mp4", fps, n, 400, 300, n / fps)
    tracks = {t: [] for t in ("player", "hoop", "ball")}
    for i in range(n):
        t = i / fps
        tracks["player"].append(
            TrackPoint(i)
            if player_disappears and t >= 0.4
            else point(i, 70, 150, 60, 100)
        )
        tracks["hoop"].append(
            TrackPoint(i) if hoop_late and t < 0.55 else point(i, 200, 65, 40, 8)
        )
        if t < 0.4:
            x, y = 75, 125
        elif t < 0.6:
            r = (t - 0.4) / 0.2
            x, y = 120 + 80 * r, 90 - 45 * r
        elif t < 0.8:
            x, y = 200, 45 + (t - 0.6) / 0.2 * 60
        else:
            tracks["ball"].append(TrackPoint(i))
            continue
        tracks["ball"].append(point(i, x, y))
    return info, tracks


@pytest.mark.parametrize("fps", [15, 30, 60])
@pytest.mark.parametrize(
    "player_missing,hoop_late", [(False, False), (True, False), (True, True)]
)
def test_short_shot_and_late_hoop(fps, player_missing, hoop_late):
    info, tracks = shot(fps, player_missing, hoop_late)
    events = detect_attacks(build_frame_states(tracks, info), info)
    assert len(events) == 1
    assert events[0].outcome == "likely_made"
    assert events[0].rim_contact_frame / fps < 0.85
    assert events[0].clip_end_frame < info.frame_count


def test_rim_distance_without_visible_player():
    info, tracks = shot(player_disappears=True)
    states = build_frame_states(tracks, info)
    assert not states[20].player.visible
    assert states[20].ball_hoop_distance is not None


def test_dunk_with_ball_inside_extended_player_box():
    info, tracks = shot()
    tracks["player"] = [point(i, 180, 130, 90, 210) for i in range(info.frame_count)]
    tracks["ball"] = [
        point(i, 180, 100) if i < 15 else point(i, 200, 40 + (i - 15) * 8)
        for i in range(info.frame_count)
    ]
    events = detect_attacks(build_frame_states(tracks, info), info)
    assert len(events) == 1
    assert events[0].outcome == "likely_made"


def test_ball_missing_after_release_recovers():
    info, tracks = shot()
    for i in [14, 15, 16]:
        tracks["ball"][i] = TrackPoint(i)
    events = detect_attacks(build_frame_states(tracks, info), info)
    assert len(events) == 1
    assert events[0].outcome == "likely_made"


def test_dribble_is_not_a_shot():
    info, tracks = shot()
    tracks["ball"] = [
        point(i, 75, 125 if i % 12 < 7 else 230) for i in range(info.frame_count)
    ]
    assert detect_attacks(build_frame_states(tracks, info), info) == []


def test_no_possession_no_attribution():
    info, tracks = shot()
    tracks["player"] = [point(i, 350, 200, 30, 70) for i in range(info.frame_count)]
    assert detect_attacks(build_frame_states(tracks, info), info) == []


def test_interpolated_ball_cannot_create_a_score():
    info, tracks = shot()
    tracks["ball"] = [
        replace(p, interpolated=True) if i >= 12 else p
        for i, p in enumerate(tracks["ball"])
    ]
    assert detect_attacks(build_frame_states(tracks, info), info) == []


def test_upward_crossing_is_not_a_make():
    info, tracks = shot()
    tracks["ball"] = [point(i, 200, 100 - i * 3) for i in range(info.frame_count)]
    assert not any(s.downward_crossing for s in build_frame_states(tracks, info))


def test_camera_motion_is_not_a_falling_ball():
    info, tracks = shot()
    for i in range(info.frame_count):
        tracks["hoop"][i] = point(i, 200 + i, 50 + i, 40, 8)
        tracks["ball"][i] = point(i, 200 + i, 40 + i)
    assert not any(s.downward_crossing for s in build_frame_states(tracks, info))


def test_long_gap_does_not_make_a_crossing():
    info, tracks = shot()
    for i in range(15, 23):
        tracks["ball"][i] = TrackPoint(i)
    assert not any(s.downward_crossing for s in build_frame_states(tracks, info))


def test_cut_resets_possession_and_crossing():
    info, tracks = shot()
    assert detect_attacks(build_frame_states(tracks, info, [15]), info) == []


def test_overlapping_clips_keep_separate_attacks():
    events = [AttackEvent(5, 12, 21, 0, 75), AttackEvent(40, 48, 55, 10, 95)]
    assert len(merge_events(events)) == 2
    assert events[0].clip_end_frame == 75


def test_two_shots_survive_overlapping_buffers():
    info, tracks = shot()
    first = {t: pts[:30] for t, pts in tracks.items()}
    tracks = {
        t: [*pts, *[replace(p, frame_idx=p.frame_idx + 30) for p in pts]]
        for t, pts in first.items()
    }
    events = detect_attacks(build_frame_states(tracks, info), info)
    assert len(events) == 2
    assert events[0].clip_end_frame >= events[1].clip_start_frame
