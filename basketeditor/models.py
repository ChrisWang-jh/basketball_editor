"""跨模块共享的数据模型与配置。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parent.parent

TARGET_IDS = {"player": 1, "hoop": 2, "ball": 3}
ID_TO_TARGET = {value: key for key, value in TARGET_IDS.items()}
COLORS = {
    "player": (52, 211, 80),
    "hoop": (70, 110, 255),
    "ball": (255, 210, 50),
}

# 持球距离按人物高度归一化，篮圈判定使用篮圈自身尺寸，互不依赖。
RULES = {
    "possession_distance": 0.18,
    "possession_window_seconds": 1 / 3,
    "possession_confirm_seconds": 0.16,
    "rim_contact_distance": 0.20,
    "rim_contact_window_seconds": 1 / 6,
    "rim_contact_confirm_seconds": 1 / 30,
    "release_confirm_seconds": 0.12,
    "max_attack_seconds": 6.0,
    "max_ball_missing_seconds": 1.25,
    "pre_buffer_seconds": 1.0,
    "post_buffer_seconds": 1.8,
    "cooldown_seconds": 0.4,
    "max_crossing_gap_seconds": 0.20,
    "min_shot_seconds": 0.06,
}


@dataclass(frozen=True)
class ReIDOptions:
    """Local identity models; manual templates never expire during a video."""

    enabled: bool = True
    repo: str = str(REPO_DIR / "third_party" / "dinov3")
    checkpoint: str = str(REPO_DIR / "checkpoints" / "dinov3_vits16.pth")
    image_size: int = 448
    search_candidates: int = 6
    confirm_frames: int = 2
    audit_interval: int = 10

    def __post_init__(self) -> None:
        if self.image_size < 224 or self.image_size % 16:
            raise ValueError("DINO image size must be a multiple of 16, at least 224")
        if not 2 <= self.search_candidates <= 16:
            raise ValueError("Search candidates must be between 2 and 16")
        if not 2 <= self.confirm_frames <= 10 or self.audit_interval < 1:
            raise ValueError("Invalid identity confirmation interval")


@dataclass(frozen=True)
class TrackingOptions:
    """自动模式抑制局部抖动；固定模式仅用于固定机位。"""

    hoop_mode: str = "auto"
    ball_gap_seconds: float = 0.16
    player_gap_seconds: float = 0.30
    hoop_gap_seconds: float = 0.35
    min_confidence: float = 0.55

    def __post_init__(self) -> None:
        if self.hoop_mode not in {"auto", "moving", "fixed"}:
            raise ValueError("Unknown hoop mode")
        if not 0 <= self.min_confidence <= 1:
            raise ValueError("Confidence must be between 0 and 1")
        for value in (
            self.ball_gap_seconds,
            self.player_gap_seconds,
            self.hoop_gap_seconds,
        ):
            if not 0 <= value <= 2:
                raise ValueError("Gap duration must be between 0 and 2 seconds")


@dataclass
class VideoInfo:
    path: str
    fps: float
    frame_count: int
    width: int
    height: int
    duration: float
    frame_times: list[float] | None = None

    def frame_time(self, frame_idx: int, end: bool = False) -> float:
        index = frame_idx + int(end)
        if self.frame_times and 0 <= index < len(self.frame_times):
            return self.frame_times[index]
        return (
            min(self.duration, index / self.fps)
            if not self.frame_times
            else self.duration
        )


@dataclass
class TrackPoint:
    frame_idx: int
    bbox: tuple[float, float, float, float] | None = None
    center: tuple[float, float] | None = None
    visible: bool = False
    confidence: float = 0.0
    interpolated: bool = False
    source: str = "sam2"
    anchored: bool = False
    identity_score: float | None = None
    identity_margin: float | None = None
    reason: str = ""


@dataclass
class FrameState:
    frame_idx: int
    player: TrackPoint
    hoop: TrackPoint
    ball: TrackPoint
    player_ball_distance: float | None = None
    ball_hoop_distance: float | None = None
    raw_possession: bool = False
    confirmed_possession: bool = False
    raw_rim_contact: bool = False
    confirmed_rim_contact: bool = False
    stage: str = "idle"
    downward_crossing: bool = False
    scene_start: bool = False


@dataclass
class AttackEvent:
    possession_start_frame: int
    release_frame: int
    rim_contact_frame: int
    clip_start_frame: int
    clip_end_frame: int
    outcome: str = "rim_contact"
    evidence: str = "rim_proximity"
