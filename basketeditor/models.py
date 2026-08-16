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

# 距离统一使用当前主人公外接框高度归一化，避免分辨率变化影响阈值。
RULES = {
    "possession_distance": 0.18, # 持球条件threshold
    "rim_contact_distance": 0.10, # 碰框条件threshold
    "possession_window": 10, # 
    "possession_min_frames": 6, # 持球确认最小帧数（窗口范围内）
    "rim_contact_window": 5, # 
    "rim_contact_min_frames": 2, # 碰框确认最小帧数（窗口范围内）
    "release_confirm_seconds": 0.65, # 确认进攻是否持续
    "max_attack_seconds": 6.0, # 最长持球时间
    "max_ball_missing_seconds": 0.75, # 最长篮球丢失时间
    "pre_buffer_seconds": 1.0, # 进攻开始前缓冲时间
    "post_buffer_seconds": 1.8, # 进攻结束后缓冲时间
    "cooldown_seconds": 2.5,
    "max_short_gap_frames": 5, # 规定篮球丢失追踪的最大秒数
    "smoothing_alpha": 0.45,
}


@dataclass
class VideoInfo:
    path: str
    fps: float
    frame_count: int
    width: int
    height: int
    duration: float


@dataclass
class TrackPoint:
    frame_idx: int
    bbox: tuple[float, float, float, float] | None = None
    center: tuple[float, float] | None = None
    visible: bool = False
    confidence: float = 0.0
    interpolated: bool = False


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


@dataclass
class AttackEvent:
    possession_start_frame: int
    release_frame: int
    rim_contact_frame: int
    clip_start_frame: int
    clip_end_frame: int
