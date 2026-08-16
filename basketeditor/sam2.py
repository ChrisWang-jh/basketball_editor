"""SAM2 模型加载、静态帧预览和整段视频追踪。"""

from __future__ import annotations

import gc
import importlib
import sys
from contextlib import nullcontext, suppress
from pathlib import Path
from typing import Any

from .models import ID_TO_TARGET, TARGET_IDS, TrackPoint, VideoInfo


def select_device(torch: Any, requested_device: str) -> str:
    """未显式指定设备时优先使用 CUDA，其次 MPS，最后 CPU。"""
    if requested_device != "auto":
        return requested_device
    if torch.cuda.is_available():
        return "cuda"
    if getattr(getattr(torch.backends, "mps", None), "is_available", lambda: False)():
        return "mps"
    return "cpu"


def _configure_sam2_repo(sam2_repo: Path | None) -> None:
    if sam2_repo is None:
        return
    if not sam2_repo.is_dir():
        raise FileNotFoundError(f"SAM2 repository not found: {sam2_repo}")
    repo_path = str(sam2_repo.resolve())
    if repo_path not in sys.path:
        sys.path.insert(0, repo_path)


def _load_torch(sam2_repo: Path | None) -> Any:
    _configure_sam2_repo(sam2_repo)
    try:
        return importlib.import_module("torch")
    except (ImportError, OSError) as exc:
        raise RuntimeError(
            "Could not import torch. See the setup instructions."
        ) from exc


def _autocast(torch: Any, device: str) -> Any:
    if device.startswith("cuda"):
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return nullcontext()


def mask_to_track_point(frame_idx: int, mask: Any, np: Any) -> TrackPoint:
    """将 SAM2 掩码转换成规则层使用的外接框和中心点。"""
    if hasattr(mask, "detach"):
        mask = mask.detach()
    if hasattr(mask, "float"):
        mask = mask.float()
    if hasattr(mask, "cpu"):
        mask = mask.cpu()
    if hasattr(mask, "numpy"):
        mask = mask.numpy()
    array = np.squeeze(np.asarray(mask))
    if array.ndim != 2:
        raise RuntimeError(f"SAM2 returned an invalid mask shape: {array.shape}")
    y, x = np.nonzero(array > 0.0)
    if x.size < 4:
        return TrackPoint(frame_idx)
    x1, y1, x2, y2 = (
        float(x.min()),
        float(y.min()),
        float(x.max()),
        float(y.max()),
    )
    foreground_values = np.clip(array[array > 0.0], -60.0, 60.0)
    confidence = float((1.0 / (1.0 + np.exp(-foreground_values))).mean())
    return TrackPoint(
        frame_idx,
        (x1, y1, x2, y2),
        ((x1 + x2) / 2, (y1 + y2) / 2),
        True,
        confidence,
    )


def run_sam2_tracking(
    frames_dir: Path,
    info: VideoInfo,
    annotations: list[dict[str, Any]],
    checkpoint: Path,
    model_config: str,
    device_setting: str,
    sam2_repo: Path | None,
    np: Any,
) -> dict[str, list[TrackPoint]]:
    """注入三个目标的提示点，并对视频进行前向和反向掩码传播。"""
    if not checkpoint.is_file():
        raise FileNotFoundError(f"SAM2 checkpoint not found: {checkpoint}")
    torch = _load_torch(sam2_repo)
    try:
        builder = importlib.import_module("sam2.build_sam").build_sam2_video_predictor
    except (ImportError, OSError) as exc:
        raise RuntimeError(
            "Could not import the official SAM2 package. See the setup instructions."
        ) from exc

    device = select_device(torch, device_setting)
    try:
        predictor = builder(model_config, str(checkpoint.resolve()), device=device)
    except Exception as exc:
        raise RuntimeError(
            f"Could not build the SAM2 predictor. Check model config {model_config!r}."
        ) from exc

    track_dict: dict[str, dict[int, TrackPoint]] = {target: {} for target in TARGET_IDS}

    def consume_output(output: tuple[Any, Any, Any]) -> None:
        frame_idx_value, obj_ids, mask_values = output
        frame_idx = int(frame_idx_value)
        if hasattr(obj_ids, "detach"):
            obj_ids = obj_ids.detach().cpu().numpy()
        for position, raw_obj_id in enumerate(np.asarray(obj_ids).reshape(-1).tolist()):
            target = ID_TO_TARGET.get(int(raw_obj_id))
            if target is None:
                continue
            candidate = mask_to_track_point(frame_idx, mask_values[position], np)
            current = track_dict[target].get(frame_idx)
            if (
                current is None
                or (candidate.visible and not current.visible)
                or candidate.confidence > current.confidence
            ):
                track_dict[target][frame_idx] = candidate

    groups: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for point in annotations:
        groups.setdefault((point["target"], int(point["frame_idx"])), []).append(point)
    annotated_frames = [frame_idx for _, frame_idx in groups]

    with torch.inference_mode(), _autocast(torch, device):
        inference_state = predictor.init_state(
            video_path=str(frames_dir),
            offload_video_to_cpu=True,
            offload_state_to_cpu=False,
        )
        predictor.reset_state(inference_state)
        for (target, frame_idx), points in sorted(
            groups.items(), key=lambda item: item[0][1]
        ):
            coords = np.asarray(
                [[point["x"], point["y"]] for point in points], dtype=np.float32
            )
            labels = np.asarray([point["label"] for point in points], dtype=np.int32)
            consume_output(
                predictor.add_new_points_or_box(
                    inference_state=inference_state,
                    frame_idx=frame_idx,
                    obj_id=TARGET_IDS[target],
                    points=coords,
                    labels=labels,
                )
            )
        for output in predictor.propagate_in_video(
            inference_state, start_frame_idx=min(annotated_frames), reverse=False
        ):
            consume_output(output)
        for output in predictor.propagate_in_video(
            inference_state, start_frame_idx=max(annotated_frames), reverse=True
        ):
            consume_output(output)

    return {
        target: [
            track_dict[target].get(frame_idx, TrackPoint(frame_idx))
            for frame_idx in range(info.frame_count)
        ]
        for target in TARGET_IDS
    }


class Sam2Preview:
    """懒加载并复用 SAM2 静态图像预测器。"""

    def __init__(
        self,
        checkpoint: Path,
        model_config: str,
        device_setting: str,
        sam2_repo: Path | None,
        np: Any,
    ) -> None:
        self.checkpoint = checkpoint
        self.model_config = model_config
        self.device_setting = device_setting
        self.sam2_repo = sam2_repo
        self.np = np
        self.predictor: Any = None
        self.torch: Any = None
        self.device: str | None = None
        self.frame_idx: int | None = None

    def _ensure_predictor(self) -> tuple[Any, Any, str]:
        if self.predictor is not None:
            return self.predictor, self.torch, self.device
        if not self.checkpoint.is_file():
            raise FileNotFoundError(f"SAM2 checkpoint not found: {self.checkpoint}")
        torch = _load_torch(self.sam2_repo)
        try:
            build_sam2 = importlib.import_module("sam2.build_sam").build_sam2
            predictor_class = importlib.import_module(
                "sam2.sam2_image_predictor"
            ).SAM2ImagePredictor
        except (ImportError, OSError) as exc:
            raise RuntimeError(
                "Could not import SAM2ImagePredictor. Install the official SAM2 package."
            ) from exc
        device = select_device(torch, self.device_setting)
        try:
            model = build_sam2(
                self.model_config,
                str(self.checkpoint.resolve()),
                device=device,
            )
            predictor = predictor_class(model)
        except Exception as exc:
            raise RuntimeError(
                f"Could not build the SAM2 image predictor. Check model config "
                f"{self.model_config!r} and checkpoint."
            ) from exc
        self.predictor, self.torch, self.device = predictor, torch, device
        self.frame_idx = None
        return predictor, torch, device

    def release(self) -> None:
        """释放预览模型，避免与视频 predictor 同时占用显存。"""
        predictor, torch = self.predictor, self.torch
        if predictor is not None:
            with suppress(Exception):
                predictor.reset_predictor()
        self.predictor = self.torch = self.device = self.frame_idx = None
        gc.collect()
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()

    def predict(
        self,
        frame: Any,
        frame_idx: int,
        annotations: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """为当前帧有正提示点的目标生成预览掩码。"""
        current = [
            point for point in annotations if int(point["frame_idx"]) == int(frame_idx)
        ]
        if not any(int(point["label"]) == 1 for point in current):
            return {}
        predictor, torch, device = self._ensure_predictor()
        if self.frame_idx != int(frame_idx):
            with torch.inference_mode(), _autocast(torch, device):
                predictor.set_image(frame)
            self.frame_idx = int(frame_idx)

        masks_by_target: dict[str, Any] = {}
        for target in TARGET_IDS:
            points = [point for point in current if point["target"] == target]
            if not any(int(point["label"]) == 1 for point in points):
                continue
            coords = self.np.asarray(
                [[point["x"], point["y"]] for point in points],
                dtype=self.np.float32,
            )
            labels = self.np.asarray(
                [int(point["label"]) for point in points], dtype=self.np.int32
            )
            with torch.inference_mode(), _autocast(torch, device):
                masks, scores, _ = predictor.predict(
                    point_coords=coords,
                    point_labels=labels,
                    multimask_output=len(points) == 1,
                )
            masks = self.np.asarray(masks)
            scores = self.np.asarray(scores).reshape(-1)
            best_idx = int(self.np.argmax(scores)) if len(scores) > 1 else 0
            masks_by_target[target] = self.np.squeeze(masks[best_idx]).astype(bool)
        return masks_by_target
