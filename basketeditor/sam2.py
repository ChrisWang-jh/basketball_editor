"""SAM2 模型加载、静态帧预览和整段视频追踪。"""

from __future__ import annotations

import gc
import importlib
import sys
import tempfile
from collections import OrderedDict
from contextlib import nullcontext, suppress
from pathlib import Path
from typing import Any

from .models import TARGET_IDS, TrackPoint, VideoInfo


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
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        return torch.autocast("cuda", dtype=dtype)
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


def _numpy_mask(mask: Any, np: Any) -> Any:
    if hasattr(mask, "detach"):
        mask = mask.detach().float().cpu().numpy()
    return np.squeeze(np.asarray(mask))


def select_component(
    raw_mask: Any,
    np: Any,
    cv2: Any,
    points: list[dict],
    previous: TrackPoint | None = None,
) -> Any:
    """Keep a single coherent component, preferentially the one the user clicked."""
    array = _numpy_mask(raw_mask, np)
    if array.ndim != 2:
        raise RuntimeError(f"Invalid SAM2 mask dimensions: {array.shape}")
    if cv2 is None:
        return array
    count, labels, stats, centers = cv2.connectedComponentsWithStats(
        (array > 0).astype(np.uint8), 8
    )
    if count <= 1:
        return array
    positive = [p for p in points if p["label"] == 1 and "box" not in p]
    votes = {}
    for point in positive:
        x, y = round(point["x"]), round(point["y"])
        component = int(
            labels[min(y, labels.shape[0] - 1), min(x, labels.shape[1] - 1)]
        )
        if component:
            votes[component] = votes.get(component, 0) + 1
    if votes:
        chosen = max(votes, key=lambda c: (votes[c], stats[c, cv2.CC_STAT_AREA]))
    else:
        boxes = [p["box"] for p in points if "box" in p]
        reference = (
            ((boxes[-1][0] + boxes[-1][2]) / 2, (boxes[-1][1] + boxes[-1][3]) / 2)
            if boxes
            else (previous.center if previous and previous.visible else None)
        )

        def rank(component: int) -> float:
            area = float(stats[component, cv2.CC_STAT_AREA])
            if reference is None:
                return area
            distance = float(np.linalg.norm(centers[component] - np.asarray(reference)))
            return area**0.5 / (1 + distance)

        chosen = max(range(1, count), key=rank)
    return np.where(labels == chosen, array, -32.0)


class LazyVideoFrames:
    """SAM2-compatible indexed frames with a bounded CPU cache (no full-video tensor)."""

    def __init__(self, directory: Path, count: int, size: int, torch: Any, np: Any):
        self.directory, self.count, self.size = directory, count, size
        self.torch, self.np = torch, np
        self.cache: OrderedDict = OrderedDict()

    def __len__(self) -> int:
        return self.count

    def __getitem__(self, index: int) -> Any:
        if not 0 <= index < self.count:
            raise IndexError(index)
        if index not in self.cache:
            from PIL import Image

            with Image.open(self.directory / f"{index:08d}.jpg") as image:
                array = (
                    self.np.array(
                        image.convert("RGB").resize((self.size, self.size)),
                        dtype=self.np.float32,
                    )
                    / 255.0
                )
            tensor = self.torch.from_numpy(array).permute(2, 0, 1)
            mean = self.torch.tensor((0.485, 0.456, 0.406))[:, None, None]
            std = self.torch.tensor((0.229, 0.224, 0.225))[:, None, None]
            self.cache[index] = (tensor - mean) / std
            if len(self.cache) > 4:
                self.cache.popitem(last=False)
        self.cache.move_to_end(index)
        return self.cache[index]


def init_lazy_state(
    predictor: Any, frames_dir: Path, info: VideoInfo, torch: Any, np: Any
) -> Any:
    # Initialize the official state on one frame, then provide its public indexed
    # image store. Model/state initialization stays owned by the installed SAM2.
    with tempfile.TemporaryDirectory(
        prefix="sam2_init_", dir=frames_dir.parent
    ) as temp:
        (Path(temp) / "00000000.jpg").symlink_to(
            (frames_dir / "00000000.jpg").resolve()
        )
        state = predictor.init_state(
            video_path=temp, offload_video_to_cpu=True, offload_state_to_cpu=True
        )
    state["images"] = LazyVideoFrames(
        frames_dir, info.frame_count, predictor.image_size, torch, np
    )
    state["num_frames"] = info.frame_count
    return state


def _track_prompted(
    predictor, state, info, groups, masks_dir, np, cv2, scene_cuts, progress
):
    """Small SAM2-only baseline, with separate memory for each object/direction."""
    tracks = {
        target: [TrackPoint(i) for i in range(info.frame_count)]
        for target in TARGET_IDS
    }
    boundaries = sorted({0, info.frame_count, *(scene_cuts or [])})
    for target, object_id in TARGET_IDS.items():
        ranks = {}
        if masks_dir is not None:
            (masks_dir / target).mkdir(parents=True, exist_ok=True)
        for begin, end in zip(boundaries, boundaries[1:]):
            prompts = {
                i: points
                for (name, i), points in groups.items()
                if name == target and begin <= i < end
            }
            anchors = sorted(
                i for i, pts in prompts.items() if any(p["label"] == 1 for p in pts)
            )
            if not anchors:
                continue
            for reverse in (False, True):
                predictor.reset_state(state)
                previous = None

                def consume(output):
                    nonlocal previous
                    index, ids, masks = output
                    index = int(index)
                    for j, oid in enumerate(ids):
                        if int(oid) != object_id:
                            continue
                        mask = select_component(
                            masks[j], np, cv2, prompts.get(index, []), previous
                        )
                        point = mask_to_track_point(index, mask, np)
                        point.anchored = index in anchors
                        point.source = "anchor" if point.anchored else "sam2"
                        rank = (
                            point.anchored,
                            point.visible,
                            -min(abs(index - a) for a in anchors),
                            point.confidence,
                        )
                        if index not in ranks or rank > ranks[index]:
                            ranks[index] = rank
                            tracks[target][index] = point
                            if masks_dir is not None and point.visible:
                                path = masks_dir / target / f"{index:08d}.png"
                                if not cv2.imwrite(
                                    str(path), (mask > 0).astype(np.uint8) * 255
                                ):
                                    raise RuntimeError(
                                        f"Could not write tracking mask: {path}"
                                    )
                        if point.visible:
                            previous = point

                for index, points in sorted(prompts.items()):
                    coords = np.asarray(
                        [[p["x"], p["y"]] for p in points if "box" not in p], np.float32
                    ).reshape(-1, 2)
                    labels = np.asarray(
                        [p["label"] for p in points if "box" not in p], np.int32
                    )
                    kwargs = dict(
                        inference_state=state,
                        frame_idx=index,
                        obj_id=object_id,
                        points=coords if len(coords) else None,
                        labels=labels if len(labels) else None,
                    )
                    boxes = [p["box"] for p in points if "box" in p]
                    if boxes:
                        kwargs["box"] = np.asarray(boxes[-1], np.float32)
                    consume(predictor.add_new_points_or_box(**kwargs))
                previous = None
                start = anchors[-1] if reverse else anchors[0]
                length = start - begin if reverse else end - start - 1
                for output in predictor.propagate_in_video(
                    state,
                    start_frame_idx=start,
                    max_frame_num_to_track=length,
                    reverse=reverse,
                ):
                    consume(output)
                    if progress:
                        progress(
                            0.2 + 0.53 * int(output[0]) / max(1, info.frame_count),
                            desc=f"SAM2 {target}",
                        )
    return tracks


def run_sam2_tracking(
    frames_dir: Path,
    info: VideoInfo,
    annotations: list[dict[str, Any]],
    checkpoint: Path,
    model_config: str,
    device_setting: str,
    sam2_repo: Path | None,
    np: Any,
    masks_dir: Path | None = None,
    cv2: Any = None,
    scene_cuts: list[int] | None = None,
    progress: Any = None,
    reid_options: Any = None,
    diagnostics: dict | None = None,
    identity_file: Path | None = None,
) -> dict[str, list[TrackPoint]]:
    """Track with permanent manual identity templates and verified reacquisition.

    All annotated views initialize a fixed identity gallery, then a single scan
    searches even before the first annotation and after camera cuts. SAM2-only
    propagation remains an explicit baseline, never a silent model fallback.
    """
    from .annotations import validate_annotations
    from .models import ReIDOptions

    annotations = validate_annotations(annotations, info, require_all=False)
    options = reid_options or ReIDOptions()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"SAM2 checkpoint not found: {checkpoint}")
    if options.enabled:
        if not Path(options.repo).is_dir() or not Path(options.checkpoint).is_file():
            raise FileNotFoundError(
                "本地 DINOv3 源码或权重不存在，请检查 --dino-repo / --dino-checkpoint。"
            )
        if cv2 is None:
            cv2 = importlib.import_module("cv2")
    if masks_dir is not None and cv2 is None:
        raise ValueError("cv2 is required when masks_dir is provided")
    torch = _load_torch(sam2_repo)
    builder = importlib.import_module("sam2.build_sam").build_sam2_video_predictor
    device = select_device(torch, device_setting)
    predictor = image_predictor = inference_state = None
    groups = {}
    for point in annotations:
        groups.setdefault((point["target"], point["frame_idx"]), []).append(point)
    try:
        predictor = builder(model_config, str(checkpoint.resolve()), device=device)
        with torch.inference_mode(), _autocast(torch, device):
            inference_state = init_lazy_state(predictor, frames_dir, info, torch, np)
            if options.enabled:
                from .tracker import track_identities

                image_class = importlib.import_module(
                    "sam2.sam2_image_predictor"
                ).SAM2ImagePredictor
                image_predictor = image_class(predictor)
                return track_identities(
                    predictor,
                    image_predictor,
                    inference_state,
                    frames_dir,
                    info,
                    groups,
                    options,
                    masks_dir,
                    scene_cuts,
                    progress,
                    diagnostics if diagnostics is not None else {},
                    identity_file,
                )
            return _track_prompted(
                predictor,
                inference_state,
                info,
                groups,
                masks_dir,
                np,
                cv2,
                scene_cuts,
                progress,
            )
    finally:
        if predictor is not None and inference_state is not None:
            with suppress(Exception):
                predictor.reset_state(inference_state)
        del image_predictor, inference_state, predictor
        gc.collect()
        if device.startswith("cuda"):
            torch.cuda.empty_cache()


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
        self.frame_key: tuple[str, int] | None = None

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
        self.frame_key = None
        return predictor, torch, device

    def reset_frame(self) -> None:
        """让下一次预览重新向 predictor 注入图像。"""
        self.frame_key = None

    def release(self) -> None:
        """释放预览模型，避免与视频 predictor 同时占用显存。"""
        predictor, torch = self.predictor, self.torch
        if predictor is not None:
            with suppress(Exception):
                predictor.reset_predictor()
        self.predictor = self.torch = self.device = self.frame_key = None
        gc.collect()
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()

    def predict(
        self,
        frame: Any,
        frame_idx: int,
        annotations: list[dict[str, Any]],
        source_key: str = "",
    ) -> dict[str, Any]:
        """为当前帧有正提示点的目标生成预览掩码。"""
        current = [
            point for point in annotations if int(point["frame_idx"]) == int(frame_idx)
        ]
        if not any(int(point["label"]) == 1 for point in current):
            return {}
        predictor, torch, device = self._ensure_predictor()
        frame_key = (source_key, int(frame_idx))
        if self.frame_key != frame_key:
            with torch.inference_mode(), _autocast(torch, device):
                predictor.set_image(frame)
            self.frame_key = frame_key

        masks_by_target: dict[str, Any] = {}
        for target in TARGET_IDS:
            points = [point for point in current if point["target"] == target]
            if not any(int(point["label"]) == 1 for point in points):
                continue
            coords = self.np.asarray(
                [[point["x"], point["y"]] for point in points if "box" not in point],
                dtype=self.np.float32,
            )
            coords = coords.reshape(-1, 2)
            boxes = [point["box"] for point in points if "box" in point]
            labels = self.np.asarray(
                [int(point["label"]) for point in points if "box" not in point],
                dtype=self.np.int32,
            )
            with torch.inference_mode(), _autocast(torch, device):
                masks, scores, _ = predictor.predict(
                    point_coords=coords if len(coords) else None,
                    point_labels=labels if len(labels) else None,
                    box=self.np.asarray(boxes[-1], dtype=self.np.float32)
                    if boxes
                    else None,
                    multimask_output=len(points) == 1 and not boxes,
                )
            masks = self.np.asarray(masks)
            scores = self.np.asarray(scores).reshape(-1)
            best_idx = int(self.np.argmax(scores)) if len(scores) > 1 else 0
            masks_by_target[target] = self.np.squeeze(masks[best_idx]).astype(bool)
        return masks_by_target
