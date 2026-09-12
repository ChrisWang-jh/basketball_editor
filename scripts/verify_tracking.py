"""Real-weight recovery regression using basketball players from assets/demo.mp4.

The fixture is deliberately controlled: real SAM2 cutouts of two teammates are
translated over a court background, removed, and returned away from the last
position. Two copies of the same player form an intentionally ambiguous case.
This checks recovery mechanics; it is not a natural-video accuracy benchmark.
Use --natural to inspect an unchanged source video without claiming ground-truth
accuracy. Existing annotation JSON files can be supplied with --annotations.
Use --natural --no-reid to compare the same source with the SAM2-only baseline.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from collections import Counter
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--source", type=Path, default=REPO / "assets/demo.mp4")
    parser.add_argument(
        "--natural",
        action="store_true",
        help="Track the unchanged source video instead of generating the controlled fixture",
    )
    parser.add_argument(
        "--no-reid",
        action="store_true",
        help="Use the SAM2-only baseline; requires --natural",
    )
    parser.add_argument(
        "--annotations",
        type=Path,
        help="Saved UI annotations, pipeline annotations, or a JSON annotation list",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        help="Limit natural-video validation to the first N frames",
    )
    parser.add_argument("--sam2-repo", type=Path, default=REPO / "third_party/sam2")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=REPO / "checkpoints/sam2.1_hiera_base_plus.pt",
    )
    parser.add_argument("--model-config", default="configs/sam2.1/sam2.1_hiera_b+.yaml")
    parser.add_argument("--dino-repo", type=Path)
    parser.add_argument("--dino-checkpoint", type=Path)
    parser.add_argument("--image-size", type=int, default=448)
    parser.add_argument("--gap-frames", type=int, default=6)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.max_frames is not None and args.max_frames < 1:
        parser.error("--max-frames must be positive")
    if not args.natural and (
        args.annotations is not None or args.max_frames is not None or args.no_reid
    ):
        parser.error("--annotations, --max-frames, and --no-reid require --natural")
    return args


def write_video(path, images, fps):
    height, width = images[0].shape[:2]
    command = [
        "ffmpeg",
        "-y",
        "-v",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-s:v",
        f"{width}x{height}",
        "-r",
        str(fps),
        "-i",
        "pipe:0",
        "-an",
        "-c:v",
        "libx264",
        "-crf",
        "16",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(path),
    ]
    subprocess.run(command, input=b"".join(im.tobytes() for im in images), check=True)


def make_fixture(args, output, np, cv2):
    from basketeditor.sam2 import Sam2Preview, select_component

    cap = cv2.VideoCapture(str(args.source))
    ok, source = cap.read()
    cap.release()
    if not ok or source.shape[:2] != (720, 1280):
        raise ValueError(
            "This fixture requires the unmodified 1280x720 assets/demo.mp4."
        )
    source_rgb = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
    preview = Sam2Preview(
        args.checkpoint, args.model_config, args.device, args.sam2_repo, np
    )
    cutouts = []
    try:
        # Number 8 is the target; number 3 is a real teammate in the same uniform.
        for x, y in ((145, 430), (878, 448)):
            annotation = [
                {"target": "player", "frame_idx": 0, "x": x, "y": y, "label": 1}
            ]
            raw = preview.predict(source_rgb, 0, annotation, str(args.source))["player"]
            mask = select_component(raw.astype(np.float32), np, cv2, annotation) > 0
            rows, cols = np.nonzero(mask)
            if len(rows) < 500:
                raise AssertionError(
                    "The fixture player segmentation is unexpectedly empty."
                )
            x1, x2, y1, y2 = cols.min(), cols.max() + 1, rows.min(), rows.max() + 1
            crop = source[y1:y2, x1:x2]
            silhouette = mask[y1:y2, x1:x2]
            height = 226
            width = round(crop.shape[1] * height / crop.shape[0])
            if width > 210:
                raise AssertionError("SAM2 included more than one fixture player.")
            cutouts.append(
                (
                    cv2.resize(crop, (width, height)),
                    cv2.resize(
                        silhouette.astype(np.uint8),
                        (width, height),
                        interpolation=cv2.INTER_NEAREST,
                    ).astype(bool),
                )
            )
    finally:
        preview.release()

    background = cv2.resize(source[590:720, 0:1280], (640, 360))
    images, truths, phases = [], [], []

    def append(phase, people):
        frame = background.copy()
        truth = np.zeros((360, 640), dtype=bool)
        for identity, x in people:
            crop, mask = cutouts[identity]
            y, height, width = 80, *mask.shape
            destination = frame[y : y + height, x : x + width]
            destination[mask] = crop[mask]
            if identity == 0:
                truth[y : y + height, x : x + width] |= mask
        images.append(frame)
        truths.append(truth)
        phases.append(phase)

    for x in (45, 50, 55):
        append("initial", [(0, x)])
    for i in range(max(3, args.gap_frames)):
        append(
            "teammate_only" if i % 2 == 0 else "empty", [(1, 245)] if i % 2 == 0 else []
        )
    first_return = len(images)
    for x in (420, 415, 410):
        append("return_after_cut", [(1, 65), (0, x)])
    for _ in range(3):
        append("empty", [])
    for _ in range(2):
        append("ambiguous_duplicates", [(0, 45), (0, 420)])
    append("empty", [])
    second_return = len(images)
    for x in (245, 250, 255):
        append("return_unique", [(0, x)])

    frames_dir, truth_dir = output / "frames", output / "ground_truth"
    frames_dir.mkdir()
    truth_dir.mkdir()
    for index, (frame, truth) in enumerate(zip(images, truths)):
        cv2.imwrite(
            str(frames_dir / f"{index:08d}.jpg"), frame, [cv2.IMWRITE_JPEG_QUALITY, 95]
        )
        cv2.imwrite(str(truth_dir / f"{index:08d}.png"), truth.astype(np.uint8) * 255)
    cv2.imwrite(
        str(output / "fixture_contact_sheet.jpg"),
        np.concatenate(
            [
                np.concatenate([images[i] for i in row], axis=1)
                for row in (
                    (0, 3, 4),
                    (first_return, first_return + 3, first_return + 6),
                )
            ],
            axis=0,
        ),
    )
    source_path = output / "recovery_input.mp4"
    write_video(source_path, images, 10)
    distance = cv2.distanceTransform(truths[0].astype(np.uint8), cv2.DIST_L2, 5)
    y, x = np.unravel_index(int(distance.argmax()), distance.shape)
    annotations = [
        {"target": "player", "frame_idx": 0, "x": float(x), "y": float(y), "label": 1}
    ]
    return (
        frames_dir,
        truths,
        phases,
        annotations,
        (first_return, second_return),
        source_path,
    )


def analyse(args, output, info, frames, annotations, scene_cuts, np, cv2):
    """Run the same tracking/export path for both validation modes."""
    from basketeditor.models import ReIDOptions
    from basketeditor.rules import build_frame_states
    from basketeditor.sam2 import run_sam2_tracking
    from basketeditor.tracking import prepare_tracks
    from basketeditor.video import export_debug_video

    overrides = {}
    if args.dino_repo is not None:
        overrides["repo"] = str(args.dino_repo)
    if args.dino_checkpoint is not None:
        overrides["checkpoint"] = str(args.dino_checkpoint)
    options = ReIDOptions(
        enabled=not args.no_reid, image_size=args.image_size, **overrides
    )
    diagnostics = {}
    identity_file = None if args.no_reid else output / "identity_memory.npz"
    started = time.monotonic()
    last_progress = started

    def progress(value, desc):
        nonlocal last_progress
        if time.monotonic() - last_progress >= 15:
            print(f"{value:.0%} {desc}", flush=True)
            last_progress = time.monotonic()

    print(f"Tracking {info.frame_count} frames from {info.path}", flush=True)
    tracks = run_sam2_tracking(
        frames,
        info,
        annotations,
        args.checkpoint,
        args.model_config,
        args.device,
        args.sam2_repo,
        np,
        output / "masks",
        cv2,
        scene_cuts,
        progress=progress,
        reid_options=options,
        diagnostics=diagnostics,
        identity_file=identity_file,
    )
    elapsed = time.monotonic() - started
    raw_tracks = tracks
    tracks = prepare_tracks(tracks, info, scene_cuts=scene_cuts)
    states = build_frame_states(tracks, info, scene_cuts)
    export_debug_video(
        info,
        states,
        output / "debug_tracking.mp4",
        output / "masks",
        np,
        cv2,
        observed_tracks=raw_tracks,
    )
    return raw_tracks, tracks, diagnostics, identity_file, elapsed


def make_natural(args, output, np, cv2):
    from basketeditor.annotations import validate_annotations
    from basketeditor.video import read_frame_timestamps, read_video_info

    info = read_video_info(args.source, cv2)
    if args.annotations is None:
        if args.source.resolve() != (REPO / "assets/demo.mp4").resolve():
            raise ValueError(
                "Supply --annotations when using a source other than assets/demo.mp4."
            )
        points = [
            {"target": "player", "frame_idx": 0, "x": 145.0, "y": 430.0, "label": 1}
        ]
    else:
        payload = json.loads(args.annotations.read_text(encoding="utf-8"))
        points = payload if isinstance(payload, list) else payload["annotations"]
        metadata = (
            payload.get("video", payload.get("video_info", {}))
            if isinstance(payload, dict)
            else {}
        )
        if any(
            metadata.get(key, getattr(info, key)) != getattr(info, key)
            for key in ("width", "height")
        ):
            raise ValueError("The annotation dimensions do not match the source video.")
    points = validate_annotations(points, info, require_all=False)
    read_frame_timestamps(info)
    original_count = info.frame_count
    limit = min(info.frame_count, args.max_frames or info.frame_count)
    frames = output / "frames"
    frames.mkdir()
    capture = cv2.VideoCapture(info.path)
    previous = None
    cuts = []
    written = 0
    try:
        for index in range(limit):
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError(f"Source video ended before frame {index}.")
            # Match the application's cut detector while decoding only this prefix.
            small = cv2.resize(frame, (64, 36))
            histogram = cv2.calcHist(
                [cv2.cvtColor(small, cv2.COLOR_BGR2HSV)],
                [0, 1],
                None,
                [24, 16],
                [0, 180, 0, 256],
            )
            cv2.normalize(histogram, histogram)
            if (
                previous is not None
                and float(cv2.absdiff(previous[0], small).mean()) / 255 > 0.25
                and cv2.compareHist(previous[1], histogram, cv2.HISTCMP_BHATTACHARYYA)
                > 0.55
            ):
                cuts.append(index)
            previous = small, histogram
            if not cv2.imwrite(
                str(frames / f"{index:08d}.jpg"), frame, [cv2.IMWRITE_JPEG_QUALITY, 95]
            ):
                raise RuntimeError(f"Could not save validation frame {index}.")
            written += 1
    finally:
        capture.release()
    info.duration = (
        info.frame_times[written] if written < original_count else info.duration
    )
    info.frame_times = info.frame_times[:written]
    info.frame_count = written
    selected = [point for point in points if point["frame_idx"] < written]
    if not any(point["label"] == 1 for point in selected):
        raise ValueError(
            "No positive annotation occurs within the selected prefix; increase --max-frames."
        )
    return info, frames, selected, cuts, len(points) - len(selected), original_count


def contact_sheet(video, frame_count, output, np, cv2):
    capture = cv2.VideoCapture(str(video))
    thumbnails = []
    try:
        for index in np.linspace(0, frame_count - 1, min(12, frame_count), dtype=int):
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(index))
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError(f"Could not read debug contact-sheet frame {index}.")
            frame = cv2.resize(
                frame, (480, round(frame.shape[0] * 480 / frame.shape[1]))
            )
            banner = np.full((28, 480, 3), 245, np.uint8)
            cv2.putText(
                banner,
                f"Frame {index}",
                (8, 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (35, 30, 25),
                1,
                cv2.LINE_AA,
            )
            thumbnails.append(np.concatenate((banner, frame), axis=0))
    finally:
        capture.release()
    columns = min(4, len(thumbnails))
    while len(thumbnails) % columns:
        thumbnails.append(np.zeros_like(thumbnails[0]))
    sheet = np.concatenate(
        [
            np.concatenate(thumbnails[i : i + columns], axis=1)
            for i in range(0, len(thumbnails), columns)
        ],
        axis=0,
    )
    if not cv2.imwrite(str(output), sheet):
        raise RuntimeError(f"Could not write contact sheet {output}.")


def run_natural(args, output, np, cv2):
    from basketeditor.tracking import tracking_diagnostics

    info, frames, points, cuts, excluded, original_count = make_natural(
        args, output, np, cv2
    )
    (output / "annotations.json").write_text(
        json.dumps(
            {"video_info": asdict(info), "annotations": points},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    raw, tracks, diagnostics, identity_file, elapsed = analyse(
        args, output, info, frames, points, cuts, np, cv2
    )
    summary = tracking_diagnostics(tracks, info)
    annotated = sorted({point["target"] for point in points if point["label"] == 1})
    for target in annotated:
        trajectory = tracks[target]
        summary[target].update(
            {
                "accepted_observation_fraction": sum(
                    point.visible and not point.interpolated for point in trajectory
                )
                / info.frame_count,
                "source_counts": dict(Counter(point.source for point in trajectory)),
                "reason_counts": dict(Counter(point.reason for point in raw[target])),
                "reacquisition_frames": [
                    point.frame_idx
                    for point in trajectory
                    if point.source == "reacquired" and point.visible
                ],
            }
        )
    report = {
        "description": "Unmodified natural-video tracking inspection. Visibility counts are model decisions, not accuracy or ground-truth identity measurements.",
        "completed": True,
        "accuracy_verified": False,
        "source": str(args.source.resolve()),
        "annotation_source": str(args.annotations.resolve())
        if args.annotations
        else "assets/demo.mp4 frame 0 player (145, 430)",
        "video_info": asdict(info),
        "original_frame_count": original_count,
        "annotations_outside_prefix": excluded,
        "annotated_targets": annotated,
        "device": args.device,
        "tracking_backend": "sam2_only" if args.no_reid else "sam2_dinov3",
        "inference_seconds": elapsed,
        "scene_cuts": cuts,
        "targets": {target: summary[target] for target in annotated},
        "identity_diagnostics": diagnostics,
        "saved_identity": str(identity_file)
        if identity_file is not None and identity_file.is_file()
        else None,
    }
    (output / "natural_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output / "tracks.json").write_text(
        json.dumps(
            {
                "video_info": asdict(info),
                "tracks": {
                    target: [asdict(point) for point in trajectory]
                    for target, trajectory in tracks.items()
                },
                "raw_tracks": {
                    target: [asdict(point) for point in trajectory]
                    for target, trajectory in raw.items()
                },
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    contact_sheet(
        output / "debug_tracking.mp4",
        info.frame_count,
        output / "contact_sheet.jpg",
        np,
        cv2,
    )
    print(
        json.dumps(
            {
                "source": info.path,
                "frames": info.frame_count,
                "tracking_backend": report["tracking_backend"],
                "inference_seconds": elapsed,
                "accuracy_verified": False,
                "targets": {
                    target: {
                        "observed_frames": summary[target]["observed_frames"],
                        "missing_frames": summary[target]["missing_frames"],
                        "reacquisitions": len(summary[target]["reacquisition_frames"]),
                    }
                    for target in annotated
                },
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    print(f"Artifacts: {output}", flush=True)


def main():
    import cv2
    import numpy as np

    from basketeditor.models import VideoInfo
    from basketeditor.reid import load_memories

    args = arguments()
    prefix = "natural" if args.natural else "recovery"
    output = (
        args.output
        or REPO
        / "outputs/validation"
        / datetime.now().astimezone().strftime(f"{prefix}_%Y%m%d_%H%M%S")
    )
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    if args.natural:
        run_natural(args, output, np, cv2)
        return
    frames, truths, phases, annotations, returns, source = make_fixture(
        args, output, np, cv2
    )
    info = VideoInfo(str(source), 10, len(truths), 640, 360, len(truths) / 10)
    _, tracks, diagnostics, identity_file, elapsed = analyse(
        args, output, info, frames, annotations, [returns[0]], np, cv2
    )
    rows = []
    for index, (truth, phase, point) in enumerate(
        zip(truths, phases, tracks["player"])
    ):
        path = output / "masks/player" / f"{index:08d}.png"
        mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE) if path.is_file() else None
        predicted = (
            mask > 0 if mask is not None and point.visible else np.zeros_like(truth)
        )
        intersection = np.logical_and(predicted, truth).sum()
        union = np.logical_or(predicted, truth).sum()
        iou = float(intersection / union) if union else None
        rows.append(
            {
                "frame": index,
                "phase": phase,
                "visible": point.visible,
                "source": point.source,
                "mask_iou": iou,
                "point": asdict(point),
            }
        )
    false_positives = [
        r["frame"]
        for r in rows
        if r["phase"] in {"empty", "teammate_only"} and r["visible"]
    ]
    ambiguous = [
        r["frame"]
        for r in rows
        if r["phase"] == "ambiguous_duplicates" and r["visible"]
    ]
    delays = []
    for first in returns:
        valid = [
            i
            for i in range(first, first + 3)
            if rows[i]["visible"] and (rows[i]["mask_iou"] or 0) >= 0.4
        ]
        delays.append(valid[0] - first if valid else None)
    failures = []
    if false_positives:
        failures.append(f"Absent target was accepted on frames {false_positives}.")
    if ambiguous:
        failures.append(
            f"Indistinguishable duplicate was assigned an identity on frames {ambiguous}."
        )
    if delays != [0, 0]:
        failures.append(
            f"First visible frame recovery failed; measured delays: {delays}."
        )
    restored = load_memories(identity_file)
    if len(restored["player"].anchors) != 1:
        failures.append("The single manual identity anchor was not preserved on disk.")
    report = {
        "description": "Controlled synthetic translations of real same-team basketball player SAM2 cutouts; not a natural-video accuracy benchmark.",
        "source": str(args.source),
        "device": args.device,
        "elapsed_seconds": elapsed,
        "initial_annotation_only": annotations,
        "scene_cuts": [returns[0]],
        "return_frames": returns,
        "recovery_delay_frames": delays,
        "absent_false_positives": false_positives,
        "ambiguous_acceptances": ambiguous,
        "identity_diagnostics": diagnostics,
        "saved_identity": str(identity_file),
        "passed": not failures,
        "failures": failures,
        "frames": rows,
    }
    (output / "recovery_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                key: value
                for key, value in report.items()
                if key not in {"frames", "identity_diagnostics"}
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    print(f"Artifacts: {output}", flush=True)
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
