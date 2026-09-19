"""Shared SAM2 tracking + mosaic pipeline, used by the CLI and the web UI.

`run_pipeline` is a generator so callers can watch segmentation happen frame by
frame (for a live preview) instead of only getting the final video.
"""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CHECKPOINT = os.environ.get(
    "SAM2_CHECKPOINT", str(_REPO_ROOT / "checkpoints" / "sam2.1_hiera_large.pt")
)
DEFAULT_CONFIG = os.environ.get("SAM2_CONFIG", "configs/sam2.1/sam2.1_hiera_l.yaml")

MASK_COLORS = [
    (0, 0, 255), (255, 0, 0), (0, 200, 0), (0, 200, 255),
    (255, 0, 255), (0, 128, 255), (255, 255, 0), (128, 0, 255),
]


@dataclass
class Box:
    frame_idx: int
    obj_id: int
    xyxy: list[float]


def parse_bbox(text: str) -> tuple[int | None, list[float]]:
    """Parse '[frame:]x1,y1,x2,y2' into (frame_override_or_None, [x1,y1,x2,y2])."""
    frame_override = None
    if text.count(":") == 1:
        frame_str, text = text.split(":", 1)
        frame_override = int(frame_str)
    parts = [float(p) for p in text.split(",")]
    if len(parts) != 4:
        raise ValueError("bbox must be [frame:]x1,y1,x2,y2")
    return frame_override, parts


def read_frame(video_path: str, frame_idx: int) -> np.ndarray:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"could not open {video_path}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"could not read frame {frame_idx}")
    return frame


def video_meta(video_path: str) -> dict:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"could not open {video_path}")
    meta = {
        "fps": cap.get(cv2.CAP_PROP_FPS) or 30.0,
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "n_frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
    }
    cap.release()
    return meta


def pixelate(image: np.ndarray, mask: np.ndarray, block_size: int) -> np.ndarray:
    """Pixelate `image` inside `mask` (bool array, same H/W as image)."""
    h, w = image.shape[:2]
    small = cv2.resize(
        image,
        (max(1, w // block_size), max(1, h // block_size)),
        interpolation=cv2.INTER_LINEAR,
    )
    blocky = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
    out = image.copy()
    out[mask] = blocky[mask]
    return out


def dilate_mask(mask: np.ndarray, pixels: int) -> np.ndarray:
    if pixels <= 0:
        return mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (pixels * 2 + 1,) * 2)
    return cv2.dilate(mask.astype(np.uint8), kernel, iterations=1).astype(bool)


def overlay_masks(frame: np.ndarray, obj_masks: dict[int, np.ndarray]) -> np.ndarray:
    """Draw each object's mask in a distinct translucent color, for live preview."""
    overlay = frame.copy()
    for obj_id, mask in obj_masks.items():
        color = MASK_COLORS[(obj_id - 1) % len(MASK_COLORS)]
        overlay[mask] = color
    return cv2.addWeighted(overlay, 0.45, frame, 0.55, 0)


def mux_audio_and_finish(src_video, silent_video: str, out_path) -> None:
    """Encode the rendered frames to browser-playable H.264 and copy over the source audio, if any."""
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a", "-show_entries",
         "stream=index", "-of", "csv=p=0", str(src_video)],
        capture_output=True, text=True,
    )
    has_audio = probe.stdout.strip() != ""
    cmd = ["ffmpeg", "-y", "-i", silent_video]
    if has_audio:
        cmd += ["-i", str(src_video), "-map", "0:v:0", "-map", "1:a:0", "-c:a", "aac", "-shortest"]
    cmd += ["-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
            "-movflags", "+faststart", str(out_path)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    Path(silent_video).unlink(missing_ok=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg encode failed:\n{result.stderr}")


def run_pipeline(
    video_path: str,
    output_path: str,
    boxes: list[Box],
    block_size: int = 20,
    dilate: int = 6,
    checkpoint: str = DEFAULT_CHECKPOINT,
    config: str = DEFAULT_CONFIG,
    device: str = "cuda:0",
) -> Iterator[dict]:
    """Track `boxes` through the video and mosaic them.

    Yields progress events:
      {"type": "log", "text": str}
      {"type": "segment_preview", "frame_idx": int, "image": np.ndarray}  # BGR, mask overlay
      {"type": "mosaic_preview", "frame_idx": int, "image": np.ndarray}   # BGR, final pixelated frame
      {"type": "done", "output_path": str}
    """
    from sam2.build_sam import build_sam2_video_predictor

    if device.startswith("cuda") and not torch.cuda.is_available():
        yield {"type": "log", "text": "CUDA not available, falling back to cpu"}
        device = "cpu"

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"could not open {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    if not frames:
        raise RuntimeError("no frames read from video")
    yield {"type": "log", "text": f"read {len(frames)} frames ({width}x{height} @ {fps:.2f} fps)"}

    yield {"type": "log", "text": "loading SAM2 video predictor..."}
    predictor = build_sam2_video_predictor(config, checkpoint, device=device)

    masks_by_frame: dict[int, dict[int, np.ndarray]] = {}  # frame_idx -> {obj_id: mask}

    with torch.inference_mode(), torch.autocast(
        device_type="cuda" if device.startswith("cuda") else "cpu",
        dtype=torch.bfloat16,
        enabled=device.startswith("cuda"),
    ):
        state = predictor.init_state(video_path=str(video_path))
        for box in boxes:
            predictor.add_new_points_or_box(
                inference_state=state,
                frame_idx=box.frame_idx,
                obj_id=box.obj_id,
                box=np.array(box.xyxy, dtype=np.float32),
            )

        def record(out_frame_idx, out_obj_ids, out_mask_logits):
            per_obj = {}
            for i, obj_id in enumerate(out_obj_ids):
                m = (out_mask_logits[i, 0] > 0.0).cpu().numpy()
                if m.any():
                    per_obj[obj_id] = m
            masks_by_frame[out_frame_idx] = per_obj
            preview = overlay_masks(frames[out_frame_idx], per_obj)
            return {"type": "segment_preview", "frame_idx": out_frame_idx, "image": preview}

        yield {"type": "log", "text": f"tracking {len(boxes)} object(s) forward..."}
        for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(state):
            yield record(out_frame_idx, out_obj_ids, out_mask_logits)

        min_frame = min(b.frame_idx for b in boxes)
        if min_frame > 0:
            yield {"type": "log", "text": "tracking backward to frame 0..."}
            for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(
                state, start_frame_idx=min_frame, reverse=True
            ):
                yield record(out_frame_idx, out_obj_ids, out_mask_logits)

    yield {"type": "log", "text": f"tracked mask on {len(masks_by_frame)}/{len(frames)} frames, rendering mosaic..."}

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    tmp_out = str(output_path) + ".noaudio.mp4"
    writer = cv2.VideoWriter(tmp_out, fourcc, fps, (width, height))
    covered = 0
    for idx, frame in enumerate(frames):
        per_obj = masks_by_frame.get(idx, {})
        if per_obj:
            covered += 1
            union = np.zeros((height, width), dtype=bool)
            for m in per_obj.values():
                union |= m
            union = dilate_mask(union, dilate)
            frame = pixelate(frame, union, block_size)
        writer.write(frame)
        yield {"type": "mosaic_preview", "frame_idx": idx, "image": frame}
    writer.release()
    yield {"type": "log", "text": f"mosaicked {covered}/{len(frames)} frames, muxing audio..."}

    mux_audio_and_finish(video_path, tmp_out, output_path)
    yield {"type": "done", "output_path": str(output_path)}
