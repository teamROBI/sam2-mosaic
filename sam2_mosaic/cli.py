"""Track one or more bounding boxes through a video with SAM2 and mosaic them.

Usage (from the repository root):
    # 1. Inspect a frame to pick pixel coordinates for each box.
    python -m sam2_mosaic.cli dump-frame IN.mp4 --frame 0 --out first_frame.png

    # 2. Track + mosaic. Repeat --bbox for each object (same reference frame by
    #    default, or prefix a box with "N:" to give it its own frame, e.g. 10:x1,y1,x2,y2).
    python -m sam2_mosaic.cli run IN.mp4 OUT.mp4 \
        --bbox x1,y1,x2,y2 --bbox x1,y1,x2,y2 [--frame 0]

Or draw boxes in the browser and watch segmentation live:
    python -m sam2_mosaic.webapp
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from .core import DEFAULT_CHECKPOINT, DEFAULT_CONFIG, Box, parse_bbox, run_pipeline


def cmd_dump_frame(args: argparse.Namespace) -> None:
    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise RuntimeError(f"could not open {args.video}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, args.frame)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"could not read frame {args.frame}")
    cv2.imwrite(str(args.out), frame)
    h, w = frame.shape[:2]
    print(f"wrote {args.out} ({w}x{h}); pick x1,y1,x2,y2 for --bbox")


def cmd_run(args: argparse.Namespace) -> None:
    boxes = []
    for obj_id, raw in enumerate(args.bbox, start=1):
        frame_override, xyxy = parse_bbox(raw)
        frame_idx = args.frame if frame_override is None else frame_override
        boxes.append(Box(frame_idx=frame_idx, obj_id=obj_id, xyxy=xyxy))

    for event in run_pipeline(
        video_path=str(args.video),
        output_path=str(args.output),
        boxes=boxes,
        block_size=args.block_size,
        dilate=args.dilate,
        checkpoint=args.checkpoint,
        config=args.config,
        device=args.device,
    ):
        if event["type"] == "log":
            print(event["text"])
        elif event["type"] == "done":
            print(f"wrote {event['output_path']}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_dump = sub.add_parser("dump-frame", help="save one frame to inspect for bbox coordinates")
    p_dump.add_argument("video", type=Path)
    p_dump.add_argument("--frame", type=int, default=0)
    p_dump.add_argument("--out", type=Path, default=Path("frame.png"))
    p_dump.set_defaults(func=cmd_dump_frame)

    p_run = sub.add_parser("run", help="track the bbox(es) and mosaic them through the video")
    p_run.add_argument("video", type=Path)
    p_run.add_argument("output", type=Path)
    p_run.add_argument(
        "--bbox", required=True, action="append",
        help="[frame:]x1,y1,x2,y2 in pixel coords; repeat for multiple objects. "
             "Omit 'frame:' to use --frame (default: same reference frame for all boxes)",
    )
    p_run.add_argument("--frame", type=int, default=0, help="default frame index for boxes without their own frame:")
    p_run.add_argument("--block-size", type=int, default=20, help="mosaic block size in pixels")
    p_run.add_argument("--dilate", type=int, default=6, help="grow the tracked mask by N px before mosaicking")
    p_run.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    p_run.add_argument("--config", default=DEFAULT_CONFIG)
    p_run.add_argument("--device", default="cuda:0")
    p_run.set_defaults(func=cmd_run)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
