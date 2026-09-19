# sam2_mosaic

Mosaic (pixelate) objects in a video. Give it one or more bounding boxes on a
single frame, and [SAM 2](https://github.com/facebookresearch/sam2) tracks and
segments each object through the whole video; the tracked regions are then
pixelated in the output.

It comes with two front ends on top of the same pipeline:

- **Web UI** – draw boxes with the mouse and watch the segmentation happen live.
- **CLI** – scriptable, boxes given as pixel coordinates.

## Features

- Multiple objects at once (each is tracked as its own SAM 2 object; masks are unioned for the mosaic).
- Boxes can be drawn on different reference frames; tracking runs both forward and backward from the earliest one.
- Live preview: a per-object colored mask overlay streams to the browser (MJPEG) while SAM 2 propagates, followed by the rendered mosaic frames.
- The original audio track is preserved; output is H.264 (`yuv420p`, `faststart`) so it plays in browsers.
- Tunable mosaic block size and mask dilation (to avoid leaking edges).

## Requirements

- Linux, Python >= 3.10, an NVIDIA GPU (CPU works but is slow)
- `ffmpeg` and `ffprobe` on `PATH`
- A SAM 2.1 checkpoint (see below)

## Installation

```bash
python3 -m venv .venv && source .venv/bin/activate

# Install torch first, matching your CUDA version
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

pip install -r requirements.txt
```

Download a checkpoint into `checkpoints/`:

```bash
mkdir -p checkpoints
curl -L -o checkpoints/sam2.1_hiera_large.pt \
  https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt
```

Other sizes (`tiny`, `small`, `base_plus`) are smaller and faster; use them with
`--checkpoint` and `--config` (e.g. `configs/sam2.1/sam2.1_hiera_t.yaml`).
The defaults can also be set with the `SAM2_CHECKPOINT` and `SAM2_CONFIG`
environment variables.

> On startup SAM 2 may print `cannot import name '_C' from 'sam2'`. This only
> disables an optional CUDA hole-filling post-processing step and is harmless.

## Web UI

```bash
python -m sam2_mosaic.webapp            # http://0.0.0.0:8765
python -m sam2_mosaic.webapp --port 9000
```

Open the page (on a remote machine, use SSH/VS Code port forwarding to the port).
Paths you type in are resolved on the **server**.

1. Enter the video path and a reference frame number, then click *Load frame*.
2. Drag on the canvas to draw one box per object. To add an object that only
   appears later, change the reference frame number, load that frame, and draw
   its box there; each box remembers the frame it was drawn on.
3. Set the output path (optional), block size, mask dilation and GPU, then run.
4. The live view shows each tracked mask in its own color as SAM 2 propagates,
   then the mosaicked frames as they are rendered. When done, the result plays
   in place and can be downloaded.

## CLI

Run from the repository root.

```bash
# Save a frame to look up pixel coordinates
python -m sam2_mosaic.cli dump-frame input.mp4 --frame 0 --out frame.png

# Track + mosaic. Repeat --bbox once per object (x1,y1,x2,y2 in pixels)
python -m sam2_mosaic.cli run input.mp4 output.mp4 \
    --bbox 100,80,220,200 \
    --bbox 400,120,520,260 \
    --frame 0

# A box can carry its own reference frame with an "N:" prefix
python -m sam2_mosaic.cli run input.mp4 output.mp4 \
    --bbox 0:100,80,220,200 --bbox 45:400,120,520,260
```

| Option | Default | Description |
| --- | --- | --- |
| `--bbox [N:]x1,y1,x2,y2` | required | Box for one object; repeat for more. `N:` overrides `--frame`. |
| `--frame` | `0` | Reference frame for boxes without an `N:` prefix. |
| `--block-size` | `20` | Mosaic block size in pixels; larger hides more. |
| `--dilate` | `6` | Grow each tracked mask by this many pixels before pixelating. |
| `--checkpoint` | `checkpoints/sam2.1_hiera_large.pt` | SAM 2 weights. |
| `--config` | `configs/sam2.1/sam2.1_hiera_l.yaml` | SAM 2 config matching the checkpoint. |
| `--device` | `cuda:0` | Torch device; falls back to CPU if CUDA is unavailable. |

## Layout

```
sam2_mosaic/
  core.py     tracking + mosaic pipeline (a generator yielding progress and preview frames)
  cli.py      command-line front end
  webapp.py   Flask app: frame picker, job runner, MJPEG live stream
checkpoints/  SAM 2 weights (not tracked by git)
```

The pipeline is also usable as a library:

```python
from sam2_mosaic.core import Box, run_pipeline

boxes = [Box(frame_idx=0, obj_id=1, xyxy=[100, 80, 220, 200])]
for event in run_pipeline("input.mp4", "output.mp4", boxes, block_size=20):
    if event["type"] == "log":
        print(event["text"])
```

## Limitations

- All frames are decoded into memory (and SAM 2 loads its own resized copy), so
  very long or high-resolution videos need a lot of RAM/VRAM; trim or downscale first.
- Tracking quality is SAM 2's: objects that leave the frame, are heavily occluded
  or change drastically may be lost. Add a box on a later frame to recover them.
- The web app keeps job state in memory, has no authentication, and reads/writes
  arbitrary server paths. Run it only on a trusted network or behind port forwarding.

## Acknowledgements

Segmentation and tracking are done by
[SAM 2: Segment Anything in Images and Videos](https://github.com/facebookresearch/sam2)
(Meta AI, Apache-2.0). Model weights are downloaded separately from Meta and are
subject to their own license; see the SAM 2 repository.

```bibtex
@article{ravi2024sam2,
  title   = {SAM 2: Segment Anything in Images and Videos},
  author  = {Ravi, Nikhila and Gabeur, Valentin and Hu, Yuan-Ting and Hu, Ronghang and Ryali, Chaitanya and Ma, Tengyu and Khedr, Haitham and R{\"a}dle, Roman and Rolland, Chloe and Gustafson, Laura and Mintun, Eric and Pan, Junting and Alwala, Kalyan Vasudev and Carion, Nicolas and Wu, Chao-Yuan and Girshick, Ross and Doll{\'a}r, Piotr and Feichtenhofer, Christoph},
  journal = {arXiv preprint arXiv:2408.00714},
  year    = {2024}
}
```

## License

Licensed under the [Apache License 2.0](LICENSE).
