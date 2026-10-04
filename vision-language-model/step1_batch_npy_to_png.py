import argparse
from pathlib import Path

import numpy as np
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument("input_dir", type=Path)
parser.add_argument("output_dir", type=Path)
parser.add_argument("--step", type=int, default=10)
parser.add_argument("--channel-axis", type=int, default=-1)
parser.add_argument("--vmin", type=float, default=0.0)
parser.add_argument("--vmax", type=float, default=1.0)
args = parser.parse_args()

if not args.input_dir.is_dir():
    parser.error("Input directory does not exist.")
if args.step < 1:
    parser.error("--step must be at least 1.")
if args.vmax <= args.vmin:
    parser.error("--vmax must be greater than --vmin.")

# Numeric filenames represent simulation times.
files = list(args.input_dir.glob("*.npy"))
if not files:
    raise SystemExit("No .npy files found.")

try:
    files.sort(key=lambda path: float(path.stem))
except ValueError:
    raise SystemExit("Expected numeric filenames, such as 73.004154.npy.")

args.output_dir.mkdir(parents=True, exist_ok=True)
success = 0
selected = list(range(0, len(files), args.step))

for index in selected:
    source = files[index]

    try:
        data = np.load(source, allow_pickle=False)

        if data.ndim == 2:
            channel = data
        elif data.ndim == 3:
            channel = np.take(data, 0, axis=args.channel_axis)
        else:
            raise ValueError(f"Unsupported shape: {data.shape}")

        channel = channel.astype(np.float64)
        if not np.isfinite(channel).all():
            raise ValueError("Channel contains NaN or infinite values.")

        scaled = (channel - args.vmin) / (args.vmax - args.vmin)
        pixels = np.rint(np.clip(scaled, 0, 1) * 255).astype(np.uint8)

        # Preserve both the original frame number and source filename.
        target = args.output_dir / (
            f"frame_{index + 1:06d}_{source.stem}.png"
        )
        Image.fromarray(pixels).save(target)
        success += 1
        print(f"Frame {index + 1}: {source.name} -> {target.name}")

    except Exception as exc:
        print(f"FAILED: {source.name}: {exc}")

print(f"Done: {success}/{len(selected)} selected frames converted.")
if success != len(selected):
    raise SystemExit(1)