#!/usr/bin/env python3
"""
Re-crop dataset images (or video frames) using MediaPipe instead of Haar Cascade.

FER2013 PNGs are already 48x48 face chips — use --source on raw photos/videos only.
For FER-style folders (class subdirs), this re-detects faces and saves aligned crops.

Usage:
  python scripts/preprocess_dataset_mediapipe.py --source data/raw --output data/mediapipe_crops
  python scripts/preprocess_dataset_mediapipe.py --source videos/ --output crops/ --size 224
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.face.mediapipe_face import MediaPipeFaceProcessor

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


def iter_inputs(source: Path, recursive: bool):
    if source.is_file():
        yield source
        return
    pattern = "**/*" if recursive else "*"
    for p in source.glob(pattern):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS | VIDEO_EXTS:
            yield p


def save_crop(crop_bgr, out_path: Path, size: int) -> bool:
    if crop_bgr is None or crop_bgr.size == 0:
        return False
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if size > 0:
        crop_bgr = cv2.resize(crop_bgr, (size, size), interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(out_path), crop_bgr)
    return True


def process_image(path: Path, processor: MediaPipeFaceProcessor, out_dir: Path, size: int) -> int:
    bgr = cv2.imread(str(path))
    if bgr is None:
        return 0
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    faces = processor.process(rgb)
    saved = 0
    stem = path.stem
    for i, face in enumerate(faces):
        suffix = f"_{i}" if len(faces) > 1 else ""
        out_path = out_dir / f"{stem}{suffix}.png"
        crop_bgr = cv2.cvtColor(face.crop_rgb, cv2.COLOR_RGB2BGR)
        if save_crop(crop_bgr, out_path, size):
            saved += 1
    return saved


def process_video(path: Path, processor: MediaPipeFaceProcessor, out_dir: Path, size: int, every_n: int) -> int:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return 0
    saved = 0
    frame_idx = 0
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        frame_idx += 1
        if frame_idx % every_n != 0:
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        faces = processor.process(rgb)
        for i, face in enumerate(faces):
            out_path = out_dir / path.stem / f"frame_{frame_idx:06d}_face{i}.png"
            crop_bgr = cv2.cvtColor(face.crop_rgb, cv2.COLOR_RGB2BGR)
            if save_crop(crop_bgr, out_path, size):
                saved += 1
    cap.release()
    return saved


def main() -> None:
    parser = argparse.ArgumentParser(description="MediaPipe dataset face cropping")
    parser.add_argument("--source", type=Path, required=True, help="Image, video, or folder")
    parser.add_argument("--output", type=Path, required=True, help="Output directory")
    parser.add_argument("--size", type=int, default=224, help="Output square size (0 = native)")
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--video-every", type=int, default=5, help="Sample every N frames")
    parser.add_argument("--max-faces", type=int, default=1)
    args = parser.parse_args()

    processor = MediaPipeFaceProcessor(max_faces=args.max_faces)
    paths = list(iter_inputs(args.source, args.recursive))
    total_saved = 0

    for path in tqdm(paths, desc="Processing"):
        rel = path.relative_to(args.source) if args.source.is_dir() else path.name
        out_dir = args.output / Path(rel).parent

        if path.suffix.lower() in VIDEO_EXTS:
            total_saved += process_video(
                path, processor, out_dir, args.size, args.video_every
            )
        else:
            total_saved += process_image(path, processor, out_dir, args.size)

    processor.release()
    print(f"Done. Saved {total_saved} face crops to {args.output}")


if __name__ == "__main__":
    main()
