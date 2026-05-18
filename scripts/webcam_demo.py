#!/usr/bin/env python3
"""
Real-time webcam demo: MediaPipe landmarks + EmotiEffLib emotion labels.

Usage (from project root):
  python scripts/webcam_demo.py
  python scripts/webcam_demo.py --camera 1 --mesh-style tesselation
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.detector import predict_emotion
from src.face.mediapipe_face import MediaPipeFaceProcessor


def main() -> None:
    parser = argparse.ArgumentParser(description="MediaPipe + emotion webcam demo")
    parser.add_argument("--camera", type=int, default=0, help="Webcam device index")
    parser.add_argument(
        "--mesh-style",
        choices=("dots",),
        default="dots",
        help="Landmark drawing style (dots recommended for FPS)",
    )
    parser.add_argument("--max-faces", type=int, default=2)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    args = parser.parse_args()

    processor = MediaPipeFaceProcessor(max_faces=args.max_faces)
    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    if not cap.isOpened():
        print("Error: cannot open webcam. Try --camera 1")
        sys.exit(1)

    print("Press Q to quit.")
    fps_smooth = 0.0

    try:
        while True:
            t0 = time.perf_counter()
            ok, frame = cap.read()
            if not ok:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            faces = processor.process(rgb)

            processor.draw(frame, faces)

            # Emotion on largest face (same as Flask API)
            result = predict_emotion(frame)
            label = result.get("emotion", "No Face")
            conf = result.get("confidence", 0.0)
            sub = result.get("subEmotion", "")

            y = 30
            cv2.putText(
                frame,
                f"{label} ({conf:.1f}%)",
                (10, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (255, 255, 255),
                2,
            )
            if sub and sub != "None":
                y += 32
                cv2.putText(
                    frame,
                    sub,
                    (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (200, 200, 200),
                    2,
                )

            for i, face in enumerate(faces):
                x1, y1, x2, y2 = face.bbox
                cv2.putText(
                    frame,
                    f"face {i + 1}",
                    (x1, max(0, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    cfg_color(i),
                    1,
                )

            dt = time.perf_counter() - t0
            fps = 1.0 / dt if dt > 0 else 0.0
            fps_smooth = fps_smooth * 0.9 + fps * 0.1
            cv2.putText(
                frame,
                f"FPS: {fps_smooth:.1f}",
                (10, frame.shape[0] - 12),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )

            cv2.imshow("MediaPipe Emotion Demo", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        processor.release()
        cap.release()
        cv2.destroyAllWindows()


def cfg_color(index: int) -> tuple[int, int, int]:
    palette = [(0, 255, 255), (255, 0, 255), (255, 255, 0)]
    return palette[index % len(palette)]


if __name__ == "__main__":
    main()
