"""Download MediaPipe Tasks model files on first use."""

from __future__ import annotations

import os
import urllib.request

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")

LANDMARKER_NAME = "face_landmarker.task"
DETECTOR_NAME = "blaze_face_short_range.tflite"

LANDMARKER_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)
DETECTOR_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_detector/"
    "blaze_face_short_range/float16/1/blaze_face_short_range.tflite"
)


def _download(url: str, dest: str) -> str:
    if os.path.isfile(dest):
        return dest
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    print(f"Downloading MediaPipe model to {dest} ...")
    urllib.request.urlretrieve(url, dest)
    return dest


def get_landmarker_model_path() -> str:
    return _download(LANDMARKER_URL, os.path.join(MODEL_DIR, LANDMARKER_NAME))


def get_detector_model_path() -> str:
    return _download(DETECTOR_URL, os.path.join(MODEL_DIR, DETECTOR_NAME))
