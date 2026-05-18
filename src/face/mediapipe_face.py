"""
MediaPipe Tasks API: Face Landmarker + optional Face Detector.

Replaces Haar Cascade / MTCNN with real-time landmark-based face crops.
Compatible with MediaPipe >= 0.10.9 (Tasks API; no mp.solutions).
"""

from __future__ import annotations

from dataclasses import dataclass, field
import threading

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision

from src.face import config as cfg
from src.face.crop import (
    bbox_from_landmarks_pixel,
    box_area,
    crop_rgb,
    expand_box,
)
from src.face.landmarks import extract_au_like_features, landmarks_to_pixel
from src.face.model_assets import get_detector_model_path, get_landmarker_model_path


@dataclass
class FaceResult:
    """One detected face in a frame."""

    bbox: tuple[int, int, int, int]
    landmarks_normalized: np.ndarray  # (N, 3) x,y,z
    landmarks_pixel: np.ndarray       # (N, 2)
    crop_rgb: np.ndarray
    confidence: float = 1.0
    au_features: dict[str, float] = field(default_factory=dict)

    @property
    def area(self) -> int:
        return box_area(self.bbox)


def _landmarks_to_arrays(face_landmarks) -> np.ndarray:
    return np.array(
        [[lm.x, lm.y, lm.z] for lm in face_landmarks],
        dtype=np.float32,
    )


def _bbox_from_detection(det, width: int, height: int) -> tuple[int, int, int, int]:
    box = det.bounding_box
    x1 = int(box.origin_x)
    y1 = int(box.origin_y)
    x2 = int(box.origin_x + box.width)
    y2 = int(box.origin_y + box.height)
    return expand_box((x1, y1, x2, y2), width, height, margin_ratio=cfg.FACE_MARGIN_RATIO)


class MediaPipeFaceProcessor:
    """Thread-safe MediaPipe Face Landmarker (default) or Face Detector."""

    def __init__(
        self,
        max_faces: int = cfg.MAX_NUM_FACES,
        min_detection_confidence: float = cfg.MIN_DETECTION_CONFIDENCE,
        min_tracking_confidence: float = cfg.MIN_TRACKING_CONFIDENCE,
        use_face_mesh: bool = True,
    ):
        self._use_face_mesh = use_face_mesh
        self._lock = threading.Lock()
        self._frame_ts_ms = 0

        if use_face_mesh:
            options = vision.FaceLandmarkerOptions(
                base_options=mp_tasks.BaseOptions(
                    model_asset_path=get_landmarker_model_path(),
                ),
                running_mode=vision.RunningMode.VIDEO,
                num_faces=max_faces,
                min_face_detection_confidence=min_detection_confidence,
                min_face_presence_confidence=min_tracking_confidence,
                min_tracking_confidence=min_tracking_confidence,
                output_face_blendshapes=False,
            )
            self._landmarker = vision.FaceLandmarker.create_from_options(options)
            self._detector = None
        else:
            options = vision.FaceDetectorOptions(
                base_options=mp_tasks.BaseOptions(
                    model_asset_path=get_detector_model_path(),
                ),
                running_mode=vision.RunningMode.VIDEO,
                min_detection_confidence=min_detection_confidence,
            )
            self._detector = vision.FaceDetector.create_from_options(options)
            self._landmarker = None

    def _next_timestamp(self) -> int:
        self._frame_ts_ms += 33
        return self._frame_ts_ms

    def _to_mp_image(self, rgb: np.ndarray) -> mp.Image:
        if not rgb.flags["C_CONTIGUOUS"]:
            rgb = np.ascontiguousarray(rgb)
        return mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    def process(self, rgb: np.ndarray) -> list[FaceResult]:
        if rgb is None or rgb.size == 0:
            return []

        h, w = rgb.shape[:2]
        mp_image = self._to_mp_image(rgb)
        ts = self._next_timestamp()

        with self._lock:
            if self._landmarker is not None:
                result = self._landmarker.detect_for_video(mp_image, ts)
            else:
                result = self._detector.detect_for_video(mp_image, ts)

        faces: list[FaceResult] = []

        if self._landmarker is not None:
            if not result.face_landmarks:
                return faces

            for face_lms in result.face_landmarks:
                lms_norm = _landmarks_to_arrays(face_lms)
                lms_px = landmarks_to_pixel(lms_norm, w, h)
                bbox = bbox_from_landmarks_pixel(
                    lms_px, w, h, margin_ratio=cfg.FACE_MARGIN_RATIO
                )
                crop = crop_rgb(rgb, bbox)
                if crop is None:
                    continue
                faces.append(
                    FaceResult(
                        bbox=bbox,
                        landmarks_normalized=lms_norm,
                        landmarks_pixel=lms_px,
                        crop_rgb=crop,
                        confidence=1.0,
                        au_features=extract_au_like_features(lms_norm),
                    )
                )
        else:
            if not result.detections:
                return faces

            for det in result.detections:
                bbox = _bbox_from_detection(det, w, h)
                crop = crop_rgb(rgb, bbox)
                if crop is None:
                    continue
                score = 1.0
                if det.categories:
                    score = float(det.categories[0].score)
                faces.append(
                    FaceResult(
                        bbox=bbox,
                        landmarks_normalized=np.zeros((0, 3), dtype=np.float32),
                        landmarks_pixel=np.zeros((0, 2), dtype=np.float32),
                        crop_rgb=crop,
                        confidence=score,
                        au_features={},
                    )
                )

        faces.sort(key=lambda f: f.area, reverse=True)
        return faces

    def process_largest(self, rgb: np.ndarray) -> FaceResult | None:
        faces = self.process(rgb)
        return faces[0] if faces else None

    def draw(
        self,
        frame_bgr: np.ndarray,
        faces: list[FaceResult],
        draw_mesh: bool = True,
        draw_bbox: bool = True,
    ) -> np.ndarray:
        h, w = frame_bgr.shape[:2]
        for face in faces:
            if draw_bbox:
                x1, y1, x2, y2 = face.bbox
                cv2.rectangle(
                    frame_bgr, (x1, y1), (x2, y2),
                    cfg.BBOX_COLOR, cfg.BBOX_THICKNESS,
                )
            if draw_mesh and len(face.landmarks_pixel) > 0:
                for x, y in face.landmarks_pixel.astype(int):
                    if 0 <= x < w and 0 <= y < h:
                        cv2.circle(
                            frame_bgr, (x, y),
                            cfg.LANDMARK_RADIUS, cfg.LANDMARK_COLOR, -1,
                        )
        return frame_bgr

    def release(self) -> None:
        with self._lock:
            if self._landmarker is not None:
                self._landmarker.close()
            if self._detector is not None:
                self._detector.close()


_processor: MediaPipeFaceProcessor | None = None
_processor_lock = threading.Lock()


def get_face_processor(**kwargs) -> MediaPipeFaceProcessor:
    global _processor
    with _processor_lock:
        if _processor is None:
            _processor = MediaPipeFaceProcessor(**kwargs)
        return _processor


def expand_box_for_emotion(
    box: tuple[int, int, int, int],
    width: int,
    height: int,
) -> tuple[int, int, int, int]:
    return expand_box(
        box, width, height,
        margin_px=cfg.FACE_MARGIN_PX,
        margin_ratio=cfg.FACE_MARGIN_RATIO,
    )
