"""MediaPipe face pipeline configuration."""

# Face Mesh
MAX_NUM_FACES = 2
MIN_DETECTION_CONFIDENCE = 0.5
MIN_TRACKING_CONFIDENCE = 0.5
REFINE_LANDMARKS = True  # iris / lip detail; set False for max FPS

# Face Detection (optional fast path when mesh is disabled)
FACE_DETECTION_MODEL = 0  # 0 = short-range (~2 m), 1 = full-range (~5 m)

# Cropping
FACE_MARGIN_RATIO = 0.12  # fraction of bbox size added on each side
FACE_MARGIN_PX = 20       # minimum pixel margin (used by emotion model path)

# Drawing
LANDMARK_RADIUS = 1
LANDMARK_COLOR = (0, 255, 128)
BBOX_COLOR = (255, 128, 0)
BBOX_THICKNESS = 2

# Emotion model input (EmotiEffLib)
EMOTION_IMG_SIZE = 224
