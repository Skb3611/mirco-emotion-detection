import os
import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from facenet_pytorch import MTCNN


# ───────────────────────────────────────────────────────────────
# Device
# ───────────────────────────────────────────────────────────────

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

torch.set_grad_enabled(False)

if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True


# ───────────────────────────────────────────────────────────────
# Paths
# ───────────────────────────────────────────────────────────────

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "enet_b0_8_best_vgaf.pt",
)

MODEL_URL = (
    "https://github.com/sb-ai-lab/EmotiEffLib/raw/main/"
    "models/affectnet_emotions/enet_b0_8_best_vgaf.pt"
)

IMG_SIZE = 224
FACE_MARGIN = 20


def _ensure_model() -> str:
    if os.path.isfile(MODEL_PATH):
        return MODEL_PATH
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    import urllib.request

    print(f"Downloading model to {MODEL_PATH} ...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    return MODEL_PATH


# ───────────────────────────────────────────────────────────────
# Model
# ───────────────────────────────────────────────────────────────

model = torch.load(
    _ensure_model(),
    map_location=DEVICE,
    weights_only=False,
)

model.to(DEVICE)
model.eval()


# ───────────────────────────────────────────────────────────────
# Face detector (boxes only; preprocessing matches EmotiEffLib)
# ───────────────────────────────────────────────────────────────

mtcnn = MTCNN(
    keep_all=False,
    device=DEVICE,
)


# ───────────────────────────────────────────────────────────────
# Transform (ImageNet — same as EmotiEffLib)
# ───────────────────────────────────────────────────────────────

IMAGE_TRANSFORM = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


def _expand_box(box, width: int, height: int, margin: int) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    x1 = max(0, int(x1) - margin)
    y1 = max(0, int(y1) - margin)
    x2 = min(width, int(x2) + margin)
    y2 = min(height, int(y2) + margin)
    return x1, y1, x2, y2


def _face_tensor_from_rgb(rgb: np.ndarray) -> torch.Tensor | None:
    boxes, _ = mtcnn.detect(rgb)
    if boxes is None or len(boxes) == 0:
        return None

    h, w = rgb.shape[:2]
    x1, y1, x2, y2 = _expand_box(boxes[0], w, h, FACE_MARGIN)
    if x2 <= x1 or y2 <= y1:
        return None

    face = rgb[y1:y2, x1:x2]
    return IMAGE_TRANSFORM(Image.fromarray(face))


def _predict_probs(rgb: np.ndarray) -> np.ndarray | None:
    face_tensor = _face_tensor_from_rgb(rgb)
    if face_tensor is None:
        return None

    input_tensor = face_tensor.unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.softmax(outputs, dim=1)[0]

    return probs.cpu().numpy()


# ───────────────────────────────────────────────────────────────
# AffectNet labels (EmotiEffLib 8-class index order)
# ───────────────────────────────────────────────────────────────

AFFECTNET_LABELS = [
    "Anger",
    "Contempt",
    "Disgust",
    "Fear",
    "Happy",
    "Neutral",
    "Sad",
    "Surprise",
]


# ───────────────────────────────────────────────────────────────
# AffectNet → Wheel Base
# ───────────────────────────────────────────────────────────────

AFFECTNET_TO_WHEEL_BASE = {
    "Anger":    "Angry",
    "Contempt": "Confident",
    "Disgust":  "Embarrassed",
    "Fear":     "Scared",
    "Happy":    "Happy",
    "Neutral":  "Neutral",
    "Sad":      "Sad",
    "Surprise": "Happy",
}


# ───────────────────────────────────────────────────────────────
# AffectNet → Category
# ───────────────────────────────────────────────────────────────

AFFECTNET_TO_CATEGORY = {
    "Anger":    "Uncomfortable",
    "Contempt": "Comfortable",
    "Disgust":  "Uncomfortable",
    "Fear":     "Uncomfortable",
    "Happy":    "Comfortable",
    "Neutral":  "Comfortable",
    "Sad":      "Uncomfortable",
    "Surprise": "Comfortable",
}


# ───────────────────────────────────────────────────────────────
# Sub-emotions
# ───────────────────────────────────────────────────────────────

WHEEL_SUB_MAP = {
    "Sad":         [(75.0, "Hurt"),        (45.0, "Disappointed"), (0.0, "Lonely")],
    "Scared":      [(75.0, "Overwhelmed"), (45.0, "Powerless"),    (0.0, "Anxious")],
    "Angry":       [(75.0, "Annoyed"),     (45.0, "Jealous"),      (0.0, "Bored")],
    "Embarrassed": [(75.0, "Ashamed"),     (45.0, "Excluded"),     (0.0, "Guilty")],
    "Happy":       [(75.0, "Excited"),     (45.0, "Grateful"),     (0.0, "Caring")],
    "Neutral":     [(75.0, "Creative"),    (45.0, "Calm"),         (0.0, "Relaxed")],
    "Loved":       [(75.0, "Respected"),   (45.0, "Valued"),       (0.0, "Accepted")],
    "Confident":   [(75.0, "Powerful"),    (45.0, "Brave"),        (0.0, "Hopeful")],
}


# ───────────────────────────────────────────────────────────────
# Emotion Wheel Order
# ───────────────────────────────────────────────────────────────

WHEEL_ORDER = [
    ("Uncomfortable", "Sad"),
    ("Uncomfortable", "Scared"),
    ("Uncomfortable", "Angry"),
    ("Uncomfortable", "Embarrassed"),
    ("Comfortable",   "Happy"),
    ("Comfortable",   "Loved"),
    ("Comfortable",   "Confident"),
    ("Comfortable",   "Neutral"),
]


# ───────────────────────────────────────────────────────────────
# Utility
# ───────────────────────────────────────────────────────────────

def get_active_sub(base: str, confidence: float) -> str:
    for min_conf, sub_label in WHEEL_SUB_MAP.get(base, []):
        if confidence >= min_conf:
            return sub_label
    return "None"


# ───────────────────────────────────────────────────────────────
# Build Wheel List
# ───────────────────────────────────────────────────────────────

def get_wheel_base_list(preds: np.ndarray) -> list:

    base_conf = {base: 0.0 for _, base in WHEEL_ORDER}

    for i, affect_label in enumerate(AFFECTNET_LABELS):

        conf = float(preds[i]) * 100

        wheel_base = AFFECTNET_TO_WHEEL_BASE[affect_label]

        if wheel_base in base_conf:
            base_conf[wheel_base] += conf

    result = []

    for category, base in WHEEL_ORDER:

        conf = round(base_conf[base], 2)

        active_sub = get_active_sub(base, conf)

        all_subs = [s for _, s in WHEEL_SUB_MAP.get(base, [])]

        result.append({
            "category": category,
            "wheelBase": base,
            "confidence": conf,
            "activeSub": active_sub,
            "allSubs": all_subs,
            "fromModel": base not in ["Loved"],
        })

    return result


# ───────────────────────────────────────────────────────────────
# Build Response
# ───────────────────────────────────────────────────────────────

def build_emotion_response(preds: np.ndarray) -> dict:

    max_index = int(np.argmax(preds))

    affectnet_label = AFFECTNET_LABELS[max_index]

    wheel_base_list = get_wheel_base_list(preds)

    top = max(
        wheel_base_list,
        key=lambda x: x["confidence"]
    )

    return {
        "category": top["category"],
        "emotion": top["wheelBase"],
        "subEmotion": top["activeSub"],
        "confidence": top["confidence"],
        "affectnetLabel": affectnet_label,

        "wheelBaseList": wheel_base_list,

        "wheelBaseListSorted": sorted(
            wheel_base_list,
            key=lambda x: x["confidence"],
            reverse=True
        ),
    }


# ───────────────────────────────────────────────────────────────
# Empty Response
# ───────────────────────────────────────────────────────────────

def empty_emotion_response() -> dict:
    return {
        "category": "None",
        "emotion": "No Face",
        "subEmotion": "None",
        "confidence": 0.0,
        "affectnetLabel": "No Face",
        "wheelBaseList": [],
        "wheelBaseListSorted": [],
    }


# ───────────────────────────────────────────────────────────────
# Predict Single Frame
# ───────────────────────────────────────────────────────────────

def predict_emotion(frame: np.ndarray) -> dict:

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    preds = _predict_probs(rgb)
    if preds is None:
        return empty_emotion_response()

    return build_emotion_response(preds)


# ───────────────────────────────────────────────────────────────
# Predict Video Emotion
# ───────────────────────────────────────────────────────────────

def predict_video_emotion(
    video_path: str,
    frame_step: int = 5
) -> dict:

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        return empty_emotion_response()

    frame_count = 0
    sampled_frame_count = 0
    valid_face_frames = 0

    preds_sum = np.zeros(
        len(AFFECTNET_LABELS),
        dtype=np.float64
    )

    frame_results = []

    while True:

        ok, frame = cap.read()

        if not ok:
            break

        frame_count += 1

        if frame_count % frame_step != 0:
            continue

        sampled_frame_count += 1

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        preds = _predict_probs(rgb)

        if preds is None:

            frame_results.append({
                "frameIndex": frame_count,
                "hasFace": False,
                "emotion": "No Face",
                "subEmotion": "None",
                "confidence": 0.0,
                "affectnetLabel": "No Face",
            })

            continue

        preds_sum += preds

        valid_face_frames += 1

        frame_response = build_emotion_response(preds)

        frame_results.append({
            "frameIndex": frame_count,
            "hasFace": True,
            "emotion": frame_response["emotion"],
            "subEmotion": frame_response["subEmotion"],
            "confidence": frame_response["confidence"],
            "affectnetLabel": frame_response["affectnetLabel"],
        })

    cap.release()

    if valid_face_frames == 0:

        result = empty_emotion_response()

        result["videoMeta"] = {
            "totalFramesRead": frame_count,
            "frameStep": frame_step,
            "sampledFrames": sampled_frame_count,
            "validFaceFrames": valid_face_frames,
        }

        result["frameResults"] = frame_results

        return result

    avg_preds = preds_sum / valid_face_frames

    result = build_emotion_response(avg_preds)

    result["videoMeta"] = {
        "totalFramesRead": frame_count,
        "frameStep": frame_step,
        "sampledFrames": sampled_frame_count,
        "validFaceFrames": valid_face_frames,
    }

    result["frameResults"] = frame_results

    return result
