import cv2
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D

# ── Model Architecture ───────────────────────────────────────────
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48, 48, 1)))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(7, activation='softmax'))

model.load_weights("src/model.h5")

face_cascade = cv2.CascadeClassifier(
    "src/haarcascade_frontalface_default.xml"
)

# ── FER13 labels (DO NOT change order) ──────────────────────────
FER13_LABELS = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]

# ── FER13 → Wheel base mapping ───────────────────────────────────
FER13_TO_WHEEL_BASE = {
    "Angry":    "Angry",
    "Disgust":  "Embarrassed",
    "Fear":     "Scared",
    "Happy":    "Happy",
    "Neutral":  "Neutral",
    "Sad":      "Sad",
    "Surprise": "Happy",        # no wheel match → nearest
}

# ── FER13 → Category mapping ─────────────────────────────────────
FER13_TO_CATEGORY = {
    "Angry":    "Uncomfortable",
    "Disgust":  "Uncomfortable",
    "Fear":     "Uncomfortable",
    "Sad":      "Uncomfortable",
    "Happy":    "Comfortable",
    "Neutral":  "Comfortable",
    "Surprise": "Comfortable",
}

# ── Sub-emotion thresholds (for active sub calculation) ──────────
WHEEL_SUB_MAP = {
    "Sad":         [(75.0, "Hurt"),        (45.0, "Disappointed"), (0.0, "Lonely")   ],
    "Scared":      [(75.0, "Overwhelmed"), (45.0, "Powerless"),    (0.0, "Anxious")  ],
    "Angry":       [(75.0, "Annoyed"),     (45.0, "Jealous"),      (0.0, "Bored")    ],
    "Embarrassed": [(75.0, "Ashamed"),     (45.0, "Excluded"),     (0.0, "Guilty")   ],
    "Happy":       [(75.0, "Excited"),     (45.0, "Grateful"),     (0.0, "Caring")   ],
    "Neutral":     [(75.0, "Creative"),    (45.0, "Calm"),         (0.0, "Relaxed")  ],
    "Loved":       [(75.0, "Respected"),   (45.0, "Valued"),       (0.0, "Accepted") ],
    "Confident":   [(75.0, "Powerful"),    (45.0, "Brave"),        (0.0, "Hopeful")  ],
}

# ── Fixed wheel order (category → base) ──────────────────────────
# This defines display order — Uncomfortable first, then Comfortable
WHEEL_ORDER = [
    ("Uncomfortable", "Sad"),
    ("Uncomfortable", "Scared"),
    ("Uncomfortable", "Angry"),
    ("Uncomfortable", "Embarrassed"),
    ("Comfortable",   "Happy"),
    ("Comfortable",   "Loved"),         # not in FER13 → always 0%
    ("Comfortable",   "Confident"),     # not in FER13 → always 0%
    ("Comfortable",   "Neutral"),
]


def get_active_sub(base: str, confidence: float) -> str:
    """Get active sub-emotion for a base at given confidence."""
    for min_conf, sub_label in WHEEL_SUB_MAP.get(base, []):
        if confidence >= min_conf:
            return sub_label
    return "None"


def get_wheel_base_list(preds: np.ndarray) -> list:
    """
    Build Level 2 list — all 8 wheel base emotions with confidence.

    Since Surprise and Happy both map to Happy base,
    their confidences are summed under Happy.

    Returns list in fixed wheel order (not sorted),
    so the caller can choose to sort or display as-is.
    """
    # Step 1: accumulate confidence per wheel base
    base_conf = {base: 0.0 for _, base in WHEEL_ORDER}

    for i, fer_label in enumerate(FER13_LABELS):
        conf       = float(preds[i]) * 100
        wheel_base = FER13_TO_WHEEL_BASE[fer_label]
        if wheel_base in base_conf:
            base_conf[wheel_base] += conf

    # Step 2: build result list in wheel order
    result = []
    for category, base in WHEEL_ORDER:
        conf       = round(base_conf[base], 2)
        active_sub = get_active_sub(base, conf)
        all_subs   = [s for _, s in WHEEL_SUB_MAP.get(base, [])]

        result.append({
            "category":   category,
            "wheelBase":  base,
            "confidence": conf,
            "activeSub":  active_sub,
            "allSubs":    all_subs,
            "fromFer13":  base != "Loved" and base != "Confident",
        })

    return result


def predict_emotion(frame: np.ndarray) -> dict:
    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        roi = gray[y:y+h, x:x+w]
        roi = cv2.resize(roi, (48, 48))
        roi = roi / 255.0
        roi = roi.reshape(1, 48, 48, 1)

        preds      = model.predict(roi, verbose=0)[0]
        max_index  = int(np.argmax(preds))
        fer13_label = FER13_LABELS[max_index]
        confidence  = float(preds[max_index]) * 100

        # Get Level 2 list
        wheel_base_list = get_wheel_base_list(preds)

        # Top result = highest confidence wheel base
        top = max(wheel_base_list, key=lambda x: x["confidence"])

        return {
            # ── Primary detection ─────────────────────────────────
            "category":   top["category"],
            "emotion":    top["wheelBase"],
            "subEmotion": top["activeSub"],
            "confidence": top["confidence"],
            "fer13Label": fer13_label,

            # ── Level 2: all 8 wheel base emotions ────────────────
            # In fixed wheel order (Uncomfortable → Comfortable)
            "wheelBaseList": wheel_base_list,

            # ── Level 2: sorted by confidence (for ranked display) ─
            "wheelBaseListSorted": sorted(
                wheel_base_list,
                key=lambda x: x["confidence"],
                reverse=True
            ),
        }

    return {
        "category":            "None",
        "emotion":             "No Face",
        "subEmotion":          "None",
        "confidence":          0.0,
        "fer13Label":          "No Face",
        "wheelBaseList":       [],
        "wheelBaseListSorted": [],
    }