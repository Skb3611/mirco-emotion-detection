import numpy as np
import librosa
import torch
import warnings
from transformers import AutoModelForAudioClassification, Wav2Vec2Processor

# ── Model Setup ───────────────────────────────────────────────────
# Dpngtm — trained on ~12,000 files from multiple datasets, ~80% accuracy
# Labels: neutral, happy, sad, angry, fearful, disgust, surprised
# (calm merged into neutral)
print("Loading voice emotion model (Dpngtm multi-dataset)...")

_processor = Wav2Vec2Processor.from_pretrained(
    "Dpngtm/wav2vec2-emotion-recognition"
)
_model = AutoModelForAudioClassification.from_pretrained(
    "Dpngtm/wav2vec2-emotion-recognition"
)
_model.eval()
_model = _model.float()
print("✓ Voice emotion model loaded")

SAMPLE_RATE = 16000

# ── Labels — read from model config ──────────────────────────────
# Dpngtm properly saves id2label so we read it directly
VOICE_LABELS = [
    _model.config.id2label[i]
    for i in range(_model.config.num_labels)
]
print("Labels:", VOICE_LABELS)

# ── Label → Wheel base mapping ────────────────────────────────────
VOICE_TO_WHEEL_BASE = {
    "neutral":   "Neutral",
    "calm":      "Neutral",      # in case calm appears
    "happy":     "Happy",
    "sad":       "Sad",
    "angry":     "Angry",
    "fearful":   "Scared",
    "fear":      "Scared",
    "disgust":   "Embarrassed",
    "surprised": "Happy",
    "surprise":  "Happy",
}

# ── Label → Category mapping ──────────────────────────────────────
VOICE_TO_CATEGORY = {
    "neutral":   "Comfortable",
    "calm":      "Comfortable",
    "happy":     "Comfortable",
    "sad":       "Uncomfortable",
    "angry":     "Uncomfortable",
    "fearful":   "Uncomfortable",
    "fear":      "Uncomfortable",
    "disgust":   "Uncomfortable",
    "surprised": "Comfortable",
    "surprise":  "Comfortable",
}

# ── Sub-emotion thresholds (identical to face detector) ───────────
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

# ── Fixed wheel order (identical to face detector) ────────────────
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


def get_active_sub(base: str, confidence: float) -> str:
    for min_conf, sub_label in WHEEL_SUB_MAP.get(base, []):
        if confidence >= min_conf:
            return sub_label
    return "None"


def get_wheel_base_list(label_scores: dict) -> list:
    base_conf = {base: 0.0 for _, base in WHEEL_ORDER}

    for voice_label, conf in label_scores.items():
        wheel_base = VOICE_TO_WHEEL_BASE.get(voice_label.lower())
        if wheel_base and wheel_base in base_conf:
            base_conf[wheel_base] += conf * 100

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
            "fromVoice":  base not in ("Loved", "Confident"),
        })

    return result


def build_voice_response(label_scores: dict) -> dict:
    top_voice_label = max(label_scores, key=label_scores.get)

    wheel_base_list = get_wheel_base_list(label_scores)
    top             = max(wheel_base_list, key=lambda x: x["confidence"])

    return {
        "category":   top["category"],
        "emotion":    top["wheelBase"],
        "subEmotion": top["activeSub"],
        "confidence": top["confidence"],
        "voiceLabel": top_voice_label,
        "voiceScores": {
            k: round(v * 100, 2)
            for k, v in sorted(
                label_scores.items(),
                key=lambda x: x[1],
                reverse=True
            )
        },
        "wheelBaseList": wheel_base_list,
        "wheelBaseListSorted": sorted(
            wheel_base_list,
            key=lambda x: x["confidence"],
            reverse=True
        ),
    }


def empty_voice_response(reason: str = "No audio") -> dict:
    return {
        "category":            "None",
        "emotion":             reason,
        "subEmotion":          "None",
        "confidence":          0.0,
        "voiceLabel":          reason,
        "voiceScores":         {},
        "wheelBaseList":       [],
        "wheelBaseListSorted": [],
    }


def predict_voice_emotion(audio_path: str) -> dict:
    """
    Predict emotion from an audio file.
    Supports WAV, MP3, FLAC, OGG — any format librosa can read.
    This is the only public function — call this from your API.
    """
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="PySoundFile failed\\. Trying audioread instead\\.")
            warnings.filterwarnings("ignore", message="librosa\\.core\\.audio\\.__audioread_load")
            audio, _ = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)

        if np.max(np.abs(audio)) < 0.01:
            return empty_voice_response("Silent audio")

        inputs = _processor(
            audio,
            sampling_rate=SAMPLE_RATE,
            return_tensors="pt",
            padding=True,
            max_length=160000,
            truncation=True
        )

        with torch.no_grad():
            outputs = _model(inputs.input_values)
            probs   = torch.nn.functional.softmax(
                outputs.logits, dim=-1
            )[0].numpy()

        label_scores = {
            VOICE_LABELS[i]: float(probs[i])
            for i in range(len(VOICE_LABELS))
        }

        return build_voice_response(label_scores)

    except Exception as e:
        return empty_voice_response(f"Error: {str(e)}")