import numpy as np
import librosa
import torch
import warnings
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2Processor

# ── Model Setup ───────────────────────────────────────────────────
# Trained on RAVDESS dataset
print("Loading voice emotion model...")

_processor = Wav2Vec2Processor.from_pretrained(
    "AventIQ-AI/wav2vec2-base_speech_emotion_recognition"
)
_model = Wav2Vec2ForSequenceClassification.from_pretrained(
    "AventIQ-AI/wav2vec2-base_speech_emotion_recognition"
)
_model.eval()
_model = _model.float()   # fix HalfTensor vs FloatTensor mismatch
print("Voice emotion model loaded")

SAMPLE_RATE = 16000

# ── RAVDESS standard label order ─────────────────────────────────
# Model config returns LABEL_0..7 with no names.
# Hardcoded from official model card and RAVDESS dataset encoding:
# 01=neutral, 02=calm, 03=happy, 04=sad, 05=angry, 06=fearful, 07=disgust, 08=surprised
RAVDESS_LABELS = [
    "neutral",    # LABEL_0
    "calm",       # LABEL_1
    "happy",      # LABEL_2
    "sad",        # LABEL_3
    "angry",      # LABEL_4
    "fearful",    # LABEL_5
    "disgust",    # LABEL_6
    "surprised",  # LABEL_7
]

# ── RAVDESS → Wheel base mapping ──────────────────────────────────
VOICE_TO_WHEEL_BASE = {
    "neutral":   "Neutral",
    "calm":      "Neutral",      # calm + neutral → Neutral (summed)
    "happy":     "Happy",
    "sad":       "Sad",
    "angry":     "Angry",
    "fearful":   "Scared",
    "disgust":   "Embarrassed",
    "surprised": "Happy",        # no wheel match → nearest
}

# ── RAVDESS → Category mapping ────────────────────────────────────
VOICE_TO_CATEGORY = {
    "neutral":   "Comfortable",
    "calm":      "Comfortable",
    "happy":     "Comfortable",
    "sad":       "Uncomfortable",
    "angry":     "Uncomfortable",
    "fearful":   "Uncomfortable",
    "disgust":   "Uncomfortable",
    "surprised": "Comfortable",
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
    """
    Build Level 2 list — all 8 wheel base emotions with confidence.
    happy + surprised → Happy (summed)
    neutral + calm    → Neutral (summed)
    """
    base_conf = {base: 0.0 for _, base in WHEEL_ORDER}

    for voice_label, conf in label_scores.items():
        wheel_base = VOICE_TO_WHEEL_BASE.get(voice_label)
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
        # Load + resample to 16kHz + mono.
        # Incoming chunks are webm, so librosa may use audioread fallback; suppress known non-fatal warning spam.
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="PySoundFile failed\\. Trying audioread instead\\.")
            warnings.filterwarnings("ignore", message="librosa\\.core\\.audio\\.__audioread_load")
            audio, _ = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)

        # Reject silent audio
        if np.max(np.abs(audio)) < 0.01:
            return empty_voice_response("Silent audio")

        # Run model
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

        # Map index → emotion name using RAVDESS standard order
        label_scores = {
            RAVDESS_LABELS[i]: float(probs[i])
            for i in range(len(RAVDESS_LABELS))
        }

        return build_voice_response(label_scores)

    except Exception as e:
        return empty_voice_response(f"Error: {str(e)}")