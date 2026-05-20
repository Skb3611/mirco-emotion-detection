import os
import numpy as np
import librosa
import torch
import warnings
from transformers import (
    AutoModelForAudioClassification,
    Wav2Vec2Processor,
    AutoFeatureExtractor,
    AutoModelForAudioClassification as AutoAudioModel,
)

# ── Model Setup ───────────────────────────────────────────────────
# PRIMARY: Dpngtm — ~80% accuracy, works across languages because
# emotion is conveyed via acoustic prosody (pitch, energy, rhythm),
# not language-specific words.
#
# SECONDARY (optional): facebook/mms-lid-126 — language detector.
# Disable via env var if container has no internet:
#   DISABLE_LANG_DETECT=1   → skip LID entirely (fast startup)
#   TRANSFORMERS_OFFLINE=1  → use only locally cached models
#
# To pre-cache MMS-LID into your Docker image, add to Dockerfile:
#   RUN python -c "from transformers import AutoFeatureExtractor, \
#       AutoModelForAudioClassification; \
#       AutoFeatureExtractor.from_pretrained('facebook/mms-lid-126'); \
#       AutoModelForAudioClassification.from_pretrained('facebook/mms-lid-126')"
#
# SUPPORTED LANGUAGES: English, Hindi (हिन्दी), Marathi (मराठी)

SAMPLE_RATE = 16000

# ── Load primary emotion model ────────────────────────────────────
print("Loading primary emotion model (Dpngtm)...")
_emo_processor = Wav2Vec2Processor.from_pretrained(
    "Dpngtm/wav2vec2-emotion-recognition"
)
_emo_model = AutoModelForAudioClassification.from_pretrained(
    "Dpngtm/wav2vec2-emotion-recognition"
)
_emo_model.eval()
_emo_model = _emo_model.float()
print("✓ Emotion model loaded")

VOICE_LABELS = [
    _emo_model.config.id2label[i]
    for i in range(_emo_model.config.num_labels)
]
print("Emotion labels:", VOICE_LABELS)

# ── Load language identification model (optional) ─────────────────
_lid_feature_extractor = None
_lid_model             = None
LID_AVAILABLE          = False
LID_LABELS             = []

_DISABLE_LID = os.environ.get("DISABLE_LANG_DETECT", "").strip() in ("1", "true", "yes")

if _DISABLE_LID:
    print("⚠ Language detection disabled via DISABLE_LANG_DETECT env var.")
else:
    print("Loading language identification model (MMS-LID-126)...")
    print("  Tip: set DISABLE_LANG_DETECT=1 to skip if container has no internet.")

    import threading

    _lid_result    = {}
    _LID_TIMEOUT_S = int(os.environ.get("LID_LOAD_TIMEOUT", "30"))   # default 30s

    def _load_lid():
        try:
            fe = AutoFeatureExtractor.from_pretrained("facebook/mms-lid-126")
            m  = AutoAudioModel.from_pretrained("facebook/mms-lid-126")
            m.eval()
            m = m.float()
            _lid_result["fe"]    = fe
            _lid_result["model"] = m
            _lid_result["ok"]    = True
        except Exception as e:
            _lid_result["ok"]    = False
            _lid_result["error"] = str(e)

    _t = threading.Thread(target=_load_lid, daemon=True)
    _t.start()
    _t.join(timeout=_LID_TIMEOUT_S)

    if _t.is_alive():
        print(f"⚠ Language ID model load timed out after {_LID_TIMEOUT_S}s "
              f"(likely no internet). Language detection disabled.")
        print("  → Set DISABLE_LANG_DETECT=1 in your Docker run command to skip this step.")
        LID_AVAILABLE = False
    elif _lid_result.get("ok"):
        _lid_feature_extractor = _lid_result["fe"]
        _lid_model             = _lid_result["model"]
        LID_AVAILABLE          = True
        LID_LABELS             = [
            _lid_model.config.id2label[i]
            for i in range(_lid_model.config.num_labels)
        ]
        print("✓ Language ID model loaded")
    else:
        print(f"⚠ Language ID model failed: {_lid_result.get('error')}. "
              f"Language detection disabled.")

# ── Language code → display name ──────────────────────────────────
LANGUAGE_DISPLAY = {
    "eng": "English",
    "hin": "Hindi",
    "mar": "Marathi",
}

SUPPORTED_LANGUAGES = {"eng", "hin", "mar"}

# ── Per-language confidence calibration ──────────────────────────
# Dpngtm was trained on English (RAVDESS-style). Prosodic features
# still carry emotion in Hindi/Marathi, but confidence may be flatter.
# Tune these after evaluating on your own data.
LANG_CONFIDENCE_SCALE = {
    "eng":     1.00,
    "hin":     0.90,
    "mar":     0.88,
    "unknown": 0.80,
}

# ── Emotion → Wheel mappings ──────────────────────────────────────
VOICE_TO_WHEEL_BASE = {
    "neutral":   "Neutral",
    "calm":      "Neutral",
    "happy":     "Happy",
    "sad":       "Sad",
    "angry":     "Angry",
    "fearful":   "Scared",
    "fear":      "Scared",
    "disgust":   "Embarrassed",
    "surprised": "Happy",
    "surprise":  "Happy",
}

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


# ── Language detection ────────────────────────────────────────────

def detect_language(audio: np.ndarray) -> dict:
    """
    Detect language from audio waveform.
    Returns unknown gracefully if LID model is not available.
    """
    if not LID_AVAILABLE:
        return {
            "language":          "unknown",
            "language_display":  "Unknown",
            "lang_confidence":   0.0,
            "is_supported":      True,   # don't block emotion detection
            "top_languages":     {},
        }

    try:
        inputs = _lid_feature_extractor(
            audio,
            sampling_rate=SAMPLE_RATE,
            return_tensors="pt",
            padding=True,
        )
        with torch.no_grad():
            logits = _lid_model(**inputs).logits
            probs  = torch.nn.functional.softmax(logits, dim=-1)[0].numpy()

        top_indices = np.argsort(probs)[::-1][:5]
        top_langs   = {
            LID_LABELS[i]: round(float(probs[i]) * 100, 2)
            for i in top_indices
        }
        top_code = LID_LABELS[top_indices[0]]
        top_conf = float(probs[top_indices[0]]) * 100

        return {
            "language":          top_code,
            "language_display":  LANGUAGE_DISPLAY.get(top_code, top_code.upper()),
            "lang_confidence":   round(top_conf, 2),
            "is_supported":      top_code in SUPPORTED_LANGUAGES,
            "top_languages":     top_langs,
        }

    except Exception as e:
        return {
            "language":          "unknown",
            "language_display":  "Unknown",
            "lang_confidence":   0.0,
            "is_supported":      True,
            "top_languages":     {},
            "lang_detect_error": str(e),
        }


# ── Emotion prediction ────────────────────────────────────────────

def _run_emotion_model(audio: np.ndarray) -> dict:
    inputs = _emo_processor(
        audio,
        sampling_rate=SAMPLE_RATE,
        return_tensors="pt",
        padding=True,
        max_length=160000,
        truncation=True,
    )
    with torch.no_grad():
        outputs = _emo_model(inputs.input_values)
        probs   = torch.nn.functional.softmax(
            outputs.logits, dim=-1
        )[0].numpy()

    return {VOICE_LABELS[i]: float(probs[i]) for i in range(len(VOICE_LABELS))}


def get_active_sub(base: str, confidence: float) -> str:
    for min_conf, sub_label in WHEEL_SUB_MAP.get(base, []):
        if confidence >= min_conf:
            return sub_label
    return "None"


def get_wheel_base_list(label_scores: dict, lang_scale: float = 1.0) -> list:
    base_conf = {base: 0.0 for _, base in WHEEL_ORDER}

    for voice_label, conf in label_scores.items():
        wheel_base = VOICE_TO_WHEEL_BASE.get(voice_label.lower())
        if wheel_base and wheel_base in base_conf:
            base_conf[wheel_base] += conf * 100 * lang_scale

    result = []
    for category, base in WHEEL_ORDER:
        conf       = round(min(base_conf[base], 100.0), 2)
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


def build_voice_response(label_scores: dict, lang_info: dict) -> dict:
    top_voice_label = max(label_scores, key=label_scores.get)
    lang_scale      = LANG_CONFIDENCE_SCALE.get(
        lang_info.get("language", "unknown"), 0.80
    )

    wheel_base_list = get_wheel_base_list(label_scores, lang_scale)
    top             = max(wheel_base_list, key=lambda x: x["confidence"])

    return {
        "category":              top["category"],
        "emotion":               top["wheelBase"],
        "subEmotion":            top["activeSub"],
        "confidence":            top["confidence"],
        "voiceLabel":            top_voice_label,
        "voiceScores": {
            k: round(v * 100, 2)
            for k, v in sorted(
                label_scores.items(), key=lambda x: x[1], reverse=True
            )
        },
        "wheelBaseList":         wheel_base_list,
        "wheelBaseListSorted":   sorted(
            wheel_base_list, key=lambda x: x["confidence"], reverse=True
        ),
        "language":              lang_info.get("language", "unknown"),
        "languageDisplay":       lang_info.get("language_display", "Unknown"),
        "languageConfidence":    lang_info.get("lang_confidence", 0.0),
        "isLanguageSupported":   lang_info.get("is_supported", True),
        "topLanguages":          lang_info.get("top_languages", {}),
        "langConfidenceScale":   lang_scale,
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
        "language":            "unknown",
        "languageDisplay":     "Unknown",
        "languageConfidence":  0.0,
        "isLanguageSupported": False,
        "topLanguages":        {},
        "langConfidenceScale": 0.0,
    }


# ── Public API ────────────────────────────────────────────────────

def predict_voice_emotion(audio_path: str) -> dict:
    """
    Predict emotion from an audio file with optional language detection.

    Supports WAV, MP3, FLAC, OGG (any format librosa can read).
    Supports English, Hindi, and Marathi.

    Environment variables:
      DISABLE_LANG_DETECT=1     skip language detection entirely
      LID_LOAD_TIMEOUT=30       seconds to wait for LID model load (default 30)
      TRANSFORMERS_OFFLINE=1    use only locally cached HuggingFace models

    Returns a dict with emotion, subEmotion, confidence, category,
    language, languageDisplay, languageConfidence, and wheel data.
    """
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="PySoundFile failed\\. Trying audioread instead\\."
            )
            warnings.filterwarnings(
                "ignore",
                message="librosa\\.core\\.audio\\.__audioread_load"
            )
            audio, _ = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)

        if np.max(np.abs(audio)) < 0.01:
            return empty_voice_response("Silent audio")

        lang_info    = detect_language(audio)
        label_scores = _run_emotion_model(audio)
        return build_voice_response(label_scores, lang_info)

    except Exception as e:
        return empty_voice_response(f"Error: {str(e)}")


def predict_voice_emotion_batch(audio_paths: list) -> list:
    """Run predict_voice_emotion on a list of file paths."""
    return [predict_voice_emotion(p) for p in audio_paths]