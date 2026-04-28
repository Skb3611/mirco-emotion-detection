from typing import Dict, List, Tuple

WHEEL_ORDER: List[Tuple[str, str]] = [
    ("Uncomfortable", "Sad"),
    ("Uncomfortable", "Scared"),
    ("Uncomfortable", "Angry"),
    ("Uncomfortable", "Embarrassed"),
    ("Comfortable", "Happy"),
    ("Comfortable", "Loved"),
    ("Comfortable", "Confident"),
    ("Comfortable", "Neutral"),
]

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


def _get_active_sub(base: str, confidence: float) -> str:
    for min_conf, sub_label in WHEEL_SUB_MAP.get(base, []):
        if confidence >= min_conf:
            return sub_label
    return "None"


def _scores_from_response(response: dict) -> Dict[str, float]:
    scores = {base: 0.0 for _, base in WHEEL_ORDER}
    for item in response.get("wheelBaseList", []):
        base = item.get("wheelBase")
        conf = float(item.get("confidence", 0.0))
        if base in scores:
            scores[base] = conf
    return scores


def _audio_quality(audio_result: dict) -> float:
    """
    Entropy-based quality — measures how decisive the audio model is.
    Near-uniform output (14% each) → quality ≈ 0.0
    One clear winner (80%+) → quality ≈ 1.0
    This prevents a confused/random audio model from polluting fusion.
    """
    if not audio_result or audio_result.get("category") == "None":
        return 0.0

    scores = _scores_from_response(audio_result)
    values = [v for v in scores.values() if v > 0]
    if not values:
        return 0.0

    total = sum(values)
    if total == 0:
        return 0.0

    # Normalize to 0-1 probabilities
    probs = [v / total for v in values]
    max_prob = max(probs)

    # Random baseline for 8 classes = 0.125 (12.5%)
    # Scale quality from 0 (at random) to 1 (at 100% confident)
    quality = (max_prob - 0.125) / (1.0 - 0.125)
    return max(0.0, min(1.0, quality))


def _video_quality(video_result: dict) -> float:
    """
    Face detection rate — how many sampled frames had a detected face.
    High detection rate = reliable video signal.
    """
    if not video_result or video_result.get("category") == "None":
        return 0.0

    video_meta = video_result.get("videoMeta", {})
    sampled    = float(video_meta.get("sampledFrames", 0.0))
    valid      = float(video_meta.get("validFaceFrames", 0.0))

    if sampled <= 0:
        return 0.0

    return max(0.0, min(1.0, valid / sampled))


def fuse_audio_video(audio_result: dict, video_result: dict) -> dict:
    audio_available = bool(audio_result) and audio_result.get("category") != "None"
    video_available = bool(video_result) and video_result.get("category") != "None"

    # ── Single modality fallback ──────────────────────────────────
    if audio_available and not video_available:
        combined = dict(audio_result)
        combined["fusionMeta"] = {
            "audioQuality": round(_audio_quality(audio_result), 4),
            "videoQuality": 0.0,
            "weightsUsed":  {"audio": 1.0, "video": 0.0},
        }
        return combined

    if video_available and not audio_available:
        combined = dict(video_result)
        combined["fusionMeta"] = {
            "audioQuality": 0.0,
            "videoQuality": round(_video_quality(video_result), 4),
            "weightsUsed":  {"audio": 0.0, "video": 1.0},
        }
        return combined

    if not audio_available and not video_available:
        return {
            "category":            "None",
            "emotion":             "None",
            "subEmotion":          "None",
            "confidence":          0.0,
            "wheelBaseList":       [],
            "wheelBaseListSorted": [],
            "fusionMeta": {
                "audioQuality": 0.0,
                "videoQuality": 0.0,
                "weightsUsed":  {"audio": 0.0, "video": 0.0},
            },
        }

    # ── Compute quality scores ────────────────────────────────────
    audio_q = _audio_quality(audio_result)
    video_q = _video_quality(video_result)

    # Base weights: video 60%, audio 40%
    # Scaled by quality so a confused modality gets less influence
    audio_weight = 0.4 * audio_q
    video_weight = 0.6 * video_q

    # If both quality scores are 0, fall back to base weights
    if audio_weight + video_weight == 0:
        audio_weight = 0.4
        video_weight = 0.6

    total        = audio_weight + video_weight
    audio_weight /= total
    video_weight /= total

    # ── Combine wheel base scores ─────────────────────────────────
    audio_scores = _scores_from_response(audio_result)
    video_scores = _scores_from_response(video_result)

    combined_scores: Dict[str, float] = {}
    for _, base in WHEEL_ORDER:
        combined_scores[base] = round(
            (audio_weight * audio_scores.get(base, 0.0)) +
            (video_weight * video_scores.get(base, 0.0)),
            2
        )

    # ── Build wheel base list with sub-emotions ───────────────────
    wheel_base_list = []
    for category, base in WHEEL_ORDER:
        conf       = combined_scores[base]
        active_sub = _get_active_sub(base, conf)
        all_subs   = [s for _, s in WHEEL_SUB_MAP.get(base, [])]
        wheel_base_list.append({
            "category":   category,
            "wheelBase":  base,
            "confidence": conf,
            "activeSub":  active_sub,
            "allSubs":    all_subs,
            "fromFusion": True,
        })

    top = max(wheel_base_list, key=lambda x: x["confidence"])

    return {
        "category":   top["category"],
        "emotion":    top["wheelBase"],
        "subEmotion": top["activeSub"],
        "confidence": top["confidence"],
        "wheelBaseList": wheel_base_list,
        "wheelBaseListSorted": sorted(
            wheel_base_list,
            key=lambda x: x["confidence"],
            reverse=True,
        ),
        "fusionMeta": {
            "audioQuality": round(audio_q, 4),
            "videoQuality": round(video_q, 4),
            "weightsUsed": {
                "audio": round(audio_weight, 4),
                "video": round(video_weight, 4),
            },
        },
    }