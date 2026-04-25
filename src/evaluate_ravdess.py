import argparse
import json
import os
from typing import Dict, List, Optional, Tuple

from transformers import pipeline


RAVDESS_CODE_TO_LABEL = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised",
}
LABELS = ["neutral", "calm", "happy", "sad", "angry", "fearful", "disgust", "surprised"]


def parse_gt_from_filename(file_name: str) -> Optional[str]:
    # RAVDESS naming: xx-xx-EMOTION-xx-xx-xx-xx.wav
    parts = file_name.split("-")
    if len(parts) < 3:
        return None
    code = parts[2]
    return RAVDESS_CODE_TO_LABEL.get(code)


def parse_gt_from_folder(path: str) -> Optional[str]:
    parent = os.path.basename(os.path.dirname(path)).lower()
    if parent in LABELS:
        return parent
    return None


def collect_audio_samples(root_dir: str) -> List[Tuple[str, str]]:
    samples: List[Tuple[str, str]] = []
    for root, _, files in os.walk(root_dir):
        for file_name in files:
            if not file_name.lower().endswith(".wav"):
                continue
            fpath = os.path.join(root, file_name)
            gt = parse_gt_from_filename(file_name)
            if gt is None:
                gt = parse_gt_from_folder(fpath)
            if gt is None:
                continue
            samples.append((fpath, gt))
    return samples


def compute_metrics(y_true: List[str], y_pred: List[str]) -> Dict:
    idx = {l: i for i, l in enumerate(LABELS)}
    n = len(LABELS)
    cm = [[0 for _ in range(n)] for _ in range(n)]
    for t, p in zip(y_true, y_pred):
        cm[idx[t]][idx[p]] += 1

    total = sum(sum(r) for r in cm)
    correct = sum(cm[i][i] for i in range(n))
    accuracy = (correct / total) if total else 0.0

    per_class = {}
    precisions: List[float] = []
    recalls_nonempty: List[float] = []
    recalls_all: List[float] = []
    f1s: List[float] = []

    for i, label in enumerate(LABELS):
        tp = cm[i][i]
        fp = sum(cm[r][i] for r in range(n)) - tp
        fn = sum(cm[i][c] for c in range(n)) - tp
        support = sum(cm[i][c] for c in range(n))

        precision = (tp / (tp + fp)) if (tp + fp) else 0.0
        recall = (tp / (tp + fn)) if (tp + fn) else 0.0
        f1 = ((2 * precision * recall) / (precision + recall)) if (precision + recall) else 0.0

        per_class[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }
        precisions.append(precision)
        recalls_all.append(recall)
        if support > 0:
            recalls_nonempty.append(recall)
        f1s.append(f1)

    return {
        "accuracy": accuracy,
        "macro_precision": (sum(precisions) / len(precisions)) if precisions else 0.0,
        "macro_recall": (sum(recalls_all) / len(recalls_all)) if recalls_all else 0.0,
        "macro_f1": (sum(f1s) / len(f1s)) if f1s else 0.0,
        "uar": (sum(recalls_nonempty) / len(recalls_nonempty)) if recalls_nonempty else 0.0,
        "num_samples": total,
        "labels": LABELS,
        "confusion_matrix": cm,
        "per_class": per_class,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate speech emotion model on RAVDESS-style dataset.")
    parser.add_argument("--data_dir", required=True, help="Root folder containing RAVDESS .wav files.")
    parser.add_argument(
        "--model",
        default="ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition",
        help="HuggingFace model id",
    )
    parser.add_argument("--output", default="src/ravdess_metrics.json", help="Path to write metrics JSON.")
    args = parser.parse_args()

    if not os.path.exists(args.data_dir):
        raise FileNotFoundError(f"data_dir not found: {args.data_dir}")

    samples = collect_audio_samples(args.data_dir)
    if not samples:
        raise RuntimeError("No labeled .wav files found in data_dir.")

    classifier = pipeline("audio-classification", model=args.model)

    y_true: List[str] = []
    y_pred: List[str] = []

    for path, gt in samples:
        out = classifier(path)
        pred = str(out[0]["label"]).lower()

        # basic label normalization
        if pred == "fear":
            pred = "fearful"
        if pred == "surprise":
            pred = "surprised"
        if pred not in LABELS:
            continue

        y_true.append(gt)
        y_pred.append(pred)

    metrics = compute_metrics(y_true, y_pred)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(json.dumps({
        "accuracy": round(metrics["accuracy"] * 100, 2),
        "macro_precision": round(metrics["macro_precision"] * 100, 2),
        "macro_recall": round(metrics["macro_recall"] * 100, 2),
        "macro_f1": round(metrics["macro_f1"] * 100, 2),
        "uar": round(metrics["uar"] * 100, 2),
        "num_samples": metrics["num_samples"],
        "output": args.output,
    }, indent=2))


if __name__ == "__main__":
    main()
