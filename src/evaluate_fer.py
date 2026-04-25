import argparse
import json
import os
from typing import Dict, List, Tuple

import cv2
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D


LABELS = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]
FOLDER_TO_LABEL = {
    "angry": "Angry",
    "disgusted": "Disgust",
    "disgust": "Disgust",
    "fearful": "Fear",
    "fear": "Fear",
    "happy": "Happy",
    "neutral": "Neutral",
    "sad": "Sad",
    "surprised": "Surprise",
    "surprise": "Surprise",
}


def build_model():
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), activation="relu", input_shape=(48, 48, 1)))
    model.add(Conv2D(64, kernel_size=(3, 3), activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(128, kernel_size=(3, 3), activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, kernel_size=(3, 3), activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(1024, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(7, activation="softmax"))
    return model


def collect_image_samples(test_dir: str) -> List[Tuple[str, int]]:
    samples: List[Tuple[str, int]] = []
    for folder_name in sorted(os.listdir(test_dir)):
        folder_path = os.path.join(test_dir, folder_name)
        if not os.path.isdir(folder_path):
            continue
        label_name = FOLDER_TO_LABEL.get(folder_name.lower())
        if label_name is None:
            continue
        y_true = LABELS.index(label_name)
        for file_name in os.listdir(folder_path):
            if file_name.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".webp")):
                samples.append((os.path.join(folder_path, file_name), y_true))
    return samples


def compute_metrics(y_true: List[int], y_pred: List[int]) -> Dict:
    n = len(LABELS)
    cm = np.zeros((n, n), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1

    total = int(cm.sum())
    correct = int(np.trace(cm))
    accuracy = (correct / total) if total else 0.0

    per_class = {}
    recalls = []
    precisions = []
    f1s = []

    for i, label in enumerate(LABELS):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        support = cm[i, :].sum()

        precision = float(tp / (tp + fp)) if (tp + fp) else 0.0
        recall = float(tp / (tp + fn)) if (tp + fn) else 0.0
        f1 = float((2 * precision * recall) / (precision + recall)) if (precision + recall) else 0.0

        per_class[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": int(support),
        }
        precisions.append(precision)
        f1s.append(f1)
        if support > 0:
            recalls.append(recall)

    macro_precision = float(np.mean(precisions)) if precisions else 0.0
    macro_recall = float(np.mean([v["recall"] for v in per_class.values()])) if per_class else 0.0
    macro_f1 = float(np.mean(f1s)) if f1s else 0.0
    uar = float(np.mean(recalls)) if recalls else 0.0

    return {
        "accuracy": accuracy,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "uar": uar,
        "num_samples": total,
        "confusion_matrix": cm.tolist(),
        "labels": LABELS,
        "per_class": per_class,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate FER model on a labeled test image folder.")
    parser.add_argument("--test_dir", required=True, help="Folder with class subfolders (angry, happy, ...).")
    parser.add_argument("--weights", default="src/model.h5", help="Path to model weights file.")
    parser.add_argument("--output", default="src/fer_metrics.json", help="Path to write metrics JSON.")
    args = parser.parse_args()

    if not os.path.exists(args.test_dir):
        raise FileNotFoundError(f"Test directory not found: {args.test_dir}")
    if not os.path.exists(args.weights):
        raise FileNotFoundError(f"Weights file not found: {args.weights}")

    samples = collect_image_samples(args.test_dir)
    if not samples:
        raise RuntimeError("No labeled images found in test_dir.")

    model = build_model()
    model.load_weights(args.weights)

    y_true: List[int] = []
    y_pred: List[int] = []

    for image_path, gt in samples:
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        img = cv2.resize(img, (48, 48))
        x = (img / 255.0).reshape(1, 48, 48, 1)
        preds = model.predict(x, verbose=0)[0]
        pred_idx = int(np.argmax(preds))

        y_true.append(gt)
        y_pred.append(pred_idx)

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
