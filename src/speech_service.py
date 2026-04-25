import os
import subprocess
from transformers import pipeline

# =========================
# LOAD MODEL (ONCE)
# =========================
print("Loading Transformers model...")
classifier = pipeline(
    "audio-classification",
    model="ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition"
)
print("Model loaded.")

# =========================
# CONVERT WEBM → WAV
# =========================
def convert_to_wav(input_path, output_path):
    command = [
        "ffmpeg",
        "-y",
        "-i", input_path,
        "-ar", "16000",
        "-ac", "1",
        output_path
    ]
    subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

# =========================
# PREDICT EMOTION
# =========================
def predict_emotion(audio_path):
    results = classifier(audio_path)
    # The pipeline returns a list of dictionaries like:
    # [{'label': 'angry', 'score': 0.91}, ...]
    # The first one is the highest score
    top = results[0]

    return {
        "emotion": top["label"],
        "confidence": float(top["score"])
    }

# =========================
# MAIN PIPELINE
# =========================
def process_audio(webm_path, wav_path):
    # skip bad files
    if os.path.getsize(webm_path) < 1000:
        return {"status": "skip"}

    convert_to_wav(webm_path, wav_path)

    if not os.path.exists(wav_path) or os.path.getsize(wav_path) == 0:
        return {"status": "conversion_failed"}

    return predict_emotion(wav_path)