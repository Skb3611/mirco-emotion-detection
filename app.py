from flask import Flask, render_template, request, jsonify ,send_from_directory
import cv2
import numpy as np
from src.detector import predict_emotion, predict_video_emotion
from flask_cors import CORS
from src.fusion import fuse_audio_video
try:
    from src.voice_detector import predict_voice_emotion
except ModuleNotFoundError:
    # Fallback for runtimes where src is not on import path.
    from voice_detector import predict_voice_emotion
import os
import uuid

EMPTY_MODALITY_RESPONSE = {
    "category": "None",
    "emotion": "None",
    "subEmotion": "None",
    "confidence": 0.0,
    "wheelBaseList": [],
    "wheelBaseListSorted": [],
}


app = Flask(__name__, static_folder="static", static_url_path="")
CORS(app)

@app.route("/")
def serve_frontend():
    return send_from_directory(app.static_folder, "index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"emotion": "No image received"})

    file = request.files["image"]
    npimg = np.frombuffer(file.read(), np.uint8)
    frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    if frame is None:
        return jsonify({"emotion": "Invalid frame"})

    result = predict_emotion(frame)
    if isinstance(result, dict):
        return jsonify(result)
    else:
        return jsonify({"result": result})

@app.route("/predict-audio", methods=["POST"])
def predict_audio():
    file = request.files.get("audio")

    if not file:
        return jsonify({"error": "No file"}), 400

    os.makedirs("temp", exist_ok=True)

    uid = str(uuid.uuid4())
    webm_path = f"temp/{uid}.webm"

    file.save(webm_path)

    try:
        result = predict_voice_emotion(webm_path)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        if os.path.exists(webm_path):
            os.remove(webm_path)


@app.route("/predict-video", methods=["POST"])
def predict_video():
    file = request.files.get("video")

    if not file:
        return jsonify({"error": "No video file"}), 400

    os.makedirs("temp", exist_ok=True)

    uid = str(uuid.uuid4())
    video_path = f"temp/{uid}.webm"
    file.save(video_path)

    try:
        result = predict_video_emotion(video_path)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        if os.path.exists(video_path):
            os.remove(video_path)

@app.route("/predict-multimodal", methods=["POST"])
def predict_multimodal():
    file = request.files.get("media")
    if not file:
        return jsonify({"error": "No media file"}), 400

    os.makedirs("temp", exist_ok=True)
    uid = str(uuid.uuid4())
    _, ext = os.path.splitext(file.filename or "")
    ext = ext if ext else ".webm"
    media_path = f"temp/{uid}{ext}"
    file.save(media_path)

    try:
        audio_result = predict_voice_emotion(media_path)
        video_result = predict_video_emotion(media_path)
        if audio_result.get("category") == "None":
            audio_result = dict(EMPTY_MODALITY_RESPONSE)
        if video_result.get("category") == "None":
            video_result = dict(EMPTY_MODALITY_RESPONSE)
        combined_result = fuse_audio_video(audio_result, video_result)
        return jsonify({
            "audioResult": audio_result,
            "videoResult": video_result,
            "combinedResult": combined_result,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        if os.path.exists(media_path):
            os.remove(media_path)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
