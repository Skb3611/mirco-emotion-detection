from flask import Flask, render_template, request, jsonify ,send_from_directory
import cv2
import numpy as np
from src.detector import predict_emotion
from flask_cors import CORS


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


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
