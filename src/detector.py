import cv2
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D

# Build the SAME model architecture used in the repo
model = Sequential()

model.add(Conv2D(32, kernel_size=(3, 3), activation='relu',
                 input_shape=(48, 48, 1)))
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

# LOAD WEIGHTS (this is the key)
model.load_weights("src/model.h5")

# Face detector
face_cascade = cv2.CascadeClassifier(
    "src/haarcascade_frontalface_default.xml"
)

labels = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]

# Configuration for sub-emotions based on confidence thresholds
# Format: "Emotion": [(minimum_confidence, "Sub-emotion"), ...]
# Note: Thresholds MUST be listed in descending order.
SUB_EMOTION_MAP = {
    "Happy": [
        (80.0, "Excited"),
        (60.0, "Joyful"),
        (0.0, "Content")
    ],
    "Sad": [
        (80.0, "Depressed"),
        (0.0, "Disappointed")
    ],
    "Angry": [
        (80.0, "Furious"),
        (0.0, "Frustrated")
    ],
    "Neutral": [
        (60.0, "Focused"),
        (0.0, "Calm")
    ],
    "Fear": [
        (0.0, "Anxious")
    ],
    "Surprise": [
        (0.0, "Amazed")
    ],
    "Disgust": [
        (0.0, "Repulsed")
    ]
}

def get_sub_emotion(emotion: str, confidence: float) -> str:
    """Classify sub-emotion based on primary emotion and confidence score."""
    thresholds = SUB_EMOTION_MAP.get(emotion, [])
    
    for threshold, sub_emotion in thresholds:
        if confidence >= threshold:
            return sub_emotion
            
    return "None"


def predict_emotion(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        roi = gray[y:y+h, x:x+w]
        roi = cv2.resize(roi, (48, 48))
        roi = roi / 255.0
        roi = roi.reshape(1, 48, 48, 1)

        preds = model.predict(roi, verbose=0)
        max_index = np.argmax(preds)
        emotion = labels[max_index]
        confidence = float(preds[0][max_index]) * 100
        
        sub_emotion = get_sub_emotion(emotion, confidence)
        
        return {
            "emotion": emotion,
            "subEmotion": sub_emotion,
            "confidence": round(confidence, 2)
        }

    return {
        "emotion": "No Face",
        "subEmotion": "None",
        "confidence": 0.0
    }
