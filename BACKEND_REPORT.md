# Backend Report (Flask + ML Inference)

## 1) What this backend does

This folder (`mirco-emotion-detection/`) runs a Flask server that performs emotion inference from:

- **Image / single frame** (video frame classification via the same FER pipeline)
- **Video clips** (frame sampling + face detection + FER aggregation)
- **Audio clips** (speech emotion classification using a pretrained transformer)
- **Multimodal (audio + video)** where both outputs are fused into one calibrated result

The backend returns structured JSON including:

- `category` (Comfortable / Uncomfortable)
- `emotion` (wheel-base base emotion)
- `subEmotion` (confidence-threshold label)
- `confidence` (numeric score for the top wheel base)
- `wheelBaseListSorted` (ranked list of all wheel bases)

---

## 2) Tech stack and runtime requirements

### 2.1 Core server

- `Flask` + `flask-cors`
- Runs as `python app.py` on port `5000`

### 2.2 ML / CV dependencies

From `requirements.txt`:

- `opencv-python` (Haar cascade face detection, frame extraction)
- `tensorflow` / `keras` (FER CNN model inference)
- `torch` + `transformers` (Wav2Vec2 audio classification)
- `librosa` (audio decoding/resampling)
- `numpy` (score aggregation and numeric operations)

### 2.3 System dependencies (Docker)

From `Dockerfile`:

- `ffmpeg`
- `libgl1`, `libglib2.0-0` (OpenCV runtime support)

---

## 3) API surface (Flask endpoints)

File: `mirco-emotion-detection/app.py`

### 3.1 `POST /predict` (image)

- **Input**: multipart form field `image` (raw bytes)
- **Processing**:
  - Decode bytes with OpenCV (`cv2.imdecode`)
  - Run `predict_emotion(frame)` from `src/detector.py`
- **Output**: FER + wheel mapping response

### 3.2 `POST /predict-audio` (audio only)

- **Input**: multipart form field `audio`
- **Processing**:
  - Save upload to `temp/<uuid>.<ext>.webm` (backend accepts multiple formats via `librosa`)
  - Call `predict_voice_emotion(media_path)` from `src/voice_detector.py`
  - Delete temp file
- **Output**: wheel mapping response for the voice model

### 3.3 `POST /predict-video` (video only)

- **Input**: multipart form field `video`
- **Processing**:
  - Save upload to `temp/<uuid>.webm`
  - Call `predict_video_emotion(video_path)` from `src/detector.py`
  - Delete temp file
- **Output**: aggregated wheel mapping response + `videoMeta` + `frameResults`

### 3.4 `POST /predict-multimodal` (audio + video + fusion)

- **Input**: multipart form field `media`
- **Processing**:
  - Save upload to `temp/<uuid>.<ext>`
  - Run:
    - `audio_result = predict_voice_emotion(media_path)`
    - `video_result = predict_video_emotion(media_path)`
  - If a modality returns `category == "None"`, replace it with `EMPTY_MODALITY_RESPONSE`
  - Call `fuse_audio_video(audio_result, video_result)` from `src/fusion.py`
  - Return:
    - `audioResult`
    - `videoResult`
    - `combinedResult`
- **Output**: fused wheel-base response + `fusionMeta`

---

## 4) Core preprocessing and inference pipelines

### 4.1 FER visual pipeline (image + video)

File: `src/detector.py`

#### 4.1.1 Face detection

Uses OpenCV Haar cascade:

- `src/haarcascade_frontalface_default.xml`
- `face_cascade.detectMultiScale(gray, 1.3, 5)`

#### 4.1.2 CNN architecture (FER)

Model loaded from:

- `src/models/fer.h5`

Architecture created in code:

- Conv2D(32) -> Conv2D(64) -> MaxPool -> Dropout(0.25)
- Conv2D(128) -> MaxPool -> Conv2D(128) -> MaxPool -> Dropout(0.25)
- Flatten -> Dense(1024) -> Dropout(0.5)
- Dense(7, softmax)

#### 4.1.3 Frame preprocessing

For each detected face:

- Convert ROI to grayscale
- Resize to **48x48**
- Normalize by dividing by 255.0
- Reshape to `(1, 48, 48, 1)` for CNN input

#### 4.1.4 Wheel mapping (base emotions + category)

FER labels (fixed order in `FER13_LABELS`):

- `["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]`

Mapping to wheel base emotions:

- Angry -> Angry (Uncomfortable)
- Disgust -> Embarrassed (Uncomfortable)
- Fear -> Scared (Uncomfortable)
- Happy -> Happy (Comfortable)
- Neutral -> Neutral (Comfortable)
- Sad -> Sad (Uncomfortable)
- Surprise -> Happy (Comfortable) (note: “nearest” behavior is commented in code)

Category mapping:

- Uncomfortable: Angry, Disgust, Fear, Sad
- Comfortable: Happy, Neutral, Surprise

Wheel base ordering (`WHEEL_ORDER`) is fixed:

1. Uncomfortable: Sad, Scared, Angry, Embarrassed
2. Comfortable: Happy, Loved, Confident, Neutral

#### 4.1.5 Supported and unsupported emotions

The backend does **not** directly classify every wheel emotion as its own trained class. Some emotions are:
- **directly predicted** by the model,
- **mapped/inferred** into a wheel-base emotion,
- or **not directly modeled** and therefore usually remain zero unless introduced by fusion design.

| Emotion / Label | Visual model support | Audio model support | Backend status | Notes |
| --- | --- | --- | --- | --- |
| Angry | Direct FER label | Direct voice label | Supported directly | Native class in both modalities |
| Sad | Direct FER label | Direct voice label | Supported directly | Native class in both modalities |
| Happy | Direct FER label | Direct voice label | Supported directly | Native class in both modalities |
| Neutral | Direct FER label | Direct voice label (`neutral` / `calm`) | Supported directly | `calm` is merged into `Neutral` in audio |
| Fear / Scared | FER predicts `Fear`, mapped to `Scared` | Voice predicts `fearful` / `fear`, mapped to `Scared` | Supported via mapping | Wheel base is `Scared`, not raw `Fear` |
| Disgust / Embarrassed | FER predicts `Disgust`, mapped to `Embarrassed` | Voice predicts `disgust`, mapped to `Embarrassed` | Supported via mapping | `Embarrassed` is inferred from `Disgust` |
| Surprise | Direct FER label, mapped to `Happy` | Direct voice label (`surprised` / `surprise`), mapped to `Happy` | Supported via mapping | Not kept as its own final wheel base |
| Loved | Not a direct class | Not a direct class | Not directly supported | Exists in wheel structure, but no direct model output maps to it |
| Confident | Not a direct class | Not a direct class | Not directly supported | Exists in wheel structure, but no direct model output maps to it |
| No Face | Returned by visual fallback | Not applicable | Supported as fallback state | Used when no face is detected in image/video |
| Silent audio | Not applicable | Returned by audio fallback | Supported as fallback state | Used when waveform amplitude is near zero |

Important interpretation:
- **Supported directly** means the underlying model has a native class close to that emotion.
- **Supported via mapping** means the backend derives the wheel-base emotion from another raw class.
- **Not directly supported** means the emotion exists in the wheel output schema but is not produced by a dedicated class in either underlying model.

#### 4.1.5 Sub-emotion logic

Sub-emotion thresholds per base emotion (`WHEEL_SUB_MAP`) are identical across modalities.
Active `subEmotion` is selected by:

- confidence >= 75.0 -> first sub label
- confidence >= 45.0 -> second sub label
- confidence >= 0.0 -> third sub label
- else -> `"None"`

---

### 4.2 Video clip inference

Also in `src/detector.py` via `predict_video_emotion(video_path, frame_step=1)`

Key steps:

- Open video: `cv2.VideoCapture(video_path)`
- Sample frames: process only when `frame_count % frame_step == 0`
- Detect faces per sampled frame
- If no face:
  - append a frame entry with `emotion="No Face"` and `category/confidence` neutralized via response logic
- If faces exist:
  - pick the largest face by area (`max(faces, key=lambda f: f[2] * f[3])`)
  - preprocess and predict with FER CNN
  - sum predictions (`preds_sum`) and count valid faces (`valid_face_frames`)

Aggregation:

- if `valid_face_frames == 0` -> return `empty_emotion_response()` with `videoMeta` + `frameResults`
- else:
  - `avg_preds = preds_sum / valid_face_frames`
  - build final wheel response from averaged FER probabilities

`videoMeta` includes:

- `totalFramesRead`
- `frameStep`
- `sampledFrames`
- `validFaceFrames`

---

### 4.3 Voice / audio pipeline

File: `src/voice_detector.py`

#### 4.3.1 Model

Uses a pretrained HuggingFace transformer:

- `Dpngtm/wav2vec2-emotion-recognition`

The model and processor are loaded via:

- `Wav2Vec2Processor.from_pretrained(...)`
- `AutoModelForAudioClassification.from_pretrained(...)`

#### 4.3.2 Audio decoding and preprocessing

Audio decoding:

- `librosa.load(audio_path, sr=16000, mono=True)`

Silent audio guard:

- if `np.max(np.abs(audio)) < 0.01` -> returns an empty voice response

Wav2Vec2 preprocessing:

- tokenizer/processor produces model inputs:
  - `padding=True`
  - `max_length=160000`
  - `truncation=True`
  - `return_tensors="pt"`

Inference:

- forward pass `outputs = _model(inputs.input_values)`
- convert logits to probabilities using softmax

#### 4.3.3 Mapping to wheel bases and sub-emotion

Audio labels from model config (`VOICE_LABELS`) are mapped to wheel bases:

- neutral/calm -> Neutral
- happy -> Happy
- sad -> Sad
- angry -> Angry
- fearful/fear -> Scared
- disgust -> Embarrassed
- surprised/surprise -> Happy

Then the same:

- `WHEEL_SUB_MAP` thresholds choose `subEmotion`
- `WHEEL_ORDER` provides consistent ranked wheel output

---

## 5) Multimodal fusion method

File: `src/fusion.py`

Fusion inputs:

- `audio_result` (wheelBaseList + category)
- `video_result` (wheelBaseList + category + videoMeta)

Wheel scoring approach:

- Convert each modality into a dictionary of wheel-base confidence scores from `wheelBaseList`.

### 5.1 Modality availability and fallback

If only one modality is available:

- audio available only -> return audio result, with:
  - `fusionMeta.audioQuality = _audio_quality(...)`
  - `fusionMeta.videoQuality = 0.0`
  - `weightsUsed: audio=1.0, video=0.0`
- video available only -> return video result, with:
  - `fusionMeta.audioQuality = 0.0`
  - `fusionMeta.videoQuality = _video_quality(...)`
  - `weightsUsed: audio=0.0, video=1.0`

If neither modality is available:

- return an all-`"None"` response with zero confidences

### 5.2 Quality-weighting (how weights are computed)

Base blend ratio:

- video base weight = 0.6
- audio base weight = 0.4

Effective weights:

- `audio_weight = 0.4 * audio_q`
- `video_weight = 0.6 * video_q`
- normalize so `audio_weight + video_weight = 1`

If both quality scores yield `audio_weight + video_weight == 0`:

- fallback weights are used (audio 0.4, video 0.6) before normalization.

Quality functions:

- Audio quality (`_audio_quality`):
  - uses only positive wheel scores from `wheelBaseList`
  - computes `max_prob` after normalizing scores over wheel bases
  - random baseline for 8 classes = 0.125
  - `quality = (max_prob - 0.125) / (1 - 0.125)`, clipped to [0, 1]
- Video quality (`_video_quality`):
  - `valid / sampled`, where:
    - `sampled = videoMeta["sampledFrames"]`
    - `valid = videoMeta["validFaceFrames"]`
  - clipped to [0, 1]

### 5.3 Final combined output

For each wheel base in `WHEEL_ORDER`, compute:

- `combined_score[base] = audio_weight * audio_score[base] + video_weight * video_score[base]`

Then build `wheel_base_list`:

- assign `activeSub = _get_active_sub(base, conf)`
- include `allSubs` for UI

Top emotion:

- `top = max(wheel_base_list, key=confidence)`

Return includes:

- `fusionMeta.audioQuality`, `fusionMeta.videoQuality`
- `fusionMeta.weightsUsed.audio`, `fusionMeta.weightsUsed.video`

---

## 6) Running the backend with Docker

###  Docker

```bash
docker build -t emotion-detection-backend .
docker run -p 5000:5000 emotion-detection-backend
```

