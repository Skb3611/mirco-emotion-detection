# Emotion Detection App — Docker Quickstart

## Prerequisites
- Docker Desktop installed: https://www.docker.com/products/docker-desktop/

## Quick Start

### 1. Build the image
```bash
docker build -t emotion-detection-app .
```

### 2. Run the container
```bash
docker run -p 5000:5000 emotion-detection-app
```

### 3. Open your browser
```
http://localhost:5000
```

## Files Needed
- `Dockerfile`
- `requirements.txt`
- `.dockerignore`
- `app.py`
- `src/` folder with model files
- `static/` folder with HTML

## Commands Reference
```bash
docker build -t emotion-detection-app .                    # Build image
docker run -p 5000:5000 emotion-detection-app             # Run container
docker build --no-cache -t emotion-detection-app .        # Force rebuild
```