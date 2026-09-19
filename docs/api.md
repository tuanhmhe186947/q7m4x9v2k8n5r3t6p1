# REST API Reference

The pipeline includes a FastAPI-based server (`pig_behavior.api`) for online inference, tracking status monitoring, and real-time statistics streaming.

---

## 1. Launching the API Server

Launch using the CLI entrypoint:

```bash
pig-behavior-api --host 0.0.0.0 --port 8000
```

Or directly via Uvicorn:

```bash
uvicorn src.pig_behavior.api:app --host 0.0.0.0 --port 8000 --workers 1
```

Once running, interactive OpenAPI documentation is available at `http://localhost:8000/docs`.

---

## 2. API Endpoints

### Health & Status

#### `GET /`
Service root endpoint returning basic runtime metadata.

- **Response `200 OK`**:
  ```json
  {
    "service": "pig-behavior-api",
    "version": "0.1.0",
    "status": "ready"
  }
  ```

#### `GET /health`
Liveness probe checking GPU availability and model loader status.

- **Response `200 OK`**:
  ```json
  {
    "status": "healthy",
    "device": "cuda:0",
    "detector_loaded": true,
    "behavior_model_loaded": true
  }
  ```

---

### Behavior Recognition & Tracking

#### `POST /predict`
Run behavior classification on an uploaded batch of image crops or feature vectors.

- **Request**: `multipart/form-data` with `file` (image) or JSON body containing temporal feature tensors.
- **Response `200 OK`**:
  ```json
  {
    "track_id": 1,
    "predicted_class": "drinking",
    "confidence": 0.842,
    "probabilities": {
      "drinking": 0.842,
      "eating": 0.045,
      "standing": 0.071,
      "lying": 0.042
    }
  }
  ```

#### `POST /track`
Process a video segment or RTSP stream URL with online tracking.

- **Request Body**:
  ```json
  {
    "source": "rtsp://camera1/live",
    "mode": "realtime_fast",
    "confidence_threshold": 0.35
  }
  ```
- **Response `200 OK`**:
  ```json
  {
    "task_id": "stream_task_001",
    "status": "active",
    "active_tracks": 8
  }
  ```

#### `GET /api/stats`
Retrieve live aggregate statistics (activity budgets, total variation, detected behavior frequencies) across active tracks.

- **Response `200 OK`**:
  ```json
  {
    "frame_count": 1200,
    "active_pigs": 8,
    "behavior_distribution": {
      "lying": 0.62,
      "standing": 0.22,
      "eating": 0.11,
      "drinking": 0.05
    }
  }
  ```
