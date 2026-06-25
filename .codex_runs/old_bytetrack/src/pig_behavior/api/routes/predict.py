"""Prediction endpoints."""

from __future__ import annotations

import json
import math
import re
from typing import Annotated

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from PIL import UnidentifiedImageError

from pig_behavior.api.dependencies import (
    PTModelService,
    _confidence_from_env,
    get_model_service,
)
from pig_behavior.api.schemas import (
    DetectionResponse,
    PredictionResponse,
    PredictionScore,
)
from pig_behavior.config import TABULAR_FEATURES
from pig_behavior.services.pt_inference import PTPrediction

router = APIRouter()


@router.post("/predict", response_model=PredictionResponse)
async def predict_upload(
    image: Annotated[
        UploadFile,
        File(description="Pig image file, for example JPEG or PNG."),
    ],
    bbox: Annotated[
        str | None,
        Form(
            description=(
                "Optional bounding box as JSON array or comma-separated "
                "x1,y1,x2,y2 values."
            ),
        ),
    ] = None,
    tabular: Annotated[
        str | None,
        Form(
            description=(
                "Optional six tabular values as JSON array or comma-separated "
                "values in API metadata order."
            ),
        ),
    ] = None,
    model_service=Depends(get_model_service),
) -> PredictionResponse:
    """Run an image prediction."""
    content_type = image.content_type or ""
    if content_type and not content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Upload must be an image.")

    image_bytes = await image.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Uploaded image is empty.")

    parsed_bbox = _parse_float_tuple(
        bbox,
        expected_count=4,
        field_name="bbox",
    )
    parsed_tabular = _parse_float_list(
        tabular,
        expected_count=len(TABULAR_FEATURES),
        field_name="tabular",
    )

    try:
        if isinstance(model_service, PTModelService):
            pt_prediction = model_service.predict_bytes(
                image_bytes,
                bbox=parsed_bbox,
                confidence_threshold=_confidence_from_env(),
            )
            return _pt_prediction_response(
                image.filename,
                model_service,
                pt_prediction,
                parsed_bbox,
            )

        scores, latency_ms = model_service.predict_image(
            image_bytes,
            parsed_bbox,
            parsed_tabular,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ImportError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except UnidentifiedImageError as exc:
        raise HTTPException(
            status_code=400,
            detail="Uploaded file is not a valid image.",
        ) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    sorted_scores = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    predicted_label, confidence = sorted_scores[0]
    model_path = model_service.loaded_model_path or model_service.model_path

    return PredictionResponse(
        filename=image.filename,
        backend="tflite",
        task="classify",
        predicted_label=predicted_label,
        confidence=confidence,
        latency_ms=latency_ms,
        scores=[
            PredictionScore(label=label, confidence=score)
            for label, score in sorted_scores
        ],
        detections=[],
        bbox=list(parsed_bbox) if parsed_bbox is not None else None,
        tabular=_tabular_dict(parsed_tabular),
        model_path=str(model_path),
    )


def _parse_float_tuple(
    raw: str | None,
    *,
    expected_count: int,
    field_name: str,
) -> tuple[float, ...] | None:
    values = _parse_float_list(
        raw,
        expected_count=expected_count,
        field_name=field_name,
    )
    return tuple(values) if values is not None else None


def _parse_float_list(
    raw: str | None,
    *,
    expected_count: int,
    field_name: str,
) -> list[float] | None:
    if raw is None or not raw.strip():
        return None

    try:
        if raw.strip().startswith("["):
            parsed = json.loads(raw)
            if not isinstance(parsed, list):
                raise ValueError
            values = [float(value) for value in parsed]
        else:
            tokens = [token for token in re.split(r"[\s,]+", raw.strip()) if token]
            values = [float(token) for token in tokens]
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise HTTPException(
            status_code=422,
            detail=f"{field_name} must contain numeric values.",
        ) from exc

    if len(values) != expected_count:
        raise HTTPException(
            status_code=422,
            detail=(
                f"{field_name} must contain {expected_count} values, "
                f"got {len(values)}."
            ),
        )
    if not all(math.isfinite(value) for value in values):
        raise HTTPException(
            status_code=422,
            detail=f"{field_name} values must be finite numbers.",
        )

    return values


def _tabular_dict(values: list[float] | None) -> dict[str, float] | None:
    if values is None:
        return None
    return dict(zip(TABULAR_FEATURES, values, strict=True))


def _pt_prediction_response(
    filename: str | None,
    service: PTModelService,
    prediction: PTPrediction,
    bbox: tuple[float, ...] | None,
) -> PredictionResponse:
    scores = sorted(prediction.scores.items(), key=lambda item: item[1], reverse=True)
    return PredictionResponse(
        filename=filename,
        backend="pt",
        task=prediction.task,
        predicted_label=prediction.predicted_label or "",
        confidence=prediction.confidence or 0.0,
        latency_ms=prediction.latency_ms,
        scores=[
            PredictionScore(label=label, confidence=confidence)
            for label, confidence in scores
        ],
        detections=[
            DetectionResponse(
                label=detection.label,
                confidence=detection.confidence,
                bbox_xyxy=detection.bbox_xyxy,
                class_id=detection.class_id,
            )
            for detection in prediction.detections
        ],
        bbox=list(bbox) if bbox is not None else None,
        tabular=None,
        model_path=str(service.model_path),
    )
