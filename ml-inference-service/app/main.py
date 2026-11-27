"""Breast Cancer prediction API (Day 63 -> Day 64 containerized service)."""

import os
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field, FiniteFloat

# Resolve paths relative to this file, not the current working directory.
# This makes the app work identically locally and inside the container.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = PROJECT_ROOT / "artifacts" / "breast_cancer_model.joblib"

MODEL_PATH_ENV = "MODEL_PATH"


class PredictionRequest(BaseModel):
    features: list[FiniteFloat] = Field(
        min_length=30,
        max_length=30,
        description="Exactly 30 breast cancer dataset features",
    )


class PredictionResponse(BaseModel):
    predicted_class: int
    predicted_label: str
    probabilities: dict[str, float]
    model_version: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    model_path = Path(os.environ.get(MODEL_PATH_ENV, MODEL_PATH))

    if not model_path.exists():
        raise RuntimeError(f"Model artifact not found: {model_path}")

    artifact = joblib.load(model_path)

    app.state.model = artifact["model"]
    app.state.feature_names = artifact["feature_names"]
    app.state.class_names = artifact["class_names"]
    app.state.model_version = artifact["model_version"]

    yield


app = FastAPI(
    title="Breast Cancer Prediction API",
    description="ML inference service using FastAPI",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/")
def root():
    return {"message": "ML Prediction API is running"}


@app.get("/health")
def health(request: Request):
    return {
        "status": "healthy",
        "model_loaded": hasattr(request.app.state, "model"),
    }


@app.get("/metadata")
def metadata(request: Request):
    return {
        "model_version": request.app.state.model_version,
        "number_of_features": len(request.app.state.feature_names),
        "feature_names": request.app.state.feature_names,
        "class_names": request.app.state.class_names,
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(payload: PredictionRequest, request: Request):
    try:
        features = np.asarray(payload.features, dtype=np.float64).reshape(1, -1)

        model = request.app.state.model

        predicted_class = int(model.predict(features)[0])

        probabilities_array = model.predict_proba(features)[0]

        class_names = request.app.state.class_names
        model_classes = model.named_steps["classifier"].classes_

        probabilities = {
            class_names[int(class_id)]: float(probability)
            for class_id, probability in zip(
                model_classes,
                probabilities_array,
                strict=True,
            )
        }

        return PredictionResponse(
            predicted_class=predicted_class,
            predicted_label=class_names[predicted_class],
            probabilities=probabilities,
            model_version=request.app.state.model_version,
        )

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid prediction input: {exc}",
        ) from exc
