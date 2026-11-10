import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict

model = joblib.load("breast_cancer_model.joblib")
app = FastAPI()

LABEL_MAP = {0: "malignant", 1: "benign"}


class PatientData(BaseModel):
    model_config = ConfigDict(alias_generator=lambda name: name.replace("_", " "))

    mean_radius: float
    mean_texture: float
    mean_perimeter: float
    mean_area: float
    mean_smoothness: float
    mean_compactness: float
    mean_concavity: float
    mean_concave_points: float
    mean_symmetry: float
    mean_fractal_dimension: float

    radius_error: float
    texture_error: float
    perimeter_error: float
    area_error: float
    smoothness_error: float
    compactness_error: float
    concavity_error: float
    concave_points_error: float
    symmetry_error: float
    fractal_dimension_error: float

    worst_radius: float
    worst_texture: float
    worst_perimeter: float
    worst_area: float
    worst_smoothness: float
    worst_compactness: float
    worst_concavity: float
    worst_concave_points: float
    worst_symmetry: float
    worst_fractal_dimension: float


@app.get("/")
def home():
    return {"message": "Breast Cancer API is running"}


@app.post("/predict")
def predict(data: PatientData):
    try:
        input_df = pd.DataFrame([data.model_dump(by_alias=True)])
        prediction = int(model.predict(input_df)[0])
        probabilities = model.predict_proba(input_df)[0]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

    return {
        "prediction": LABEL_MAP[prediction],
        "confidence": round(float(probabilities[prediction]) * 100, 2),
    }