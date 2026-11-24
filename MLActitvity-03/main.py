from fastapi import FastAPI
from app.model import model, CLASS_NAMES, predict
from app.schemas import PredictionRequest, PredictionResponse, HealthResponse

app = FastAPI(
    title="Iris ML Prediction API",
    description="REST API serving a pre-trained Iris classifier with input validation.",
    version="1.0.0",
)

@app.get("/", tags=["General"])
def root():
    return {"message": "Iris ML API is running", "docs": "/docs"}

@app.get("/health", response_model=HealthResponse, tags=["Monitoring"])
def health():
    return {"status": "ok", "model_loaded": model is not None}

@app.get("/model-info", tags=["Model"])
def model_info():
    return {
        "model_name": "StandardScaler + LogisticRegression",
        "model_version": "1.0.0",
        "features": [
            "sepal_length_cm", "sepal_width_cm",
            "petal_length_cm", "petal_width_cm"
        ],
        "classes": [str(c) for c in CLASS_NAMES]
    }

@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict_endpoint(payload: PredictionRequest):
    features = [
        payload.sepal_length_cm, payload.sepal_width_cm,
        payload.petal_length_cm, payload.petal_width_cm
    ]
    class_id, label, probabilities = predict(features)
    return PredictionResponse(
        prediction=str(label),
        class_id=class_id,
        probabilities=probabilities
    )
