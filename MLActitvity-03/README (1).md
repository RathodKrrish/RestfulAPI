# Scalable RESTful API Architecture for ML Model Deployment

A production-style starter project that serves a pre-trained scikit-learn model through FastAPI, validates request payloads using Pydantic, containerizes the service with Docker, and includes integration tests.

## Architecture
- `app/main.py`: FastAPI application, health and prediction endpoints
- `app/schemas.py`: request/response validation
- `app/model.py`: load model and make prediction
- `app/train_model.py`: train and save a demo model (Iris classification)
- `tests/test_api.py`: automated integration tests using FastAPI TestClient
- `notebooks/api_testing_walkthrough.ipynb`: API test walkthrough notebook

## Endpoints
- `GET /health` - service health
- `GET /model-info` - model metadata
- `POST /predict` - accepts four numeric Iris measurements and returns class + probabilities

## Run locally
```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
# source .venv/bin/activate
pip install -r requirements.txt
python -m app.train_model
uvicorn app.main:app --reload
```
Open API docs at http://127.0.0.1:8000/docs.

## Docker
Build and run:
```bash
docker build -t iris-ml-api .
docker run --rm -p 8000:8000 iris-ml-api
```
Then browse `http://127.0.0.1:8000/docs`.

## Tests
```bash
python -m app.train_model
pytest -v
```
Tests use a temporary trained model if model artifact isn't present, and verify health, model info, valid prediction, and validation errors.

## Sample request
```json
{
  "sepal_length_cm": 5.1,
  "sepal_width_cm": 3.5,
  "petal_length_cm": 1.4,
  "petal_width_cm": 0.2
}
```

## Notes
The model is a small Iris classifier for demonstration. Replace it with your own model and schema for a real use case. For deployment, add auth, logging/monitoring, rate limiting, secrets management, model versioning, and CI/CD.
