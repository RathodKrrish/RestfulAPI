from pathlib import Path
import joblib
import numpy as np

MODEL_PATH = Path(__file__).resolve().parent / "iris_model.joblib"

def load_model():
    if not MODEL_PATH.exists():
        # Keep startup self-contained in fresh clones.
        from app.train_model import train_and_save
        train_and_save()
    return joblib.load(MODEL_PATH)

model = load_model()
CLASS_NAMES = ["setosa", "versicolor", "virginica"]

def predict(features: list[float]) -> tuple[int, str, dict[str, float]]:
    array = np.asarray([features], dtype=float)
    class_id = int(model.predict(array)[0])
    label = CLASS_NAMES[class_id]
    probs = model.predict_proba(array)[0]
    return class_id, label, {str(name): float(prob) for name, prob in zip(CLASS_NAMES, probs)}
