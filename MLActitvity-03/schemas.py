from pydantic import BaseModel, Field, ConfigDict

class PredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    sepal_length_cm: float = Field(gt=0, le=20, description="Sepal length in cm")
    sepal_width_cm: float = Field(gt=0, le=20, description="Sepal width in cm")
    petal_length_cm: float = Field(gt=0, le=20, description="Petal length in cm")
    petal_width_cm: float = Field(gt=0, le=20, description="Petal width in cm")

class PredictionResponse(BaseModel):
    prediction: str
    class_id: int
    probabilities: dict[str, float]

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
