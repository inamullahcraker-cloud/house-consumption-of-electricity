from pydantic import BaseModel, Field

class PredictionRequest(BaseModel):
    days: list[float] = Field(..., min_length=7, max_length=7, description="7-day power consumption history")

class PredictionResponse(BaseModel):
    predicted_power: float