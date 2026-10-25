from datetime import datetime

from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    date_time: datetime = Field(
        ...,
        description="Date and time of the current power reading",
    )

    global_active_power: float = Field(
        ...,
        ge=0,
        description="Current active power consumption in kW",
    )


class ForecastPoint(BaseModel):
    date_time: datetime
    predicted_power: float


class PredictionResponse(BaseModel):
    predictions: list[ForecastPoint]