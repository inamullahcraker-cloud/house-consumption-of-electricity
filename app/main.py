from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.schemas import PredictionRequest, PredictionResponse
from app.predictor import Power_predictor


BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "artifacts" / "consumption_model.pt"
SCALER_PATH = BASE_DIR / "artifacts" / "scaler.pkl"
DATA_PATH = BASE_DIR / "data" / "household_power_consumption.txt"

app = FastAPI(title="Power Consumption Forecast API")

app.mount(
    "/static",
    StaticFiles(directory=str(BASE_DIR / "static")),
    name="static",
)

templates = Jinja2Templates(
    directory=str(BASE_DIR / "templates")
)

predictor = Power_predictor(
    model_path=str(MODEL_PATH),
    scaler_path=str(SCALER_PATH),
    data_path=str(DATA_PATH),
)


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {"request": request},
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(data: PredictionRequest):
    try:
        predictions = predictor.predict(
            start_time=data.date_time,
            current_power=data.global_active_power,
        )

        return PredictionResponse(predictions=predictions)

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    except Exception as exc:
        print(f"Prediction error: {exc}")

        raise HTTPException(
            status_code=500,
            detail="Prediction failed. Check the terminal for details.",
        ) from exc