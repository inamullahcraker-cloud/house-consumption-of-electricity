
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.schemas import PredictionResponse,PredictionRequest
from app.predictor import Power_predictor
from pathlib import Path


app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")



BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "artifacts" / "consumption_model.pt"
SCALER_PATH = BASE_DIR / "artifacts" / "scaler.pkl" 

predictor = Power_predictor(str(MODEL_PATH), str(SCALER_PATH))

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict",response_model=PredictionResponse)
async def predict(data: PredictionRequest):

    prediction=predictor.predict(data.days)
    
    return PredictionResponse(predicted_power=prediction)