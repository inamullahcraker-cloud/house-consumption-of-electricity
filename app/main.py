import joblib
import numpy as np
import torch
import torch.nn as nn
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# 1. Define Model Architecture (Must match your saved model)
class PowerLSTM(nn.Module):
    def __init__(self,num_in=1,num_out=1):
        super().__init__()
        self.lstm=nn.LSTM(num_in,5,1,batch_first=True)
        self.fc1=nn.Linear(5,num_out)
        self.sigmoid=nn.Sigmoid()

    def forward(self,x):
        x,_=self.lstm(x)
        x=x[:,-1,:]
        x=self.fc1(x)
        x=self.sigmoid(x)
        return x  # Predict next timestep

# 2. Load Model & Scaler on Startup
model = PowerLSTM(1,1)
name="/home/inamullah/jupyter_env/deeplearning/RNN/house-holding/app/consumption_model.pt"
device = torch.device('cpu')
model = PowerLSTM()

# Load weights and set to eval mode
state_dict = torch.load(name, map_location=device, weights_only=True)
model.load_state_dict(state_dict)
model.to(device)
model.eval()

# Load saved MinMaxScaler (save via `joblib.dump(scaler, 'scaler.pkl')` during training)
scaler = joblib.load('scaler.pkl')

class PredictionInput(BaseModel):
    days: list[float]  # Expecting 7 values

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict")
async def predict(data: PredictionInput):
    # Convert input list to numpy array: shape (7, 1)
    input_seq = np.array(data.days).reshape(-1, 1)

    # Scale inputs using fitted scaler
    scaled_seq = scaler.transform(input_seq)

    # Reshape to 3D Tensor: (batch_size=1, sequence_length=7, features=1)
    tensor_input = torch.tensor(scaled_seq, dtype=torch.float32).unsqueeze(0).to(device)

    # Perform inference
    with torch.inference_mode():
        scaled_prediction = model(tensor_input).cpu().numpy()

    # Inverse transform prediction to original scale (kW)
    original_prediction = scaler.inverse_transform(scaled_prediction)
    predicted_val = float(original_prediction[0][0])

    return {"predicted_power": round(predicted_val, 2)}