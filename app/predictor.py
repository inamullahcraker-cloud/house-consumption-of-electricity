from torch import nn,torch
import numpy as np
from app.model import PowerGLU
import joblib

class Power_predictor():
    def __init__(self,model_path:str,scaler_path:str):
        self.device=torch.device("cpu")
        self.scaler=joblib.load(scaler_path)
        self.model=PowerGLU()
        self.state_dict=torch.load(model_path,map_location=self.device,weights_only=True)
        self.model.load_state_dict(self.state_dict)
        self.model.float()
        self.model.to(self.device)
        self.model.eval()

    def predict(self,raw_sequences:list[float]):
        input_array=np.array(raw_sequences).reshape(-1,1)
        input_scaled=self.scaler.transform(input_array)
        input_tensor=torch.tensor(input_scaled,dtype=torch.float32).unsqueeze(0)

        with torch.inference_mode():
            output=self.model(input_tensor).cpu().numpy()
        output_original=self.scaler.inverse_transform(output)
        return round(float(output_original[0][0]),2)

        
    
