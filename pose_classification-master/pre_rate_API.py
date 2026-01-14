# pred_rate_api.py
import torch
import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List
import json
import os
import logging
import traceback


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s'
)
logger = logging.getLogger(__name__)


app = FastAPI()

class KeyframeData(BaseModel):
    keypoint: List[List[List[float]]]  
    bbox: List[List[float]]
    idx: List[int]

class FCNN(torch.nn.Module):
    def __init__(self, input_dim=46, num_classes=10):
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, num_classes)
        )

    def forward(self, x):
        return self.layers(x)

# model init
MODEL = None
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

@app.on_event("startup")
def load_model():
    global MODEL
    model_path = "result/FCNN/no3p/dean/best_model.pth"
    MODEL = FCNN().to(DEVICE)
    MODEL.load_state_dict(torch.load(model_path, map_location=DEVICE))
    MODEL.eval()

def preprocess(data):
    size = 1500
    
    empty = []
    
    keypoints = data["keypoints"]
    bbox =  data["bbox"]
    idx = data["idx"]
    
    
    
    
    for i in range(len(keypoints)):
        bbox_up_x, bbox_up_y, bbox_down_x, bbox_down_y= bbox[i]

        center_x = (bbox_down_x + bbox_up_x)/2
        center_y = (bbox_down_y + bbox_up_y)/2

        diff_x = size/2 - center_x 
        diff_y = size/2 - center_y
    
        temp = [] 
        for j in range(23):
            x = keypoints[i][j][0] + diff_x
            y = keypoints[i][j][1] + diff_y
            temp.append(x)
            temp.append(y)
            
        empty.append(temp)
        
        
    return np.vstack(empty)



@app.post("/predict")
async def predict(input_data: dict):  
    try:
        
        
    
        logger.info("[INFO] preprocess")
        processed = preprocess(input_data)
        
        logger.info("[INFO] predict start")
        input_tensor = torch.tensor(processed, dtype=torch.float32).to(DEVICE)
        
        with torch.no_grad():
            outputs = MODEL(input_tensor)
            probs = torch.softmax(outputs, dim=1)
        
        class_results = {}
        for i, prob in enumerate(probs):
            class_idx = int(torch.argmax(prob).item())
            confidence = prob[class_idx].item()
            logging.info(f"[INFO] pose: {class_idx}")
            logging.info(f"[INFO] confidence: {confidence}")
            logging.info(f"[INFO] idx: {input_data['idx'][i]}")
            # for each class save the most
            if class_idx not in class_results or confidence > class_results[class_idx]["confidence"]:
                class_results[class_idx] = {
                    "idx": input_data["idx"][i],
                    "keypoints": input_data["keypoints"][i],
                    "confidence": confidence
                }
        logger.info("[INFO] predict finish")
        return {"result": class_results}
    
    except Exception as e:
        tb_str = traceback.format_exc()
        logger.error(f"fail: {tb_str}")
        return {"error": str(e), "traceback": tb_str}
        

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)

