from fastapi import FastAPI, UploadFile, File
from mmpose.apis import MMPoseInferencer
import shutil
import os
import uuid
import numpy as np
from fastapi import HTTPException
import logging
from pydantic import BaseModel


app = FastAPI()
_inferencer = None


class FilePathRequest(BaseModel):
    file_path: str


def convert(obj):
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, list):
        return [convert(i) for i in obj]
    if isinstance(obj, dict):
        return {k: convert(v) for k, v in obj.items()}
    return obj



@app.on_event("startup")
def load_model():
    global _inferencer
    _inferencer = MMPoseInferencer(
        pose2d="configs/body_2d_keypoint/rtmpose/coco/final_config.py",
        
        pose2d_weights='weight.pth'
    )


@app.post("/predict_obj")
async def predict(request: FilePathRequest):
    # pred
    logging.info("[INFO] predict_obj start")
    file_path = request.file_path
    result_generator = _inferencer(file_path, show=False)
    result = next(result_generator)
    
    safe_result = convert(result)
    logging.info("[INFO] predict_obj finish")
    return safe_result


@app.post("/predict_batch")
async def predict_batch(dir_path: str):
    try:
        logging.info("[INFO] predict_batch start")
        file_paths = []
        i = 0
        while True:
            path = os.path.join(dir_path, f"{i}.jpg")
            if not os.path.exists(path):
                break
            file_paths.append(path)
            i += 1
            
        if not file_paths:
            raise HTTPException(400, "no image")
            
        
        result_generator = _inferencer(file_paths, show=False)
        
        
        keypoints_results = []
        bbox_results = []
        
        #save result
        for result in result_generator:
            temp1 = [convert(pred) for pred in result["predictions"][0][0]["keypoints"]]
            temp2 = [convert(pred) for pred in result["predictions"][0][0]["bbox"][0]]
            #print(f"keypoints:{temp1}")
            logging.debug(f"bbox: {temp2}")
            keypoints_results.append(temp1)
            bbox_results.append(temp2)
        
        
        batch_results={"keypoints": keypoints_results, "bbox": bbox_results}
        
        logging.info("[INFO] predict_batch finish")
        #a list contain all of images 23 keypoints
        return {"results": batch_results}
        
    except Exception as e:
        raise HTTPException(500, detail=str(e))
    
    
    
"""
result = {
    "predictions": [
        [  
            {
                "keypoints": [[x,y],[x,y]...],  
                "bbox": [x1, y1, x2, y2]        
            }
        ]
    ]
}
    
    
    
"""
    
    
    
    
    
    


