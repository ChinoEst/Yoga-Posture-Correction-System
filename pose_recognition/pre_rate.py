import torch
import numpy as np
import argparse
import os
import pandas as pd
import json
from collections import defaultdict

# 定義模型架構
class FCNN(torch.nn.Module):
    def __init__(self, input_dim, num_classes):
        super(FCNN, self).__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, num_classes),
        )

    def forward(self, x):
        x = self.layers(x)
        return x.squeeze(1)

# 載入模型
def load_model(model_path, input_dim, num_classes, device='cpu'):
    model = FCNN(input_dim, num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

# 推理
def infer(model, test_data, device='cpu'):
    test_data = torch.tensor(test_data, dtype=torch.float32).to(device)
    with torch.no_grad():
        output = model(test_data)
    return output

# 資料處理
def process(path, save_list):
    size = 1500
    kp_num = 23
    empty = []

    for JSON in save_list:
        with open(f'{path}/{JSON}.json', 'r') as file:
            json_data = json.load(file)
            j_data = json_data[0]

            bbox = j_data["bbox"][0]
            bbox_up_x = bbox[0]
            bbox_up_y = bbox[1]
            bbox_down_x = bbox[2]
            bbox_down_y = bbox[3]

            center_x = (bbox_down_x + bbox_up_x)/2
            center_y = (bbox_down_y + bbox_up_y)/2

            diff_x = size/2 - center_x 
            diff_y = size/2 - center_y

            temp = []    
            for j in range(kp_num):
                x = j_data["keypoints"][j][0] + diff_x
                y = j_data["keypoints"][j][1] + diff_y
                temp.append(x)
                temp.append(y)

            empty.append(temp)

    return np.vstack(empty), save_list

# 主程式
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="data")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--name", type=str, required=True)
    parser.add_argument("--who", type=str, required=True)
    args = parser.parse_args()

    # 模型參數
    input_dim = 46
    num_classes = 10
    device = 'cpu'
    model_path = f"result/{args.model}/{args.name}/{args.who}/best_model.pth"
    model = load_model(model_path, input_dim, num_classes, device)

    # 載入資料
    path = "D:/my_project/android-camera-socket-stream-master/server/test/analyze/analyze_output"
    path = "D:/my_project/android-camera-socket-stream-master/server/formal/analyze/analyze_output"
    
    with open("D:/my_project/android-camera-socket-stream-master/server/save_list.json", "r") as f:
        save_list = json.load(f)

    test_data, file_list = process(path, save_list)

    # 推理
    predictions = infer(model, test_data, device)
    probs = torch.softmax(predictions, dim=1)

    # 找出每個類別中機率最高的圖
    best_per_class = {}

    for prob, file_number in zip(probs, file_list):
        prob_list = prob.tolist()
        pred_label = int(torch.argmax(prob).item())
        confidence = prob[pred_label].item()

        if pred_label not in best_per_class or confidence > best_per_class[pred_label][1]:
            best_per_class[pred_label] = (file_number, confidence)

    # 顯示最終選擇
    #print("\n=== 每個類別挑選機率最高的一張圖 ===")
    for cls in sorted(best_per_class.keys()):
        filename, conf = best_per_class[cls]
        print(f"{cls+1}_{filename}")

