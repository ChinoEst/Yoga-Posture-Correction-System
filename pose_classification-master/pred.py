import torch
import numpy as np
import argparse
import pandas
import os
import pandas as pd
import json


# 假设你的FCNN模型已经定义
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

# 加载保存的模型权重
def load_model(model_path, input_dim, num_classes, device='cpu'):
    # 初始化模型
    model = FCNN(input_dim, num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)  # 如果你在GPU上推理，这里用.cuda()
    model.eval()  # 切换到推理模式
    return model

# 推理函数
def infer(model, test_data, device='cpu'):
    test_data = torch.tensor(test_data, dtype=torch.float32).to(device)  # 转换为tensor
    with torch.no_grad():  # 在推理时不需要计算梯度
        output = model(test_data)
    return output


def process(path, save_list):

    size = 1500
    kp_num = 23
    
    
    
    
    
    
    empty = []
    for JSON in save_list:
        with open(f'{path}/{JSON}.json', 'r') as file:
            json_data = json.load(file)
            j_data = json_data[0]
            
            
            #read bbox
            bbox = j_data["bbox"][0]
            
            bbox_up_x = bbox[0]
            bbox_up_y = bbox[1]
            bbox_down_x = bbox[2]
            bbox_down_y = bbox[3]
            

            
            #center of bbox
            center_x = (bbox_down_x + bbox_up_x)/2
            center_y = (bbox_down_y + bbox_up_y)/2
            
            #Difference between center of bbox and center of new canvas
            Difference_x = size/2 - center_x 
            Difference_y = size/2 - center_y
            
            temp = []    
                
            for j in range(kp_num):
                
                #read keypoints coordinate in old frame -> coordinate in new canvas
                x = j_data["keypoints"][j][0] + Difference_x
                y = j_data["keypoints"][j][1] + Difference_y
                temp.append(x)
                temp.append(y)
                    
            
            empty.append(temp)
    
        
    return np.vstack(empty), save_list
    

# 示例推理流程
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="data")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--name", type=str, required=True)
    parser.add_argument("--who", type=str, required=True)
    #parser.add_argument("--save_list", type=json.loads, help="number of image", required=True)
    #parser.add_argument("--pose", type=int, required=True)
    #parser.add_argument("--data", type=str, required=True)
    args = parser.parse_args()
    
    # 假设输入维度和类别数量
    input_dim = 46  # 输入维度，比如46个特征
    num_classes = 10  # 类别数量，比如10个类别
    model_path = f"result/{args.model}/{args.name}/{args.who}/best_model.pth"
    
    # 加载模型到CPU或GPU
    device = 'cpu'  # 或者'cuda'如果你使用GPU
    model = load_model(model_path, input_dim, num_classes, device)


    path = "D:/my_project/android-camera-socket-stream-master/server/analyze/analyze_output"
    
    
    with open("D:/my_project/android-camera-socket-stream-master/server/save_list.json", "r") as f:
        save_list = json.load(f)
    
    test_data, file_list = process(path, save_list)
    

    # 推理
    predictions = infer(model, test_data, device)
    predicted_labels = torch.argmax(predictions, dim=1).tolist()
    
    
    for item, file_number in zip(predicted_labels, file_list):
        
        print(f"{item+1}_{file_number}")
        

    """
    with open(f"{path}/../output.txt", "w") as file:
        for item, file in predicted_labels, file_list:
            file.write(item+1 + "," + file + "\n")  # 每個元素寫入一行
    """


            
    