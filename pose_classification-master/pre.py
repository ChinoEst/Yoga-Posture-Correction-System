import torch
import numpy as np
import argparse
import pandas
import os
import pandas as pd

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


def process(path):
    keypoints_columns = ["x1", "y1", "x2", "y2", "x3", "y3", "x4", "y4", "x5", "y5",
                 "x6", "y6", "x7", "y7", "x8", "y8", "x9", "y9", "x10", "y10",
                 "x11", "y11", "x12", "y12", "x13", "y13", "x14", "y14", "x15", "y15",
                 "x16", "y16", "x17", "y17", "x18", "y18", "x19", "y19", "x20", "y20",
                 "x21", "y21", "x22", "y22", "x23", "y23"]
    file_list =  [CSV for CSV in os.listdir(path) if CSV.endswith(".csv")]
    
    empty = []

    
    for CSV in file_list:
        df = pd.read_csv(path + "/" + CSV)
        values = df[keypoints_columns].values.flatten()
        empty.append(values)
        
    return np.vstack(empty), file_list
    

# 示例推理流程
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="data")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--name", type=str, required=True)
    parser.add_argument("--who", type=str, required=True)
    parser.add_argument("--pose", type=int, required=True)
    parser.add_argument("--data", type=str, required=True)
    args = parser.parse_args()
    
    # 假设输入维度和类别数量
    input_dim = 46  # 输入维度，比如46个特征
    num_classes = 10  # 类别数量，比如10个类别
    model_path = f'result/{args.model}/{args.name}/{args.who}/best_model.pth' 
    
    # 加载模型到CPU或GPU
    device = 'cpu'  # 或者'cuda'如果你使用GPU
    model = load_model(model_path, input_dim, num_classes, device)


    path = f"{args.data}/{args.who}/side/{args.pose}/augmentation"
    
    test_data, file_list = process(path)
    

    # 推理
    predictions = infer(model, test_data, device)
    predicted_labels = torch.argmax(predictions, dim=1).tolist()
    
    count = 0
    # 打印预测结果
    for i in range(len(file_list)):
        if int(predicted_labels[i])+1 != args.pose:
            print(f"{file_list[i]}:{predicted_labels[i]+1}")
            count += 1
            
    print(f"total:{count}")