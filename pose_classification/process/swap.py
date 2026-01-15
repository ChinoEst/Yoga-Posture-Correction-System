import os
import pandas as pd
import shutil
import argparse
import inspect

class swap:
    def __init__(self, fold):
        # 获取当前脚本的路径
        self.cur_pth = os.path.dirname(os.path.abspath(inspect.getfile(self.get_function_path)))
        # 定义数据集根路径
        self.root = os.path.join(self.cur_pth, "..")
        self.path = os.path.join(self.root, fold)
        # 定义人物、视角、动作等信息
        self.whos = ['assistant', 'dean', 'david', 'father', 'harvey', 'jimmy', 'kevin', 'me', 'wei', 'yee']
        self.sides = ["side"]
        self.poses = [str(i) for i in range(1, 11)]
        self.kp_num = 23
        # 定义关键点列名
        self.keypoints_columns = ["x1", "y1", "x2", "y2", "x3", "y3", "x4", "y4", "x5", "y5",
                     "x6", "y6", "x7", "y7", "x8", "y8", "x9", "y9", "x10", "y10",
                     "x11", "y11", "x12", "y12", "x13", "y13", "x14", "y14", "x15", "y15",
                     "x16", "y16", "x17", "y17", "x18", "y18", "x19", "y19", "x20", "y20",
                     "x21", "y21", "x22", "y22", "x23", "y23"]

    def get_function_path(self):
        return os.path.dirname(os.path.abspath(inspect.getfile(self.get_function_path)))

    def clear(self, path):
        if os.path.exists(path):
            shutil.rmtree(path)

    def run(self):
        # 遍历人物、视角和动作
        for who in self.whos:
            for side in self.sides:
                for pose in self.poses:
                    temp_pth = os.path.join(self.path, who, side, pose)
                    # 检查路径是否存在
                    if os.path.exists(temp_pth):
                        csv_list = [C for C in os.listdir(temp_pth) if C.endswith(".csv") and not "swap" in C]
                        
                        # 清除 augmentation 文件夹
                        self.clear(f"{temp_pth}/augmentation")

                        for CSV in csv_list:
                            csv_path = os.path.join(temp_pth, CSV)
                            # 读取 CSV 数据
                            if os.path.isfile(csv_path):
                                df = pd.read_csv(csv_path)
                                df_new = df.copy()

                                # 获取图像宽度
                                w = 1500
                                for i in range(23):
                                    df_new.loc[0, self.keypoints_columns[2*i]] = w - df.loc[0, self.keypoints_columns[2*i]]

                                # 交换关键点对的 x 和 y 坐标
                                for i in range(11):
                                    temp_x = df_new[self.keypoints_columns[4*i+2]].copy()
                                    temp_y = df_new[self.keypoints_columns[4*i+3]].copy()
                                    
                                    df_new[self.keypoints_columns[4*i+2]] = df_new[self.keypoints_columns[4*i+4]]
                                    df_new[self.keypoints_columns[4*i+3]] = df_new[self.keypoints_columns[4*i+5]]
                                    
                                    df_new[self.keypoints_columns[4*i+4]] = temp_x
                                    df_new[self.keypoints_columns[4*i+5]] = temp_y
                                
                                # 保存新的 CSV 文件
                                df_new.to_csv(f"{temp_pth}/swap_{CSV}", index=False)
                                
                                print(CSV)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="data processing")
    parser.add_argument("--fold", type=str, default="data", help="label fold")
    
    args = parser.parse_args()
    processor = swap(args.fold)
    processor.run()
