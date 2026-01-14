import os 
import pandas as pd
import json
import shutil
import argparse
import sys
import numpy as np
import inspect
import random
from scipy.ndimage import gaussian_filter


class Data_Augmentation:
    def __init__(self, fold, amount):
        self.cur_pth = os.path.dirname(os.path.abspath(inspect.getfile(self.get_function_path)))
        self.root = os.path.join(self.cur_pth, "..", )
        self.path = os.path.join(self.root, fold)
        self.whos = ['assistant', 'dean', 'david', 'father', 'harvey', 'jimmy', 'kevin','me', 'wei', 'yee']
        self.sides = ["side"]
        self.poses = [str(i+1) for i in range(10)]
        self.kp_num = 23
        self.amount = amount
        self.mode2name = [
            self.shift, 
            self.rotate, 
            self.add_jitter,  # 新增隨機抖動
            self.random_scaling, # 新增隨機放大
            self.add_gaussian_noise, # 新增高斯噪音
        ]
        self.keypoints_columns = ["x1", "y1", "x2", "y2", "x3", "y3", "x4", "y4", "x5", "y5",
                     "x6", "y6", "x7", "y7", "x8", "y8", "x9", "y9", "x10", "y10",
                     "x11", "y11", "x12", "y12", "x13", "y13", "x14", "y14", "x15", "y15",
                     "x16", "y16", "x17", "y17", "x18", "y18", "x19", "y19", "x20", "y20",
                     "x21", "y21", "x22", "y22", "x23", "y23"]
        self.augmentation_list = [
            "shift", 
            "rotate", 
            "jitter",  # 新增隨機抖動
            "scaling", # 新增隨機放大
            "noise", # 新增高斯噪音
        ]

    def get_function_path(self):
        return os.path.dirname(os.path.abspath(inspect.getfile(self.get_function_path())))

    def rotate_points(self, points, angle, center=(0, 0)):
        theta = np.radians(angle)
        rotation_matrix = np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)]
        ])
        translated_points = points - center
        rotated_points = np.dot(translated_points, rotation_matrix.T)
        return rotated_points + center    

    def clip_points(self, points, width, height):
        points[:, 0] = np.clip(points[:, 0], 0, width)
        points[:, 1] = np.clip(points[:, 1], 0, height)
        return points

    def shift(self, df):

        W = df['bbox_w'].iloc[0]
        H = df['bbox_h'].iloc[0]
        
        shift_rate = random.choice([i for i in range(-5,6) if i != 0])
        
        w = W * shift_rate * 0.01
        h = H * shift_rate * 0.01
        
        for i in range(0, 23):
            df[self.keypoints_columns[2*i]] = (df[self.keypoints_columns[2*i]] + w).clip(lower=0, upper=W)
            df[self.keypoints_columns[2*i+1]] = (df[self.keypoints_columns[2*i+1]] + h).clip(lower=0, upper=H)
        return df

    def rotate(self, df):
        angle = random.choice([i for i in range(-10,11) if i != 0])
        
        keypoints = df[self.keypoints_columns].values.reshape(-1, 23, 2)
        center = (750, 750)
        for i in range(len(keypoints)):
            keypoints[i] = self.rotate_points(keypoints[i], angle, center=center)
            keypoints[i] = self.clip_points(keypoints[i], 1500, 1500)
            
        df[self.keypoints_columns] = keypoints.reshape(-1, 46)
        return df

    def add_jitter(self, df, jitter_strength=0.01):
        keypoints = df[self.keypoints_columns].values.reshape(-1, 23, 2)
        jitter = np.random.normal(0, jitter_strength, keypoints.shape)
        keypoints += jitter
        df[self.keypoints_columns] = keypoints.reshape(-1, 46)
        return df

    def random_scaling(self, df, scale_range=(0.9, 1.1)):
        scaling_factor = np.random.uniform(*scale_range)
        keypoints = df[self.keypoints_columns].values
        keypoints *= scaling_factor
        df[self.keypoints_columns] = keypoints
        return df

    def add_gaussian_noise(self, df, noise_strength=0.01):
        noise = np.random.normal(0, noise_strength, df[self.keypoints_columns].values.shape)
        df[self.keypoints_columns] = df[self.keypoints_columns].values + noise
        return df

    def run(self):
        for who in self.whos:
            for side in self.sides:
                for pose in self.poses:
                    temp_pth = os.path.join(self.path, who, side, pose, "augmentation")
                    csv_list = [C for C in os.listdir(temp_pth) if C.endswith(".csv")]
                     
                    for CSV in csv_list:
                        for ele in self.augmentation_list:
                            if ele in CSV:
                                csv_list.remove(CSV)
                                break
                    
                    amounts = len(csv_list) 
                    count = 1
                    used = []
                    
                    while (amounts>0 and amounts < self.amount and count<7):
                        while(True):
                            random_numbers = random.sample(self.mode2name, 3)
                            if not random_numbers in used:
                                used.append(random_numbers)
                                break
                        for csv in csv_list:
                            df = pd.read_csv(os.path.join(temp_pth, csv))
                            
                            for func in random_numbers:
                                df = func(df)
                            df.to_csv(f"{temp_pth}/{count}_{csv}", index=False)
                        
                        count += 1
                        amounts = len([C for C in os.listdir(temp_pth) if C.endswith(".csv")]) 
                        
                            
                            
                            
                            
                            
                            


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="data")
    parser.add_argument("--fold", type=str, default="data", help="label fold")
    parser.add_argument("--amount", type=int, default=400, help="1 for shift, 2 for rotate, 3 for shift&rotate, 4 for rotate&shift, 5 for jitter, 6 for scaling, 7 for time shift, 8 for noise, 9 for smoothing")


    args = parser.parse_args()

    processor = Data_Augmentation(args.fold, args.amount)
    processor.run()
