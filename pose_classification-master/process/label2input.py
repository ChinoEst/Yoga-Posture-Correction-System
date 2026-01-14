import os 
import pandas as pd
import json
import shutil
import argparse
import sys
import inspect
import random
import numpy as np
from scipy import interpolate




class label2input:
    def __init__(self, test_who, classes, mode, train_amount, test_amount, fold, Del, add):
        self.fold = fold
        self.test_who = test_who
        self.train_amount = train_amount
        self.test_amount = test_amount
        self.classes = classes
        self.cur_pth = os.path.dirname(os.path.abspath(inspect.getfile(self.get_function_path)))
        self.root = os.path.join(self.cur_pth, "..")
        self.mode = mode
        self.whos = ['assistant', 'dean', 'david', 'father', 'harvey', 'jimmy', 'kevin','me', 'wei', 'yee']
        self.sides = ["front", "side"]
        self.poses = [str(i) for i in range(1, self.classes+1)]
        self.output_pth = os.path.join(self.root, "dataset_input")
        self.keypoints_columns = ["x1", "y1", "x2", "y2", "x3", "y3", "x4", "y4", "x5", "y5",
                     "x6", "y6", "x7", "y7", "x8", "y8", "x9", "y9", "x10", "y10",
                     "x11", "y11", "x12", "y12", "x13", "y13", "x14", "y14", "x15", "y15",
                     "x16", "y16", "x17", "y17", "x18", "y18", "x19", "y19", "x20", "y20",
                     "x21", "y21", "x22", "y22", "x23", "y23"]
        self.column = ["keypoint", "label", "who", "W", "H"]
        self.train_cat = [[] for i in range(self.classes)]
        self.test_cat = [[] for i in range(self.classes)]
        
        if Del is not None:
            for ele1 in Del:
                self.keypoints_columns =[ele2 for ele2 in self.keypoints_columns if (ele2 != f"x{ele1}" and ele2 != f"y{ele1}")]
        if add is not None:
            for ele in add:
                self.keypoints_columns.append(ele)
        
    def get_function_path(self):
        return os.path.dirname(os.path.abspath(inspect.getfile(self.get_function_path)))
        
    
    
    def csv_process(self, path, who): 
        df = pd.read_csv(path)
        label = path.split(os.sep)[-3]
        side = path.split(os.sep)[-4]
        #subfold = path.split(os.sep)[-6]
        #who = df['who'].iloc[0]
        
        df['keypoints'] = df[self.keypoints_columns].values.tolist()
        temp = df['keypoints'].tolist()
        

        
        
        if self.mode == 1 and side == "side":
            label = 10 + int(float(label))
        if who != self.test_who:
            self.train_cat[int(label)-1].append(temp)
        else:
            self.test_cat[int(label)-1].append(temp)
        #print(path)
    
        
    def save(self):
        train_dict = {f"{i+1}": self.train_cat[i] for i in range(self.classes)}
        test_dict = {f"{i+1}": self.test_cat[i] for i in range(self.classes)}
        
        for i in range(self.classes):
            print(f"train pose {i+1}: {len(self.train_cat[i])}")
            print(f"test pose {i+1}: {len(self.test_cat[i])}")
        
        with open(f'{self.output_pth}/{self.test_who}_train.json', 'w') as json_file:
            json.dump(train_dict, json_file, indent=4)
        
        with open(f'{self.output_pth}/{self.test_who}_test.json', 'w') as json_file:
            json.dump(test_dict, json_file, indent=4)
            
            
    def gather_files_by_pose(self, trained=True):
        file_paths = {pose: [] for pose in self.poses}
        who_list = self.whos.copy()
        if trained:
            who_list.remove(self.test_who)
        else:
            who_list = [self.test_who]
    

        for side in self.sides:
            for i in self.poses:
                for who in who_list:
                    temp_pth = os.path.join(self.root, self.fold, who, side, i)
                    if os.path.exists(temp_pth):
                        files = [os.path.join(temp_pth, "augmentation", file) for file in os.listdir(os.path.join(temp_pth, "augmentation")) if file.endswith('.csv') and not "timeshift" in file]
                        
                        """
                        if i == "9" and who in ["father", "me", "assistant"]:
                            continue
                        """
                        
                        if self.mode == 1:
                            if side == "side":
                                tmp = 10 + int(float(i))
                                file_paths[str(tmp)].extend(files)
                            else:
                                file_paths[i].extend(files)
                        
                        elif self.mode == 2:
                            file_paths[i].extend(files)
                            
                        elif self.mode == 3:
                            if side == "front":
                                file_paths[i].extend(files)
                        else:
                            if side == "side":
                                file_paths[i].extend(files)
                        
        for pose, file in file_paths.items():
            print(f"{pose}: {len(file)}")
            
            
            
    
        return file_paths
    
    def data_general_balanced(self, file_paths_by_pose, amount):
        
    
        for pose, file_paths in file_paths_by_pose.items():

            if len(file_paths) < amount:
                print(f"Warning: Not enough data for pose {pose}. Requested {amount}, but found {len(file_paths)}. Using all available files.")
                selected_files = file_paths
            else:
                swap = [file for file in file_paths if "swap" in file]
                no_swap = [file for file in file_paths if not "swap" in file]

                selected_files1 = random.sample(swap, int(amount/2))
                selected_files2= random.sample(no_swap, int(amount/2))
                selected_files = selected_files1 + selected_files2
                
    
            for file_path in selected_files:    
                who = file_path.split(os.sep)[-5]
                
                self.csv_process(file_path, who)     
    
    def run(self):
        train_files_by_pose = self.gather_files_by_pose(trained=True)
        test_files_by_pose = self.gather_files_by_pose(trained=False)
    
        self.data_general_balanced(train_files_by_pose, self.train_amount)
        self.data_general_balanced(test_files_by_pose, self.test_amount)

        self.save()
        print(f"feature:{len(self.keypoints_columns)}")




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description= "data to model input, csv to json")
    parser.add_argument("--test_member", type=str, required=True , help= "assistant, dean, david, father, harvey, kevin ,me, yee for test data ,default for assistant" )
    parser.add_argument("--classes", type=int, default=10, help= "10 or 20")
    parser.add_argument("--mode", type=int, default=4, help="1 for 20 class, 2 for 10 class, 3 for front, 4 for side")
    parser.add_argument("--train", type=int, required=True, help="amount for training data")
    parser.add_argument("--test", type=int, required=True, help="amount for testing data")
    parser.add_argument("--fold", type=str, required=True, help="fold of dataset")
    parser.add_argument("--Del", type=int, nargs='*', help="ele for del")
    parser.add_argument("--add", type=str, nargs='*', help="ele for add")
    args = parser.parse_args()
    
    
    processor = label2input(args.test_member, args.classes, args.mode, args.train, args.test, args.fold, args.Del, args.add)
    processor.run()      