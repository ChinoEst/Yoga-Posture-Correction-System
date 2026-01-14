import os 
import pandas as pd
import json
import shutil
import cv2
import argparse
import sys
import inspect





class label2CSV:
    def __init__(self, fold, size):
        self.cur_pth = os.path.dirname(os.path.abspath(inspect.getfile(self.get_function_path)))
        self.root = os.path.join(self.cur_pth, "..")
        self.path = os.path.join(self.root, fold)
        self.whos = ['assistant', 'dean', 'david', 'father', 'harvey', 'jimmy', 'kevin','me', 'wei', 'yee']
        self.sides = ["side"]
        self.size = size
        self.poses = [str(i) for i in range(1, 11)]
        self.kp_num = 23
        self.column = ["x1", "y1", "x2", "y2", "x3", "y3", "x4", "y4", "x5", "y5", "x6", "y6", "x7", "y7", "x8", "y8", "x9", "y9", "x10", "y10", "x11", "y11", "x12", "y12", "x13", "y13", "x14", "y14", "x15", "y15", "x16", "y16", "x17", "y17", "x18", "y18", "x19", "y19", "x20", "y20", "x21", "y21", "x22", "y22", "x23", "y23", "label", "who", "bbox_w", "bbox_h"]
        
        
    
    def get_function_path(self):
        return os.path.dirname(os.path.abspath(inspect.getfile(self.get_function_path)))

    def run(self):        
        for who in self.whos:
            for side in self.sides:
                for pose in self.poses:
                    temp_pth = os.path.join(self.path, who, side, pose)
                    
                    if os.path.exists(temp_pth):
                        
                        json_list = os.listdir(temp_pth)
                        json_list = [jason for jason in json_list if jason.endswith(".json")]
                         
                        for jason in json_list:
                            
                            #read jason
                            with open(f'{temp_pth}/{jason}', 'r') as file:
                                json_data = json.load(file)
                                j_data = json_data[0]
                                    
                            
                            #read bbox
                            bbox = j_data["bbox"][0]
                            
                            bbox_up_x = bbox[0]
                            bbox_up_y = bbox[1]
                            bbox_down_x = bbox[2]
                            bbox_down_y = bbox[3]
                            
                            #w,h of bbox
                            w = bbox_down_x - bbox_up_x
                            h = bbox_down_y - bbox_up_y
                            
                            #center of bbox
                            center_x = (bbox_down_x + bbox_up_x)/2
                            center_y = (bbox_down_y + bbox_up_y)/2
                            
                            #Difference between center of bbox and center of new canvas
                            Difference_x = self.size/2 - center_x 
                            Difference_y = self.size/2 - center_y
                            
                            temp = []    
                                
                            for j in range(self.kp_num):
                                
                                #read keypoints coordinate in old frame -> coordinate in new canvas
                                x = j_data["keypoints"][j][0] + Difference_x
                                y = j_data["keypoints"][j][1] + Difference_y
                                temp.append(x)
                                temp.append(y)
                                    
                            file_name = jason.split(".")[0]
                            
                            temp.append(pose)
                            temp.append(who)
                            
                            temp.append(w)
                            temp.append(h)

                            df = pd.DataFrame(columns=self.column, data = [temp])
                                
                            df.to_csv(f'{temp_pth}/{file_name}.csv', index=False)
                            print(f"{file_name}")
                        



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="trans label to csv")
    parser.add_argument("--fold", type=str, default="data", help="fold of label")
    parser.add_argument("--size", type=int, default=1500, help="new canvas size for bbox paste")
    
    args = parser.parse_args()
    
    process = label2CSV(args.fold, args.size)
    process.run()
    
