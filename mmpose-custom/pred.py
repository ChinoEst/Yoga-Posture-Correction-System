from mmpose.apis import MMPoseInferencer
import os 



inferencer = MMPoseInferencer(
    pose2d = "configs/body_2d_keypoint/rtmpose/coco/final_config.py",
    pose2d_weights= 'weight.pth',
)

List = os.listdir("D:/my_project/android-camera-socket-stream-master/server/custom")

for file in List:  
    if file.endswith('.jpg'):
        result_generator = inferencer(f"D:/my_project/android-camera-socket-stream-master/server/custom/{file}", show=False, vis_out_dir = "looks", pred_out_dir = "looks")
        result = next(result_generator)
                
            
            
            
            