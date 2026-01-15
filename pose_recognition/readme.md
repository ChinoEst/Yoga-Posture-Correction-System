
##  🖥️ 開發環境配置 

建議使用 Anaconda 管理環境以確保版本相容。

## 招式
包含十種拜日式常見招式
| 動作名稱 (Yoga Pose) | 
| :--- |
| 山式 (Mountain Pose) | 
| 前彎 (Standing Forward) | 
| 後灣 (Standing Backbend) | 
| 平板式 (Plank Pose) | 
| 上犬式 (Upward-Facing Dog) | 
| 眼鏡蛇式 (Cobra Pose) | 
| 下犬式 (Downward-Facing Dog) | 
| 八肢點式 (Eight-Limbed Pose) | 
| 鱷魚式 (Four-Limbed Staff Pose) | 
| 低弓箭步 (Low Lunge) | 



### Quick Start
打開 Anaconda Prompt，執行以下指令：
```bash
conda create --name yoga-pose python=3.8 -y
conda activate yoga-pose
pip install -r requirements.txt
```

## train
```bash
python FCNN.py --feature 34 --name yoga_model --batch 64 --num_classes 10 --side both
```
| Argument | type | Description |
| feature | int | nums of features, we use 23*2 46|
| name | str | filename |
| batch | int | batch size |
| test_member | str | the data of the member is for test, other for train |
| num_classes | int | numbers of yoga pose , default = 10 |
 

## model
[model]()

## prediction
```bash
python pred.py
```

## server
```bash
python pre_rate_API.py
```
