
##  🖥️ 開發環境配置 

建議使用 Anaconda 管理環境以確保版本相容。

<br>

## 招式 (Yoga Poses)
包含十種拜日式常見招式：

| 編號 | 動作名稱 (Yoga Pose) |
| :---: | :--- |
| 1 | 山式 (Mountain Pose) |
| 2 | 前彎 (Standing Forward Fold) |
| 3 | 後彎 (Standing Backbend) |
| 4 | 平板式 (Plank Pose) |
| 5 | 上犬式 (Upward-Facing Dog) |
| 6 | 眼鏡蛇式 (Cobra Pose) |
| 7 | 下犬式 (Downward-Facing Dog) |
| 8 | 八肢點地式 (Eight-Limbed Pose) |
| 9 | 鱷魚式 (Four-Limbed Staff Pose) |
| 10 | 低弓箭步 (Low Lunge) |

<br>

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

<br>

| Argument | type | Description |
| :---: | :---: | :---: |
| feature | int | nums of features, we use 23*2 46 |
| name | str | filename |
| batch | int | batch size |
| test_member | str | the data of the member is for test, other for train |
| num_classes | int | numbers of yoga pose , default = 10 |
 
<br>

## model
[model]()

<br>

## prediction
```bash
python pred.py
```
<br>

## server
```bash
python pre_rate_API.py
```
