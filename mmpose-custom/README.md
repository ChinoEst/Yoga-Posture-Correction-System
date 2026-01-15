### Keypoint Detection Server 🦴



本模組基於 OpenMMLab 的 **MMPose** 框架，負責即時提取人體 23 個關鍵點座標（Keypoints），並將數據傳送至辨識後端。



---



## 🖥️ 開發環境配置 (Anaconda)



由於 MMPose 依賴 MMCV 與 MMEngine，建議建立專屬環境以避免版本衝突。

<br>

## 建立並啟動環境

```bash

conda create --name mmpose-yoga python=3.8 -y

conda activate mmpose-yoga

```



請至[mmpose](https://github.com/open-mmlab/mmpose)官網下載完整版，並觀看教學學習如何使用

<br>

## 自定義修改說明
為了將 MMPose 整合至本系統，請進行以下檔案遷移與配置：
<br>
custom.py 移至 configs/_base_/datasets
<br>
__init__.py 和 my_metricd.py 移至mmpose/evaluation/metrics
<br>
final_config.py 移至configs/body_2d_keypoint/rtmpose/coco

<br>

## model

[download](https://drive.google.com/file/d/1b54simMddB91Rq3FjeJnC3_S71g7etKo/view?usp=sharing)


<br>

## server

```bash

python mmpose_service.py

```
