# Yoga Posture Correction System 🧘‍♂️

本專案是一個整合了 **Android 影像串流**、**MMPose 姿態檢測**與 **深度學習動作分類** 的完整端到端系統。透過本系統，使用者可以利用手機鏡頭進行實時的瑜珈動作分析與矯正建議。

---

## 📄 論文與展示 (Research & Demo)

* **完整論文**: [點此閱讀論文全文](https://docs.google.com/document/d/16b6bfrFxcTbrZ5zuPxvS1gzXgA2JIc3-/edit?usp=drive_link&ouid=111932951906831359654&rtpof=true&sd=true)
* **Demo 影片**: [點此查看系統運行演示](https://youtube.com/shorts/sPIlRIg1YYc?feature=share)

---

## 🏗 系統整合說明 (System Integration)

本專案的核心價值在於將以下三個獨立的技術模組進行了深度整合：

1.  **實時傳輸層 (`android-socket-main`)**: 
    * 負責將手機端攝像頭影像壓縮並透過 Socket 協定傳輸。
    * **整合點**: 重新開發了 Server 端的接收緩衝區，確保影像能無縫餵入後端的推論引擎。

2.  **演算法引擎 (`mmpose-custom`)**: 
    * 基於 MMPose 進行二次開發。
    * **整合點**: 自定義了 `my_metrics.py`，將 17 個關鍵點座標轉化為解剖學上的角度數據，為分類模型提供特徵。

3.  **動作決策層 (`pose_classification-master`)**: 
    * 使用 CNN/FCNN 進行姿勢分類。
    * **整合點**: 建立了數據預處理管道，將接收到的實時流數據標準化，達成每秒 20+ 幀的實時推論與反饋。

---

## 🚀 模組設置手冊

### 1. Android 端與 Server 連線
- 修改 `android-socket-main/server/server.py` 中的 IP 位址為您的主機 IP。
- 確保手機 App 指向相同的 IP 與 Port。

### 2. MMPose 環境配置
- 進入 `mmpose-custom` 資料夾，安裝 MMPose 依賴：
  ```bash
  pip install -r requirements.txt
