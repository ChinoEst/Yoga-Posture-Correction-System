# Android Socket Stream 📱💻

本模組負責系統的數據採集與傳輸層。包含 Android 客戶端影像串流 App，以及對應的 Python Socket 接收伺服器。

---

## 📱 手機端配置 (Android App)

### 快速安裝
1. 使用 **Android Studio** 開啟本倉庫中的 `AndroidStocketSream` 資料夾。
2. 確認手機已開啟「開發者模式」與「USB 偵測」。
3. 點擊 **Run** 按鈕將 App 安裝至您的智慧型手機。

> **注意**：啟動 App 後，請進入設定畫面輸入伺服器端（PC）的 IP 位址，以確保連線建立。

---

## 🖥️ 伺服器端配置 (Python Server)

負責接收來自手機的影像流，並提供緩衝區供後端模型推論。

### 1. 環境設置
建議使用 `conda` 或 `venv` 隔絕環境：
```bash
python -m venv venv
# Windows 啟動環境
.\venv\Scripts\activate
# Linux/Mac 啟動環境
source venv/bin/activate

### 2. 運行
python server.py
