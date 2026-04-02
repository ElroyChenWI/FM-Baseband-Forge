# FM Baseband Forge: 詳細使用說明書 (ZH)

本指南將帶領你從零開始，透過 RTL-SDR 硬體與 Python 環境，重現專案中的 FM 廣播完整解碼過程。

## 1. 環境建置 (Environment Setup)

在開始任何通訊實驗前，必須確保 Python 環境與硬體驅動已正確安裝。

### 安裝必要函式庫
請在終端機 (PowerShell/CMD) 執行以下指令：
```powershell
pip install pyrtlsdr pyrtlsdrlib numpy scipy matplotlib
```
*註：`pyrtlsdrlib` 會自動處理 Windows 下缺少的 librtlsdr.dll 驅動問題。*

### 硬體連接
1. 將 RTL-SDR 接收器插入 USB 接口。
2. 連接 FM 天線。若在室內，建議將天線拉至窗口以獲得最佳 SNR (信噪比)。

---

## 2. 第一步：探索與偵測 (Exploration)

首先，我們需要確定環境中最強的 FM 電台訊號。

### 執行全頻譜掃描
使用 `fm_profiler.py` 掃描 88-108 MHz 頻段。
指令：
```powershell
python 1_rf_reconnaissance/fm_profiler.py
```
**預期結果：** 程式會產出頻譜能量圖。請記下能量最高點的頻率（例如 94.3 MHz）。

---

## 3. 第二步：立體聲訊號分析與提取 (Audio Engineering)

確認頻率後，我們開始解構類比音訊的組成成分。

### 提取 19kHz/38kHz/57kHz 組件
指令：
```powershell
python 4_spectrum_analytics/extract_harmonics.py
```
這會分離出 19kHz 導頻、38kHz 立體聲差分訊號以及 57kHz RDS 資料層，並初步驗證電台的發射完整度。

### 立體聲與去加重 (De-emphasis) 實驗
指令：
```powershell
python 3_analog_demodulation/fm_master_comparison.py
```
這會針對目標頻率產生四種對比音訊：
1. Mono (原始)
2. Mono (加上 50μs 去加重濾波)
3. Stereo (左右聲道分離)
4. Stereo (加上去加重濾波，最佳聽感)
檔案會存放於 `data/comparison/` 資料夾下。

---

## 4. 第三步：RDS 數位數據解碼 (Data Decoding)

最後，挑戰從訊號中榨取文字資訊。

### 執行最終解碼器
指令：
```powershell
python 5_digital_protocol_decoder/rds_final_decoder.py
```
此步驟會啟動「四階段 DSP 管線」：
1. **Costas Loop**：鎖定載波相位。
2. **Bit-Sync**：對齊位元採樣點。
3. **Differential Decode**：還原差分編碼。
4. **Frame Sync (Syndrome)**：鎖定 26-bit 訊框邊界。

**狀況排除：**
* 若出現 `🟢 [SYNC]` 但後方為亂碼：代表訊號強度不足，CRC 校驗出錯。
* 若完全找不到 PI 碼：請調整天線位置，並確保頻率設定正確。

---

## 5. 常見問題處理 (Troubleshooting)

### 資源佔用錯誤 (Access Denied)
若出現 `LIBUSB_ERROR_ACCESS`，代表硬體被前一個未正常關閉的程式鎖定。
1. 關閉所有 Python 終端機。
2. 拔掉並重新插入 SDR 硬體。

### 路徑報錯 (No such file)
請確保你在專案根目錄執行指令，並使用相對路徑，如 `python scripts/tools/xxx.py`。
