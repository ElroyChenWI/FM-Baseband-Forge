# 🛰️ RTL-SDR Exploration: From FM to NOAA Satellites

無線電頻譜探險紀錄。利用 RTL-SDR 接收器，嘗試從基礎地面通訊（FM、FRS）逐步進化到接收並解碼來自太空的 **NOAA 氣象衛星雲圖**。

## 🚀 專案目標
- [x] **Phase 1: Ground Base** (FM Scanning, Security Radio Detection)
- [ ] **Phase 2: Signal Analysis** (IQ Data Processing, AM/FM Demodulation)
- [ ] **Phase 3: Deep Dive** (ADS-B Tracking, IoT Decoding)
- [ ] **Phase 4: Satellite Mission** (NOAA Weather Map Reception)

## 🛠️ 技術棧
- **Hardware**: RTL-SDR (R820T2 V3)
- **Language**: Python (pyrtlsdr, numpy, matplotlib)
- **Skills**: Signal Processing, Radio Frequency (RF), Python Automation

## 📔 開發日誌 (The Debugging Path)
「解決問題的過程」具備高度參考價值。[DEBUG_LOG.md](./docs/logs/DEBUG_LOG.md) 紀錄了如何克服硬體驅動、訊號衰減與環境遮蔽等技術挑戰。

### 熱門腳本連結
- [scan_office.py](./scripts/tools/scan_office.py): 快速掃描環境頻譜。
- [security_scanner.py](./scripts/tools/security_scanner.py): 定向尋找 FRS 對講機活動。

---
*Created with ❤️ by Elroy*
