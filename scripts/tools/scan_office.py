import numpy as np
import matplotlib.pyplot as plt
from rtlsdr import RtlSdr
import time

def scan_frequency(center_freq, sample_rate=2.048e6, duration=0.1):
    """
    掃描特定中心頻率附近的訊號
    """
    sdr = RtlSdr()
    try:
        sdr.sample_rate = sample_rate
        sdr.center_freq = center_freq
        sdr.gain = 'auto'
        samples = sdr.read_samples(256*1024)
        plt.psd(samples, NFFT=1024, Fs=sdr.sample_rate/1e6, Fc=sdr.center_freq/1e6)
    finally:
        sdr.close()

def main():
    targets = [
        {"name": "FM 廣播", "freq": 100e6},
        {"name": "FRS 對講機", "freq": 467.6e6},
        {"name": "IoT/遙控器", "freq": 433.9e6},
    ]
    print("--- 辦公室 SDR 訊號探勘開始 ---")
    plt.figure(figsize=(10, 12))
    for i, target in enumerate(targets):
        print(f"正在掃描 {target['name']} ({target['freq']/1e6:.1f} MHz)...")
        plt.subplot(len(targets), 1, i+1)
        scan_frequency(target['freq'])
        plt.title(f"{target['name']} - {target['freq']/1e6:.1f} MHz")
        plt.xlabel("Frequency (MHz)")
        plt.ylabel("Relative Power (dB)")

    plt.tight_layout()
    print("掃描完成！正在開啟譜圖...")
    plt.show()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"偵測到錯誤: {e}")
        print("\n貼心提醒：")
        print("1. 請確保 SDR 已經插在電腦上。")
        print("2. 確保沒有其他程式（如 SDR#）正在佔用 SDR。")
        print("3. 如果出現 'librtlsdr not found'，可能需要手動下載 rtlsdr.dll 並放在資料夾內。")
