import numpy as np
from rtlsdr import RtlSdr
import time
import sys

def get_power(samples):
    """計算 IQ 樣本的平均功率 (dB)"""
    if len(samples) == 0:
        return -100
    p = np.mean(np.abs(samples)**2)
    return 10 * np.log10(p + 1e-12)

def scan_range(start_freq, end_freq, step_hz=1e6):
    """
    掃描指定範圍並尋找熱點
    """
    sdr = RtlSdr()
    active_points = []
    try:
        sdr.sample_rate = 2.048e6
        sdr.gain = 30
        current_f = start_freq
        print(f"{'頻率 (MHz)':<15} | {'功率 (dB)':<10} | {'狀態':<10}")
        print("-" * 40)
        while current_f <= end_freq:
            sdr.center_freq = current_f
            time.sleep(0.05)
            samples = sdr.read_samples(1024 * 64)
            power = get_power(samples)
            status = ""
            if power > -15:
                status = "🔥 [ACTIVE]"
                active_points.append((current_f / 1e6, power))
            print(f"{current_f/1e6:<15.2f} | {power:<10.2f} | {status}")
            current_f += step_hz
    finally:
        sdr.close()
    return active_points

def main():
    print("=== 辦公室頻率活動雷達 (UHF 段) ===")
    print("正在掃描 430MHz - 480MHz (對講機與商用通訊)...")
    active_list = scan_range(430e6, 480e6, step_hz=1e6)
    print("\n" + "="*30)
    if active_list:
        print("偵測到以下活躍頻點：")
        for f, p in active_list:
            print(f"👉 {f:.2f} MHz (強度: {p:.2f} dB)")
        print("\n建議：你可以用 SDR# 仔細查看這些頻點。")
    else:
        print("未偵測到明顯活動。建議調整天線位置或降低門檻值。")

if __name__ == "__main__":
    try:
        from rtlsdr import RtlSdr
    except ImportError:
        print("錯誤：找不到 pyrtlsdr。請確認已安裝。")
        sys.exit(1)
    main()
