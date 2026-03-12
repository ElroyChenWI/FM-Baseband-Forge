import numpy as np
from rtlsdr import RtlSdr
import time
import os

def get_total_power(samples):
    """計算樣本的總能量強度"""
    return 10 * np.log10(np.mean(np.abs(samples)**2) + 1e-12)

def main():
    sdr = RtlSdr()
    results = []
    START_FREQ = 88.0
    END_FREQ = 108.0
    STEP = 1.05
    print(f"--- FM 全頻譜定量掃描開始 ---")
    print(f"{'頻率 (MHz)':<15} | {'相對強度 (dB)':<15} | {'偵測結果'}")
    print("-" * 50)
    try:
        sdr.sample_rate = 2.048e6
        sdr.gain = 25
        current_f = START_FREQ
        while current_f <= END_FREQ:
            sdr.center_freq = current_f * 1e6
            time.sleep(0.05)
            samples = sdr.read_samples(1024 * 64)
            power = get_total_power(samples)
            tag = "📶 [電台載波]" if power > -15 else ""
            print(f"{current_f:<15.2f} | {power:<15.2f} | {tag}")
            results.append((current_f, power))
            current_f += STEP
        strongest = max(results, key=lambda x: x[1])
        print("\n" + "="*30)
        print(f"💡 本環境最強頻點: {strongest[0]:.2f} MHz ({strongest[1]:.2f} dB)")
        print("建議針對此頻點進行「基頻結構分析 (Phase 3.2)」。")
    finally:
        sdr.close()

if __name__ == "__main__":
    main()
