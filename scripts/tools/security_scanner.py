import numpy as np
from rtlsdr import RtlSdr
import time
import sys

FRS_CHANNELS = [
    467.5125, 467.5250, 467.5375, 467.5500, 467.5625,
    467.5750, 467.5875, 467.6000, 467.6125, 467.6250,
    467.6375, 467.6500, 467.6625, 467.6750
]

def get_power(samples):
    if len(samples) == 0: return -100
    p = np.mean(np.abs(samples)**2)
    return 10 * np.log10(p + 1e-12)

def main():
    sdr = RtlSdr()
    try:
        sdr.sample_rate = 256000
        sdr.gain = 40
        print("=== 台灣 FRS 對講機頻道監測 ===")
        print("按 Ctrl+C 結束監測\n")
        print(f"{'頻道':<6} | {'頻率 (MHz)':<12} | {'強度 (dB)':<10} | {'狀態'}")
        print("-" * 45)
        while True:
            for i, freq in enumerate(FRS_CHANNELS):
                sdr.center_freq = freq * 1e6
                time.sleep(0.02)
                samples = sdr.read_samples(1024 * 16)
                power = get_power(samples)
                status = "🟢 [可能有語音]" if power > -25 else ""
                if status:
                    print(f"CH{i+1:<4} | {freq:<12.4f} | {power:<10.2f} | {status}")
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\n監測結束。")
    finally:
        sdr.close()

if __name__ == "__main__":
    main()
