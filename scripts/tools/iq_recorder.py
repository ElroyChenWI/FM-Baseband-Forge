import numpy as np
from rtlsdr import RtlSdr
import os
import time
import argparse

def record_samples(freq_mhz, duration_sec, output_file, sample_rate=2.048e6, gain=30):
    """
    錄製原始 IQ 樣本並存至檔案
    """
    sdr = RtlSdr()
    try:
        sdr.sample_rate = sample_rate
        sdr.center_freq = freq_mhz * 1e6
        sdr.gain = gain
        print(f"--- 開始錄製 ---")
        print(f"頻率: {freq_mhz} MHz")
        print(f"取樣率: {sample_rate/1e6} Msps")
        print(f"時長: {duration_sec} 秒")
        num_samples = int(sample_rate * duration_sec)
        samples = sdr.read_samples(num_samples)
        data_to_save = np.empty(samples.size * 2, dtype=np.float32)
        data_to_save[0::2] = samples.real
        data_to_save[1::2] = samples.imag
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'wb') as f:
            data_to_save.tofile(f)
        print(f"🎉 錄製完成！檔案存儲於: {output_file}")
        print(f"檔案大小: {os.path.getsize(output_file) / 1024 / 1024:.2f} MB")
    finally:
        sdr.close()

if __name__ == "__main__":
    DEFAULT_FREQ = 467.6250
    DEFAULT_DURATION = 3.0
    DEFAULT_PATH = "data/samples/security_ch10_iq.bin"
    record_samples(DEFAULT_FREQ, DEFAULT_DURATION, DEFAULT_PATH)
