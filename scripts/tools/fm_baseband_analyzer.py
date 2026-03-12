import numpy as np
import matplotlib.pyplot as plt
from rtlsdr import RtlSdr
import scipy.signal as signal

def analyze_fm_structure(freq_mhz, duration=0.2):
    """
    分析 FM 訊號的基頻結構 (19kHz, 38kHz, 57kHz)
    """
    sdr = RtlSdr()
    try:
        sdr.sample_rate = 1.14e6
        sdr.center_freq = freq_mhz * 1e6
        sdr.gain = 'auto'
        print(f"正在擷取 {freq_mhz} MHz 的訊號進行結構分析...")
        samples = sdr.read_samples(256 * 1024)
        demodulated = np.angle(samples[1:] * np.conj(samples[:-1]))
        fs = sdr.sample_rate
        frequencies, psd = signal.welch(demodulated, fs, nperseg=4096)
        plt.figure(figsize=(10, 6))
        plt.semilogy(frequencies / 1000, psd)
        key_points = {
            19: "Stereo Pilot (19kHz)",
            38: "Stereo Sub (38kHz)",
            57: "RDS / RBDS (57kHz)"
        }
        for k_freq, label in key_points.items():
            plt.axvline(x=k_freq, color='r', linestyle='--', alpha=0.5)
            plt.text(k_freq, plt.ylim()[0], label, rotation=90, verticalalignment='bottom')
        plt.title(f"FM Baseband Spectrum Analysis - {freq_mhz} MHz")
        plt.xlabel("Frequency (kHz)")
        plt.ylabel("Power Spectral Density")
        plt.xlim(0, 100)
        plt.grid(True)
        plt.show()
    finally:
        sdr.close()

if __name__ == "__main__":
    analyze_fm_structure(100.0)
