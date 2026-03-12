import numpy as np
import scipy.signal as signal
from rtlsdr import RtlSdr
import matplotlib.pyplot as plt
import os

FS_RF = 2.4e6
FREQ = 92.7e6
DURATION = 2.0

print(f"📡 [硬體層] 擷取 {FREQ/1e6} MHz 的 RF 樣本...")
sdr = RtlSdr()
sdr.sample_rate = FS_RF
sdr.center_freq = FREQ
sdr.gain = 40.0
iq_raw = sdr.read_samples(int(FS_RF * DURATION))
sdr.close()

print("🧠 [DSP層] 降取樣與 FM 解調...")
DECIMATION = 8
FS_BASEBAND = FS_RF / DECIMATION
iq_baseband = signal.decimate(iq_raw, DECIMATION, ftype='fir')
fm_demodulated = np.angle(iq_baseband[1:] * np.conj(iq_baseband[:-1]))

def extract_band(data, center_f, fs, width=1000):
    nyq = fs / 2
    low = (center_f - width) / nyq
    high = (center_f + width) / nyq
    taps = signal.firwin(255, [low, high], pass_zero=False)
    return signal.lfilter(taps, 1.0, data)

print("🎯 [濾波層] 分離 19k (Pilot), 38k (Stereo), 57k (RDS)...")
pilot_19k = extract_band(fm_demodulated, 19000, FS_BASEBAND)
stereo_38k = extract_band(fm_demodulated, 38000, FS_BASEBAND)
rds_57k = extract_band(fm_demodulated, 57000, FS_BASEBAND)

print("✅ 處理完成，繪製對比圖...")
plot_samples = int(FS_BASEBAND * 0.005)
t = np.arange(plot_samples) / FS_BASEBAND

fig, axs = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

axs[0].plot(t, pilot_19k[-plot_samples:], color='blue')
axs[0].set_title('19kHz Pilot Tone (1X)')
axs[0].grid(True)

axs[1].plot(t, stereo_38k[-plot_samples:], color='green')
axs[1].set_title('38kHz Stereo Subcarrier (2X)')
axs[1].grid(True)

axs[2].plot(t, rds_57k[-plot_samples:], color='red')
axs[2].set_title('57kHz RDS Subcarrier (3X)')
axs[2].set_xlabel('Time [s]')
axs[2].grid(True)

plt.tight_layout()

output_path = 'docs/assets/fm_harmonics_analysis.png'
os.makedirs(os.path.dirname(output_path), exist_ok=True)
plt.savefig(output_path)
print(f"✅ 圖表已儲存至: {output_path}")
plt.show()

def get_pwr(x): return 10 * np.log10(np.mean(x**2) + 1e-12)

print("\n--- 能量分析報告 ---")
print(f"19k Pilot Power: {get_pwr(pilot_19k):.2f} dB")
print(f"38k Stereo Power: {get_pwr(stereo_38k):.2f} dB")
print(f"57k RDS Power: {get_pwr(rds_57k):.2f} dB")
