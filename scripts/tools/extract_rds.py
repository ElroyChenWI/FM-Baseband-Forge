import numpy as np
import scipy.signal as signal
from rtlsdr import RtlSdr
import matplotlib.pyplot as plt

FS_RF = 2.4e6
FREQ = 92.7e6
DURATION = 2.0

print(f"📡 [硬體層] 擷取 {FREQ/1e6} MHz 進行 RDS 數位層探勘...")
sdr = RtlSdr()
sdr.sample_rate = FS_RF
sdr.center_freq = FREQ
sdr.gain = 45.0
iq_raw = sdr.read_samples(int(FS_RF * DURATION))
sdr.close()

print("🧠 [DSP層] 執行 57kHz 數位載波鎖定...")
DECIMATION = 10
FS_BASEBAND = FS_RF / DECIMATION
iq_baseband = signal.decimate(iq_raw, DECIMATION, ftype='fir')
fm_demodulated = np.angle(iq_baseband[1:] * iq_baseband[:-1].conj())

NYQ = FS_BASEBAND / 2
bpf_rds_taps = signal.firwin(511, [56000.0/NYQ, 58000.0/NYQ], pass_zero=False)
rds_bpsk_signal = signal.lfilter(bpf_rds_taps, 1.0, fm_demodulated)

print("✅ [數位層] 正在生成 BPSK 訊號特徵圖...")

plot_samples = int(FS_BASEBAND * 0.002)
t = np.arange(plot_samples) / FS_BASEBAND

plt.figure(figsize=(12, 6))

plt.subplot(2, 1, 1)
plt.plot(t * 1000, rds_bpsk_signal[-plot_samples:], color='red')
plt.title('RDS (57kHz) BPSK Waveform - Microscopic View')
plt.xlabel('Time [ms]')
plt.ylabel('Amplitude')
plt.grid(True)

plt.subplot(2, 1, 2)
f, psd = signal.welch(fm_demodulated, FS_BASEBAND, nperseg=4096)
plt.semilogy(f/1000, psd)
plt.axvline(x=57, color='r', linestyle='--', label='RDS Center (57kHz)')
plt.title('FM Multiplex Spectrum (Zoom on RDS)')
plt.xlim(50, 65)
plt.xlabel('Frequency [kHz]')
plt.ylabel('PSD')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

print("\n--- 技術分析 ---")
print("💡 觀察紅色的 BPSK 波形：如果看到波峰突然「轉頭」或相位偏移，那就是數位資料 0 與 1 的切換點。")
print("這就是廣播電台傳送『電台名稱』與『播放曲目』的物理證據。")
