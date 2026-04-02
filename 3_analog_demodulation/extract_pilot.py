import numpy as np
import scipy.signal as signal
from rtlsdr import RtlSdr
import matplotlib.pyplot as plt
from scipy.io import wavfile
import os

FS_RF = 2.4e6
FREQ = 92.7e6
DURATION = 2.0

print("[硬體層] 正在攔截 2.4 MSPS I/Q 矩陣...")
sdr = RtlSdr()
sdr.sample_rate = FS_RF
sdr.center_freq = FREQ
sdr.gain = 35.0
iq_raw = sdr.read_samples(int(FS_RF * DURATION))
sdr.close()

print("[DSP層] 執行微積分頻率解調...")

DECIMATION = 10
FS_BASEBAND = FS_RF / DECIMATION
iq_baseband = signal.decimate(iq_raw, DECIMATION, ftype='fir')

fm_demodulated = np.angle(iq_baseband[1:] * np.conj(iq_baseband[:-1]))

print("[濾波層] 打造 19kHz 狙擊鏡，分離 Pilot Tone...")
NYQ = FS_BASEBAND / 2

bpf_taps = signal.firwin(255, [18500.0 / NYQ, 19500.0 / NYQ], pass_zero=False)

pilot_tone = signal.lfilter(bpf_taps, 1.0, fm_demodulated)

print("運算完成！繪製時域波形驗證...")

plot_samples = int(FS_BASEBAND * 0.005)
t = np.arange(plot_samples) / FS_BASEBAND

plt.figure(figsize=(10, 4))
plt.plot(t, pilot_tone[-plot_samples:])
plt.title('Extracted 19kHz Pilot Tone (Time Domain)')
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')
plt.grid(True)
plt.show()

print("[鎖相層] 倍頻技術：透過 19kHz 導頻產生同步的 38kHz 載波...")

pilot_squared = pilot_tone ** 2

bpf_38k_taps = signal.firwin(255, [37500.0 / NYQ, 38500.0 / NYQ], pass_zero=False)
nco_38k_raw = signal.lfilter(bpf_38k_taps, 1.0, pilot_squared)

nco_38k_out = nco_38k_raw / (np.max(np.abs(nco_38k_raw)) + 1e-12)

hpf_taps = signal.firwin(255, 15.0 / NYQ, pass_zero=False)
fm_demodulated = signal.lfilter(hpf_taps, 1.0, fm_demodulated)

print("[解碼層] 啟動混頻器，撕裂複合訊號...")

lpf_15k_taps = signal.firwin(255, 15000.0 / NYQ)

audio_L_plus_R = signal.lfilter(lpf_15k_taps, 1.0, fm_demodulated)

mixed_signal = fm_demodulated * nco_38k_out

audio_L_minus_R = signal.lfilter(lpf_15k_taps, 1.0, mixed_signal)
audio_L_minus_R = audio_L_minus_R * 2.0

print("[輸出層] 矩陣運算完成，左耳與右耳音軌正式分離！")
audio_left = audio_L_plus_R + audio_L_minus_R
audio_right = audio_L_plus_R - audio_L_minus_R

print("[音質層] 執行 50us 去加重濾波，還原溫潤人聲...")
def apply_de_emphasis(data, fs, tau=50e-6):
    alpha = 1 / (tau * fs + 1)
    return signal.lfilter([alpha], [1, -(1 - alpha)], data)

audio_left = apply_de_emphasis(audio_left, FS_BASEBAND)
audio_right = apply_de_emphasis(audio_right, FS_BASEBAND)

print("[存檔層] 正在將結果寫入 ../data/samples/stereo_separation.wav...")
stereo_output = np.vstack((audio_left, audio_right)).T
stereo_output /= np.max(np.abs(stereo_output)) + 1e-12
stereo_output = (stereo_output * 32767).astype(np.int16)

os.makedirs('../data/samples', exist_ok=True)
wavfile.write('../data/samples/stereo_separation.wav', int(FS_BASEBAND), stereo_output)

print("恭喜！你現在可以播放 data/samples/stereo_separation.wav 聽聽看分離效果了！")
