import numpy as np
import scipy.signal as signal
from rtlsdr import RtlSdr
from scipy.io import wavfile
import os
import matplotlib.pyplot as plt

# --- 參數設定 (與專案標準一致) ---
FS_RF = 2.4e6
DECIMATION = 10
FS_BASEBAND = FS_RF / DECIMATION  # 240,000 Hz
FREQ = 94.3e6
DURATION = 5.0
TAU = 50e-6
OUTPUT_DIR = '../data/evolution'

os.makedirs(OUTPUT_DIR, exist_ok=True)

def apply_de_emphasis(data, fs, tau=50e-6):
    alpha = 1 / (tau * fs + 1)
    return signal.lfilter([alpha], [1, -(1 - alpha)], data)

def normalize_to_int16(data):
    data = data / (np.max(np.abs(data)) + 1e-12)
    return (data * 32767).astype(np.int16)

# --- 1. 硬體擷取 ---
print(f"📡 [Stage 0]正在擷取 2.4 MSPS 原始 I/Q 串流 (這就是數位噪音的起點)...")
sdr = RtlSdr()
sdr.sample_rate = FS_RF
sdr.center_freq = FREQ
sdr.gain = 40.0
iq_raw = sdr.read_samples(int(FS_RF * DURATION))
sdr.close()

# --- 2. 處理階段 ---

# Stage 0: Raw Chaos (把 IQ 像聲音一樣播放，聽起來是白噪音)
print("🔊 產出 Stage 0: Raw RF Chaos...")
# 為方便播放，將 IQ 降頻到 48kHz 但不進行解調
iq_audible = signal.resample(iq_raw, int(len(iq_raw) * 48000 / FS_RF))
raw_output = np.vstack((iq_audible.real, iq_audible.imag)).T
wavfile.write(f'{OUTPUT_DIR}/0_raw_noise.wav', 48000, normalize_to_int16(raw_output))

# Stage 1: Mono Whisper (FM 解調 + 15kHz 低通)
print("🧠 產出 Stage 1: The First Whisper (Mono)...")
iq_baseband = signal.decimate(iq_raw, DECIMATION, ftype='fir')
mpx = np.angle(iq_baseband[1:] * iq_baseband[:-1].conj())
NYQ = FS_BASEBAND / 2
lpf_15k = signal.firwin(255, 15000.0 / NYQ)
audio_mono_raw = signal.lfilter(lpf_15k, 1.0, mpx)
# 降採樣到 48kHz 用於播放
audio_mono_48k = signal.resample(audio_mono_raw, int(len(audio_mono_raw) * 48000 / FS_BASEBAND))
wavfile.write(f'{OUTPUT_DIR}/1_mono_flat.wav', 48000, normalize_to_int16(audio_mono_48k))

# Stage 2: Warmth Refined (去加重濾波)
print("🔥 產出 Stage 2: Professional Warmth (De-emphasis)...")
audio_mono_warm = apply_de_emphasis(audio_mono_raw, FS_BASEBAND, TAU)
audio_mono_warm_48k = signal.resample(audio_mono_warm, int(len(audio_mono_warm) * 48000 / FS_BASEBAND))
wavfile.write(f'{OUTPUT_DIR}/2_mono_warm.wav', 48000, normalize_to_int16(audio_mono_warm_48k))

# Stage 3: Stereo Emergence (立體聲矩陣解調)
print("✨ 產出 Stage 3: Full Stereo Emergence...")
# 提取 19kHz 導頻
bpf_19k = signal.firwin(255, [18500.0/NYQ, 19500.0/NYQ], pass_zero=False)
pilot = signal.lfilter(bpf_19k, 1.0, mpx)
# 透過平方產生 38kHz 載波
nco_38k = signal.lfilter(signal.firwin(255, [37500.0/NYQ, 38500.0/NYQ], pass_zero=False), 1.0, pilot**2)
nco_38k /= (np.max(np.abs(nco_38k)) + 1e-12)
# 提取 L-R 差分訊號
audio_diff = signal.lfilter(lpf_15k, 1.0, mpx * nco_38k) * 2.0
# 矩陣校對
left = apply_de_emphasis(audio_mono_raw + audio_diff, FS_BASEBAND, TAU)
right = apply_de_emphasis(audio_mono_raw - audio_diff, FS_BASEBAND, TAU)
stereo_out = np.vstack((left, right)).T
stereo_48k = signal.resample(stereo_out, int(len(stereo_out) * 48000 / FS_BASEBAND))
wavfile.write(f'{OUTPUT_DIR}/3_full_stereo.wav', 48000, normalize_to_int16(stereo_48k))

# --- 3. 視覺化對比 ---
print("📊 正在生成對比圖表...")
plt.figure(figsize=(15, 10))
stages = [
    ("Stage 0: Raw IQ PSD", iq_raw, FS_RF),
    ("Stage 1: Mono MPX Spectrum", mpx, FS_BASEBAND),
    ("Stage 2: De-emphasis Curve", audio_mono_warm, FS_BASEBAND),
    ("Stage 3: Stereo Separation (L vs R)", left, FS_BASEBAND)
]

for i, (title, data, fs) in enumerate(stages):
    plt.subplot(2, 2, i+1)
    if i == 0:
        plt.psd(data, NFFT=1024, Fs=fs/1e6, color='gray')
    elif i == 3:
        plt.plot(normalize_to_int16(left[:1000]), label='Left', alpha=0.7)
        plt.plot(normalize_to_int16(right[:1000]), label='Right', alpha=0.7)
        plt.legend()
    else:
        plt.psd(data, NFFT=1024, Fs=fs/1000)
    plt.title(title)
    plt.grid(True)

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/evolution_comparison.png')

print(f"\n🎉 演進完成！請到 {OUTPUT_DIR} 資料夾查看成果。")
print("1. 0_raw_noise.wav (原始噪音)")
print("2. 1_mono_flat.wav (扁平單聲道)")
print("3. 2_mono_warm.wav (專業溫潤音質)")
print("4. 3_full_stereo.wav (完整立體聲)")
print("5. evolution_comparison.png (頻譜對稱進化圖)")
