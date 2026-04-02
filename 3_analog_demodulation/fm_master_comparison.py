import numpy as np
import scipy.signal as signal
from rtlsdr import RtlSdr
from scipy.io import wavfile
import os

FS_RF = 2.4e6
FREQ = 94.3e6
DURATION = 10.0

print(f"📡 [硬體層] 正在攔截 {FREQ/1e6} MHz (10秒) 的原始射頻數據...")
sdr = RtlSdr()
sdr.sample_rate = FS_RF
sdr.center_freq = FREQ
sdr.gain = 40.0
iq_raw = sdr.read_samples(int(FS_RF * DURATION))
sdr.close()

print("🧠 [DSP層] 執行解調與濾波矩陣運算...")
DECIMATION = 10
FS_BASEBAND = int(FS_RF / DECIMATION)
NYQ = FS_BASEBAND / 2
iq_baseband = signal.decimate(iq_raw, DECIMATION, ftype='fir')
fm_demodulated = np.angle(iq_baseband[1:] * iq_baseband[:-1].conj())

hpf_taps = signal.firwin(255, 15.0/NYQ, pass_zero=False)
fm_clean = signal.lfilter(hpf_taps, 1.0, fm_demodulated)

lpf_15k_taps = signal.firwin(255, 15000.0/NYQ)
bpf_19k_taps = signal.firwin(255, [18500.0/NYQ, 19500.0/NYQ], pass_zero=False)
bpf_38k_taps = signal.firwin(255, [37500.0/NYQ, 38500.0/NYQ], pass_zero=False)

audio_mono_raw = signal.lfilter(lpf_15k_taps, 1.0, fm_clean)

pilot_19k = signal.lfilter(bpf_19k_taps, 1.0, fm_clean)
nco_38k = signal.lfilter(bpf_38k_taps, 1.0, pilot_19k**2)
nco_38k /= np.max(np.abs(nco_38k)) + 1e-12

audio_diff_raw = signal.lfilter(lpf_15k_taps, 1.0, fm_clean * nco_38k) * 2.0

left_raw = audio_mono_raw + audio_diff_raw
right_raw = audio_mono_raw - audio_diff_raw

def de_emphasis(data, fs, tau=50e-6):
    alpha = 1 / (tau * fs + 1)
    return signal.lfilter([alpha], [1, -(1 - alpha)], data)

os.makedirs('../data/comparison', exist_ok=True)

def save_wav(name, left_ch, right_ch=None):
    print(f"💾 正在產出: {name}...")
    if right_ch is None:
        out = left_ch
    else:
        out = np.vstack((left_ch, right_ch)).T
    out /= np.max(np.abs(out)) + 1e-12
    out = (out * 32767).astype(np.int16)
    wavfile.write(f'../data/comparison/{name}', FS_BASEBAND, out)

save_wav('1_mono_no_deemph.wav', audio_mono_raw)

save_wav('2_mono_with_deemph.wav', de_emphasis(audio_mono_raw, FS_BASEBAND))

save_wav('3_stereo_no_deemph.wav', left_raw, right_raw)

save_wav('4_stereo_with_deemph.wav', de_emphasis(left_raw, FS_BASEBAND), de_emphasis(right_raw, FS_BASEBAND))

print("\n🎉 大師級對比實驗完成！請至 data/comparison/ 目錄下試聽。")
