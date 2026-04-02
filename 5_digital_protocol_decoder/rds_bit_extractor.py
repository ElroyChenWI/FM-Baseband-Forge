import numpy as np
import scipy.signal as signal
from rtlsdr import RtlSdr
import matplotlib.pyplot as plt

FS_RF = 1.14e6
FREQ = 94.3e6
DURATION = 2.0

sdr = RtlSdr()
sdr.sample_rate = FS_RF
sdr.center_freq = FREQ
sdr.gain = 40.0
iq_raw = sdr.read_samples(int(FS_RF * DURATION))
sdr.close()

DECIMATION = 4
FS_BB = FS_RF / DECIMATION
iq_bb = signal.decimate(iq_raw, DECIMATION, ftype='fir')
mpx = np.angle(iq_bb[1:] * iq_bb[:-1].conj())

NYQ = FS_BB / 2
bpf_taps = signal.firwin(255, [56000.0/NYQ, 58000.0/NYQ], pass_zero=False)
rds_raw = signal.lfilter(bpf_taps, 1.0, mpx)

print("💡 [鎖相層] 啟動 Costas Loop，正在從雜訊中同步 57kHz 相位...")

num_samples = len(rds_raw)
phase = 0.0
freq = 57000.0 * 2 * np.pi / FS_BB
alpha = 0.01
beta = alpha**2/4

bpsk_demod = np.zeros(num_samples)
error_log = np.zeros(num_samples)

for i in range(num_samples):
    nco_in = np.sin(phase)
    nco_quad = np.cos(phase)
    in_i = rds_raw[i] * nco_in
    in_q = rds_raw[i] * nco_quad
    error = in_i * in_q
    phase += freq + alpha * error
    freq += beta * error
    bpsk_demod[i] = in_i

print("✅ [數位層] 同步完成，正在繪製位元流 (Bitstream) ...")

plot_samples = int(FS_BB * 0.01)
t = np.arange(plot_samples) / FS_BB

plt.figure(figsize=(12, 6))
plt.plot(t*1000, bpsk_demod[-plot_samples:], color='purple', label='Demodulated Bits (BPSK Baseband)')
plt.axhline(0, color='black', alpha=0.3)
plt.title('RDS Digital Bit Extraction (Baseband)')
plt.xlabel('Time [ms]')
plt.ylabel('Digital Signal Level')
plt.grid(True)
plt.legend()
plt.show()

print("\n--- 技術解密 ---")
print("1. 你看到的紫色波形，如果維持在 0 以上代表 1，0 以下代表 0。")
print("2. 那些垂直的跳變就是數位資料的切換點。")
print("3. 下一關就是把這些 0 與 1 存進矩陣，並對照 RDS 協議規範翻譯成文字。")
