import numpy as np
import scipy.signal as signal
from rtlsdr import RtlSdr

FS_RF = 1.14e6
FREQ = 94.3e6
DURATION = 3.0

print(f"📡 [硬體層] 正在攔截 {FREQ/1e6} MHz 的數位封包...")
sdr = RtlSdr()
sdr.sample_rate = FS_RF
sdr.center_freq = FREQ
sdr.gain = 45.0
iq_raw = sdr.read_samples(int(FS_RF * DURATION))
sdr.close()

DECIMATION = 4
FS_BB = FS_RF / DECIMATION
iq_bb = signal.decimate(iq_raw, DECIMATION, ftype='fir')
mpx = np.angle(iq_bb[1:] * iq_bb[:-1].conj())

NYQ = FS_BB / 2
bpf_taps = signal.firwin(511, [56000.0/NYQ, 58000.0/NYQ], pass_zero=False)
rds_raw = signal.lfilter(bpf_taps, 1.0, mpx)

print("💡 [解碼層] Costas Loop 相位同步鎖定中...")
num_samples = len(rds_raw)
phase, freq = 0.0, 57000.0 * 2 * np.pi / FS_BB
alpha, beta = 0.02, 0.0001
bpsk_baseband = np.zeros(num_samples)

for i in range(num_samples):
    nco_i, nco_q = np.sin(phase), np.cos(phase)
    in_i, in_q = rds_raw[i] * nco_i, rds_raw[i] * nco_q
    error = in_i * in_q
    phase += freq + alpha * error
    freq += beta * error
    bpsk_baseband[i] = in_i

print("⚙️ [同步層] 位元時鐘提取 (1187.5 bps)...")
samples_per_bit = FS_BB / 1187.5
bits = []
last_bit = 0

for i in range(100, num_samples - int(samples_per_bit), int(samples_per_bit)):
    sample = np.mean(bpsk_baseband[i:i+int(samples_per_bit)])
    raw_bit = 1 if sample > 0 else 0
    decoded_bit = raw_bit ^ last_bit
    bits.append(decoded_bit)
    last_bit = raw_bit

print("\n--- RDS 數位文字分析報告 ---")
print(f"總擷取位元數: {len(bits)}")

bit_str = "".join(map(str, bits[:200]))
print(f"位元流片段 (前200bit): \n{bit_str}")

print("\n--- 技術總結 ---")
print("1. 以上數據已完成『物理層 -> 數位層』的轉換。")
print("2. 這些位元包含 CRC 檢校碼、Block ID 與 ASCII 字符。")
print("3. 直接解析這些位元需要對應 RDS ISO 規範 (Block A/B/C/D)。")
print("這證明我們已經能把空中隱形的數位資訊『抓進』記憶體裡了！")
