import numpy as np
import scipy.signal as signal
import wave
import os

def demodulate_nfm(input_bin, output_wav, sample_rate=2.048e6, decimation=16):
    """
    將原始 IQ 樣本解調為 NFM 音訊
    """
    print(f"--- 軟體解調開始 ---")
    print(f"讀取檔案: {input_bin}")
    data = np.fromfile(input_bin, dtype=np.float32)
    iq_samples = data[0::2] + 1j * data[1::2]
    avg_pwr = 10 * np.log10(np.mean(np.abs(iq_samples)**2) + 1e-12)
    print(f"平均訊號強度: {avg_pwr:.2f} dB")
    if avg_pwr < -25:
        print("⚠️ 警告：訊號能量極低，錄製到的可能僅是底噪（靜音）。")
    else:
        print("✅ 偵測到明顯能量，這段錄音中可能有訊號！")
    demodulated = np.angle(iq_samples[1:] * np.conj(iq_samples[:-1]))
    audio_data = signal.decimate(demodulated, decimation, ftype='fir')
    audio_data = audio_data / np.max(np.abs(audio_data))
    audio_data_int16 = (audio_data * 32767).astype(np.int16)
    new_fs = int(sample_rate / decimation)
    with wave.open(output_wav, 'wb') as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(new_fs)
        wav_file.writeframes(audio_data_int16.tobytes())
    print(f"🎉 解調完成！音訊已存於: {output_wav}")
    print(f"最終採樣率: {new_fs} Hz")

if __name__ == "__main__":
    BIN_FILE = "data/samples/security_ch10_iq.bin"
    WAV_FILE = "data/samples/security_ch10_audio.wav"
    if os.path.exists(BIN_FILE):
        demodulate_nfm(BIN_FILE, WAV_FILE)
    else:
        print(f"錯誤：找不到輸入檔案 {BIN_FILE}，請確認是否先執行過錄製。")
