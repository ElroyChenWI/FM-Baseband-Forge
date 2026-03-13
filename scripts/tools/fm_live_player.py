import numpy as np
import scipy.signal as signal
from rtlsdr import RtlSdr
import sounddevice as sd
import tkinter as tk
from tkinter import ttk
import threading
import queue
import time
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class FMRadioApp:
    def __init__(self, root):
        self.root = root
        self.root.title("FM Baseband Forge - Real-time Player")
        self.root.geometry("800x600")
        
        # 參數設定
        self.sample_rate = 250000
        self.audio_rate = 44100
        self.center_freq = 94.3e6
        self.gain = 40.0
        self.volume = 0.5
        self.running = False
        self.data_queue = queue.Queue(maxsize=10)
        
        # DSP 狀態
        self.prev_iq = 0
        self.deemph_state = 0
        self.deemph_alpha = 1 - np.exp(-1 / (self.sample_rate * 50e-6))
        
        self.setup_ui()
        self.init_sdr()

    def setup_ui(self):
        # 控制區
        ctrl_frame = ttk.LabelFrame(self.root, text="Controls")
        ctrl_frame.pack(fill="x", padx=10, pady=5)
        
        ttk.Label(ctrl_frame, text="Frequency (MHz):").grid(row=0, column=0, padx=5)
        self.freq_var = tk.DoubleVar(value=94.3)
        self.freq_entry = ttk.Entry(ctrl_frame, textvariable=self.freq_var, width=8)
        self.freq_entry.grid(row=0, column=1, padx=5)
        
        self.btn_toggle = ttk.Button(ctrl_frame, text="START", command=self.toggle_stream)
        self.btn_toggle.grid(row=0, column=2, padx=10)
        
        ttk.Label(ctrl_frame, text="Gain:").grid(row=0, column=3, padx=5)
        self.gain_var = tk.DoubleVar(value=40.0)
        self.gain_scale = ttk.Scale(ctrl_frame, from_=0, to=49.6, variable=self.gain_var, orient="horizontal", command=self.update_gain)
        self.gain_scale.grid(row=0, column=4, padx=5)
        
        ttk.Label(ctrl_frame, text="Vol:").grid(row=0, column=5, padx=5)
        self.vol_var = tk.DoubleVar(value=0.5)
        self.vol_scale = ttk.Scale(ctrl_frame, from_=0, to=1.0, variable=self.vol_var, orient="horizontal", command=self.update_vol)
        self.vol_scale.grid(row=0, column=6, padx=5)

        # 頻譜圖區
        self.fig, self.ax = plt.subplots(figsize=(5, 3))
        self.ax.set_facecolor('black')
        self.line, = self.ax.plot([], [], color='#00FF00', lw=1)
        self.ax.set_ylim(-60, 20)
        self.ax.set_xlim(-125, 125) # 250kHz range
        self.ax.set_title("Real-time Baseband Spectrum", color='white')
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.get_tk_widget().pack(fill="both", expand=True, padx=10, pady=5)

    def init_sdr(self):
        try:
            self.sdr = RtlSdr()
            self.sdr.sample_rate = self.sample_rate
            self.sdr.gain = self.gain
        except Exception as e:
            print(f"Error: SDR not found. {e}")
            tk.messagebox.showerror("Error", "RTL-SDR not found. Please plug it in.")

    def update_gain(self, val):
        self.gain = float(val)
        if hasattr(self, 'sdr') and self.running:
            try: self.sdr.gain = self.gain
            except: pass

    def update_vol(self, val):
        self.volume = float(val)

    def toggle_stream(self):
        if not self.running:
            self.running = True
            self.btn_toggle.config(text="STOP")
            self.center_freq = self.freq_var.get() * 1e6
            self.sdr.center_freq = self.center_freq
            
            # 啟動處理執行緒
            self.worker_thread = threading.Thread(target=self.audio_worker, daemon=True)
            self.worker_thread.start()
            self.gui_update_loop()
        else:
            self.running = False
            self.btn_toggle.config(text="START")

    def audio_worker(self):
        # 15kHz 低通濾波器設計 (更專業的音質)
        nyq = self.sample_rate / 2
        lpf_taps = signal.firwin(101, 15000.0/nyq)
        lpf_state = np.zeros(len(lpf_taps) - 1)

        stream = sd.OutputStream(samplerate=self.audio_rate, channels=1)
        stream.start()
        
        chunk_size = 32768 # 增加緩衝區大小
        try:
            while self.running:
                samples = self.sdr.read_samples(chunk_size)
                
                # DSP: FM Demodulation
                angle = np.angle(samples[1:] * np.conj(samples[:-1]))
                
                # DSP: 15kHz LPF & De-emphasis
                audio_lpf, lpf_state = signal.lfilter(lpf_taps, 1.0, angle, zi=lpf_state)
                
                # Simple De-emphasis (50us)
                # y[n] = x[n]*alpha + y[n-1]*(1-alpha)
                # alpha 已經在 __init__ 計算過
                audio_de = []
                last_y = self.deemph_state
                for x in audio_lpf:
                    y = x * self.deemph_alpha + last_y * (1 - self.deemph_alpha)
                    audio_de.append(y)
                    last_y = y
                self.deemph_state = last_y
                audio_de = np.array(audio_de)

                # Faster Resampling (Linear Interpolation)
                old_indices = np.arange(len(audio_de))
                new_indices = np.linspace(0, len(audio_de) - 1, int(len(audio_de) * self.audio_rate / self.sample_rate))
                play_buffer = np.interp(new_indices, old_indices, audio_de).astype(np.float32)

                # Volume control (Max multiplier is 3.0, default 0.5 * 3 = 1.5)
                play_buffer *= (self.volume * 3.0) 
                play_buffer = np.clip(play_buffer, -1.0, 1.0)
                
                # Check for output overflow
                try:
                    stream.write(play_buffer)
                except Exception as e:
                    print(f"Audio Output Error: {e}")
                
                # Push to GUI Queue (Skip if GUI is slow)
                if self.data_queue.empty():
                    self.data_queue.put(samples)
                
        except Exception as e:
            print(f"Worker Error: {e}")
        finally:
            stream.stop()
            stream.close()

    def gui_update_loop(self):
        if not self.running: return
        
        try:
            samples = self.data_queue.get_nowait()
            psd = 10 * np.log10(np.abs(np.fft.fftshift(np.fft.fft(samples)))**2 / len(samples))
            freqs = np.linspace(-self.sample_rate/2000, self.sample_rate/2000, len(psd))
            
            self.line.set_data(freqs, psd)
            self.canvas.draw_idle()
        except queue.Empty:
            pass
            
        self.root.after(50, self.gui_update_loop)

if __name__ == "__main__":
    root = tk.Tk()
    app = FMRadioApp(root)
    root.mainloop()
    if hasattr(app, 'sdr'):
        app.sdr.close()
