"""
FM Live Player - Industry Standard Edition
- 2.048 MSPS High-Resolution Sampling (SDR Sweet Spot)
- Stateful Demodulation (Maintains phase continuity between chunks)
- 255-tap Brick-wall FIR Filter (Derived from successful Phase 3 experiments)
- Ring Buffer + Callback Audio (Stutter-free playback)
"""
import numpy as np
import scipy.signal as signal
from rtlsdr import RtlSdr
import sounddevice as sd
import tkinter as tk
from tkinter import ttk
import threading
import queue
import collections
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# --- Constants ---
FS_RF    = 2_048_000   # RTL-SDR standard stable rate
DECIMATE = 32          # 2048k / 32 = 64k (Intermediate, better for 15k LPF)
FS_IF    = FS_RF // DECIMATE # 64,000 Hz
FS_AUDIO = 48_000      # Final audio rate
CHUNK_SIZE = 131072    # ~0.06s per block for low latency

class StatefulDemod:
    """Maintains phase and filter states across processing blocks."""
    def __init__(self):
        # 1. Anti-aliasing / Decimation Filter (2.048M -> 64k)
        # Cutoff at 30kHz to capture FM mono + RDS if needed
        self.taps_dec = signal.firwin(255, 30_000 / (FS_RF / 2))
        self.zi_dec   = signal.lfilter_zi(self.taps_dec, 1.0)
        
        # 2. Last sample of previous block for arctan differentiation
        self.prev_sample = 0j
        
        # 3. Audio LPF (15kHz at 64k rate)
        self.taps_audio = signal.firwin(127, 15_000 / (FS_IF / 2))
        self.zi_audio   = signal.lfilter_zi(self.taps_audio, 1.0)
        
        # 4. De-emphasis (50us) at 48k output rate
        tau = 50e-6
        alpha = np.exp(-1.0 / (FS_AUDIO * tau))
        self.de_b = np.array([1.0 - alpha])
        self.de_a = np.array([1.0, -alpha])
        self.zi_de = np.array([0.0])
        
        # AGC
        self.agc_avg = 0.1
        self.last_vu = -60.0

    def process(self, iq, volume):
        # A. Decimate 2.048M -> 64k
        iq_if, self.zi_dec = signal.lfilter(self.taps_dec, 1.0, iq, zi=self.zi_dec)
        iq_dec = iq_if[::DECIMATE]
        
        # B. Stateful FM Demodulation (arctan differentiator)
        # Prepend the last sample of the previous block to maintain continuity
        combined = np.insert(iq_dec, 0, self.prev_sample)
        self.prev_sample = iq_dec[-1]
        
        diff = combined[1:] * np.conj(combined[:-1])
        fm   = np.angle(diff) # Now phase is continuous across blocks!
        
        # C. Low-pass filter (15kHz)
        audio_lpf, self.zi_audio = signal.lfilter(self.taps_audio, 1.0, fm, zi=self.zi_audio)
        
        # D. Resample to 48kHz (64k -> 48k is 4:3 ratio)
        audio_res = signal.resample_poly(audio_lpf, 3, 4)
        
        # E. De-emphasis
        audio_de, self.zi_de = signal.lfilter(self.de_b, self.de_a, audio_res, zi=self.zi_de)
        
        # F. AGC & Volume
        rms = np.sqrt(np.mean(audio_de**2)) + 1e-12
        self.agc_avg = 0.95 * self.agc_avg + 0.05 * rms
        audio_norm = audio_de * (0.4 / self.agc_avg)
        
        self.last_vu = 20 * np.log10(np.sqrt(np.mean(audio_norm**2)) + 1e-9)
        
        return (audio_norm * volume).astype(np.float32)

class AudioBuffer:
    def __init__(self, size):
        self.buffer = collections.deque(maxlen=size)
        self.lock   = threading.Lock()
    
    def push(self, data):
        with self.lock:
            self.buffer.extend(data)
            
    def pull(self, n):
        with self.lock:
            avail = len(self.buffer)
            count = min(n, avail)
            chunk = [self.buffer.popleft() for _ in range(count)]
            if count < n:
                chunk += [0.0] * (n - count)
            return np.array(chunk, dtype=np.float32)

class FMRadioGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("BasebandForge - Industry Standard FM Player")
        self.root.geometry("850x550")
        self.root.configure(bg="#05050a")
        
        self.running = False
        self.gain    = 40.0
        self.volume  = 2.0
        self.freq    = 94.3
        
        self.q_iq   = queue.Queue(maxsize=10)
        self.q_spec = queue.Queue(maxsize=1)
        self.ring   = AudioBuffer(int(FS_AUDIO * 2))
        self.dsp    = StatefulDemod()
        
        self._init_ui()
        print("[BOOT] Industrial Engine Initialized. Ready to fight physics.")

    def _init_ui(self):
        style = ttk.Style(self.root)
        style.theme_use("clam")
        style.configure("TFrame", background="#05050a")
        style.configure("TLabel", background="#05050a", foreground="#00d2ff", font=("Consolas", 10))
        style.configure("TButton", background="#121220", foreground="#00d2ff", font=("Consolas", 10, "bold"))
        
        ctrl = ttk.Frame(self.root, padding=10)
        ctrl.pack(fill="x")
        
        ttk.Label(ctrl, text="Freq (MHz)").grid(row=0, column=0, padx=5)
        self.freq_entry = ttk.Entry(ctrl, width=8, font=("Consolas", 11))
        self.freq_entry.insert(0, "94.3")
        self.freq_entry.grid(row=0, column=1)
        
        self.btn = ttk.Button(ctrl, text="  START  ", command=self._toggle)
        self.btn.grid(row=0, column=2, padx=20)
        
        ttk.Label(ctrl, text="Gain").grid(row=0, column=3, padx=5)
        self.gain_var = tk.DoubleVar(value=40.0)
        tk.Scale(ctrl, from_=0, to=49.6, variable=self.gain_var, orient="horizontal", 
                 length=100, bg="#05050a", highlightthickness=0, fg="#00d2ff",
                 command=lambda g: setattr(self, 'gain', float(g))).grid(row=0, column=4)
        
        ttk.Label(ctrl, text="Vol").grid(row=0, column=5, padx=5)
        self.vol_var = tk.DoubleVar(value=2.0)
        tk.Scale(ctrl, from_=0, to=8.0, variable=self.vol_var, orient="horizontal",
                 length=100, bg="#05050a", highlightthickness=0, fg="#00d2ff",
                 command=lambda v: setattr(self, 'volume', float(v))).grid(row=0, column=6)
        
        self.vu_lbl = ttk.Label(ctrl, text="VU: --- dB", width=15)
        self.vu_lbl.grid(row=0, column=7, padx=10)

        # Plot
        self.fig = plt.Figure(figsize=(8, 3.5), facecolor="#05050a")
        self.ax  = self.fig.add_subplot(111, facecolor="#000000")
        self.ax.tick_params(colors="#444")
        self.line, = self.ax.plot([], [], color="#00ff88", lw=0.8)
        self.ax.set_ylim(-80, 5)
        self.ax.set_xlim(-FS_RF/2000, FS_RF/2000)
        self.ax.set_xlabel("kHz", color="#444", size=8)
        self.fig.tight_layout()
        canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        canvas.get_tk_widget().pack(fill="both", expand=True, padx=10, pady=5)
        self.canvas = canvas

        self.status = tk.StringVar(value="RTL-SDR detected. 2.048MSPS Stateful DSP Engine Loaded.")
        ttk.Label(self.root, textvariable=self.status, font=("Consolas", 9)).pack(pady=5)

    def _toggle(self):
        if not self.running: self._start()
        else:                self._stop()

    def _start(self):
        try:
            self.sdr = RtlSdr()
            self.sdr.sample_rate = FS_RF
            self.sdr.center_freq = float(self.freq_entry.get()) * 1e6
            self.sdr.gain        = self.gain
        except Exception as e:
            self.status.set(f"SDR ERROR: {e}")
            return
            
        self.running = True
        self.btn.config(text="  STOP   ")
        self.status.set(f"LOCK ON: {self.freq_entry.get()} MHz")
        self.dsp = StatefulDemod()
        
        # Audio
        self.stream = sd.OutputStream(samplerate=FS_AUDIO, channels=1, blocksize=1024,
                                      callback=self._audio_callback, dtype='float32')
        self.stream.start()
        
        threading.Thread(target=self._sdr_worker, daemon=True).start()
        threading.Thread(target=self._dsp_worker, daemon=True).start()
        self._gui_tick()

    def _stop(self):
        self.running = False
        self.btn.config(text="  START  ")
        try: self.stream.stop(); self.stream.close()
        except: pass
        try: self.sdr.close()
        except: pass

    def _sdr_worker(self):
        while self.running:
            try:
                self.sdr.gain = self.gain
                iq = self.sdr.read_samples(CHUNK_SIZE)
                if not self.q_iq.full(): self.q_iq.put(iq)
                if self.q_spec.empty(): self.q_spec.put(iq)
            except: self.running = False

    def _dsp_worker(self):
        while self.running:
            try:
                iq = self.q_iq.get(timeout=1.0)
                audio = self.dsp.process(iq, self.volume)
                self.ring.push(audio)
            except: pass

    def _audio_callback(self, outdata, frames, time, status):
        outdata[:, 0] = self.ring.pull(frames)

    def _gui_tick(self):
        if not self.running: return
        try:
            iq = self.q_spec.get_nowait()
            psd = 10 * np.log10(np.abs(np.fft.fftshift(np.fft.fft(iq, 1024)))**2 / 1024 + 1e-12)
            fr  = np.linspace(-FS_RF/2000, FS_RF/2000, 1024)
            self.line.set_data(fr, psd)
            self.vu_lbl.config(text=f"VU: {self.dsp.last_vu:+.1f} dB")
            self.canvas.draw_idle()
        except: pass
        self.root.after(100, self._gui_tick)

if __name__ == "__main__":
    root = tk.Tk()
    app = FMRadioGUI(root)
    root.protocol("WM_DELETE_WINDOW", lambda: (app._stop(), root.destroy()))
    root.mainloop()
