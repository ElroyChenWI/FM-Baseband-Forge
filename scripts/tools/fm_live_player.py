"""
FM Live Player - Pro Edition (1.152 MHz Offset Tuning)
DSP: 1.152MHz IQ (Offset -150kHz) -> Mix -> Decimate 6x -> 192kHz -> Demod -> Decimate 4x -> 48kHz Audio
This approach avoids the RTL-SDR DC spike and results in much cleaner audio.
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

# Tuning Strategy: 
# We tune the hardware 150kHz BELOW the actual station frequency.
# High-frequency noise/DC spike is now outside our signal band.
OFFSET_HZ = 150_000
FS_IN     = 1_152_000   # 1.152 MHz
FS_IF     = 192_000     # 1.152M / 6
FS_AUDIO  = 48_000      # 192k / 4
CHUNK_IQ  = 115_200     # 0.1s block


class FMDemodPro:
    def __init__(self):
        # Stage 1: DDC mix + Decimate 6x (1.152M -> 192k)
        # NCO to shift signal back to center (+150kHz)
        self.nco_phase = 0.0
        self.nco_step  = 2 * np.pi * OFFSET_HZ / FS_IN
        
        # Anti-alias filter for 192k IF (100kHz bandwidth)
        self.lpf1_b = signal.firwin(65, 80_000 / (FS_IN / 2))
        self.lpf1_zi = signal.lfilter_zi(self.lpf1_b, [1.0])

        # Stage 2: Audio LPF (15kHz at 192k)
        self.lpf2_b = signal.firwin(65, 15_000 / (FS_IF / 2))
        self.lpf2_zi = signal.lfilter_zi(self.lpf2_b, [1.0])

        # Stage 3: De-emphasis 50us at 48k
        tau = 50e-6
        alpha = np.exp(-1.0 / (FS_AUDIO * tau))
        self.de_b = np.array([1.0 - alpha])
        self.de_a = np.array([1.0, -alpha])
        self.de_zi = np.zeros(1)

        self.rms_avg = 0.1
        self.last_rms = 0.0

    def process(self, iq):
        # 1. Digital Down Conversion (Mix + Filter + Decimate)
        n = np.arange(len(iq))
        phases = self.nco_phase + n * self.nco_step
        self.nco_phase = (phases[-1] + self.nco_step) % (2 * np.pi)
        
        # Shift spectrum up by 150kHz
        iq_shifted = iq * np.exp(1j * phases)
        
        # Filter and decimate 6x
        iq_if, self.lpf1_zi = signal.lfilter(self.lpf1_b, [1.0], iq_shifted, zi=self.lpf1_zi)
        iq_dec = iq_if[::6] # 192kHz

        # 2. FM Demodulation (Arctan differentiator)
        # Handle zero division by adding eps
        diff = iq_dec[1:] * np.conj(iq_dec[:-1])
        fm = np.angle(diff) # result is -pi to pi

        # 3. Audio Extraction (LPF 15k + Decimate 4x)
        # Apply LPF at 192k IF rate
        audio_lpf, self.lpf2_zi = signal.lfilter(self.lpf2_b, [1.0], fm, zi=self.lpf2_zi)
        audio = audio_lpf[::4] # 48kHz

        # 4. De-emphasis
        audio, self.de_zi = signal.lfilter(self.de_b, self.de_a, audio, zi=self.de_zi)

        # 5. AGC
        rms = float(np.sqrt(np.mean(audio ** 2)) + 1e-9)
        self.rms_avg = 0.98 * self.rms_avg + 0.02 * rms
        audio = audio * (0.6 / self.rms_avg) # Target -4.4 dBFS
        
        self.last_rms = float(np.sqrt(np.mean(audio ** 2)))
        return audio.astype(np.float32)


class AudioRingBuffer:
    def __init__(self, capacity):
        self._buf = collections.deque(maxlen=capacity)
        self._lock = threading.Lock()

    def push(self, samples):
        with self._lock: self._buf.extend(samples.tolist())

    def pull(self, n):
        with self._lock:
            avail = len(self._buf)
            pull_n = min(n, avail)
            out = np.array([self._buf.popleft() for _ in range(pull_n)], dtype=np.float32)
            if pull_n < n:
                out = np.concatenate([out, np.zeros(n - pull_n, dtype=np.float32)])
            return out

    def size(self):
        return len(self._buf)


class FMRadioApp:
    def __init__(self, root):
        self.root = root
        self.root.title("FM Baseband Forge - Pro Edition")
        self.root.geometry("820x520")
        self.root.configure(bg="#0a0a12")

        self.gain = 40.0
        self.volume = 1.0
        self.running = False

        self.q_iq = queue.Queue(maxsize=12)
        self.q_spec = queue.Queue(maxsize=2)
        self.ring = AudioRingBuffer(FS_AUDIO * 5)
        self.demod = FMDemodPro()

        self._build_ui()

    def _build_ui(self):
        style = ttk.Style(); style.theme_use("clam")
        bg, fg, acc = "#0a0a12", "#bbbbbb", "#00d2ff"
        style.configure("TLabel", background=bg, foreground=fg, font=("Consolas", 10))
        style.configure("TButton", background="#1a1a2e", foreground=acc, font=("Consolas", 10, "bold"))
        style.configure("TLabelframe", background=bg, foreground=acc)
        style.configure("TLabelframe.Label", background=bg, foreground=acc, font=("Consolas", 10, "bold"))

        top = ttk.Frame(self.root, padding=6)
        top.pack(fill="x", side="top", padx=10, pady=5)

        ttk.Label(top, text="Frequency (MHz):").grid(row=0, column=0, padx=5)
        self.freq_var = tk.DoubleVar(value=94.3)
        ttk.Entry(top, textvariable=self.freq_var, width=8, font=("Consolas", 11)).grid(row=0, column=1)

        self.btn = ttk.Button(top, text="  START  ", command=self._toggle)
        self.btn.grid(row=0, column=2, padx=15)

        ttk.Label(top, text="RF Gain:").grid(row=0, column=3, padx=5)
        self.gain_var = tk.DoubleVar(value=40.0)
        self.gain_lbl = ttk.Label(top, text="40 dB", width=6)
        ttk.Scale(top, from_=0, to=49.6, variable=self.gain_var, length=100, 
                  command=lambda v: [setattr(self, 'gain', float(v)), self.gain_lbl.config(text=f"{float(v):.0f} dB")]).grid(row=0, column=4)
        self.gain_lbl.grid(row=0, column=5)

        ttk.Label(top, text="Vol:").grid(row=0, column=6, padx=5)
        self.vol_var = tk.DoubleVar(value=1.5)
        ttk.Scale(top, from_=0, to=5.0, variable=self.vol_var, length=80,
                  command=lambda v: setattr(self, 'volume', float(v))).grid(row=0, column=7)

        self.vu_lbl = ttk.Label(top, text="VU: ---", width=12, foreground="#00ff88")
        self.vu_lbl.grid(row=0, column=8, padx=10)

        # Plot
        self.fig = plt.Figure(figsize=(8, 3), facecolor="#0a0a12")
        self.ax = self.fig.add_subplot(111, facecolor="#050510")
        self.ax.tick_params(colors="#444444", labelsize=8)
        self.line, = self.ax.plot([], [], color="#00d2ff", lw=1)
        self.ax.set_ylim(-80, 5)
        self.ax.set_xlim(-FS_IN/2000, FS_IN/2000)
        self.ax.set_ylabel("dBFS", color="#444", size=8)
        self.fig.tight_layout()
        cv = FigureCanvasTkAgg(self.fig, master=self.root)
        cv.get_tk_widget().pack(fill="both", expand=True, padx=10, pady=5)
        self.canvas = cv

        self.status = tk.StringVar(value="Pro Ready. Offset Tuning Active (+150kHz)")
        ttk.Label(self.root, textvariable=self.status, font=("Consolas", 9)).pack(pady=4)

    def _toggle(self):
        if self.running: self._stop()
        else:            self._start()

    def _start(self):
        print(f"[BOOT] Starting SDR at {self.freq_var.get()}MHz with Offset Tuning...")
        try:
            self.sdr = RtlSdr()
            self.sdr.sample_rate = FS_IN
            # Tune 150kHz BELOW target to avoid DC spike
            self.sdr.center_freq = self.freq_var.get() * 1e6 - OFFSET_HZ
            self.sdr.gain = self.gain
        except Exception as e:
            self.status.set(f"SDR ERROR: {e}")
            return

        self.running = True
        self.demod = FMDemodPro()
        self.ring = AudioRingBuffer(FS_AUDIO * 5)
        self.btn.config(text="  STOP   ")
        
        # Pre-fill
        self.ring.push(np.zeros(FS_AUDIO // 2, dtype=np.float32))

        self.stream = sd.OutputStream(samplerate=FS_AUDIO, channels=1, blocksize=1024,
                                      dtype='float32', latency='low', callback=self._audio_cb)
        self.stream.start()

        threading.Thread(target=self._t_sdr, daemon=True).start()
        threading.Thread(target=self._t_dsp, daemon=True).start()
        self._gui_tick()

    def _stop(self):
        self.running = False
        self.btn.config(text="  START  ")
        try: self.stream.stop(); self.stream.close()
        except: pass
        try: self.sdr.close()
        except: pass

    def _t_sdr(self):
        while self.running:
            try:
                self.sdr.gain = self.gain
                iq = self.sdr.read_samples(CHUNK_IQ)
                if not self.q_iq.full(): self.q_iq.put(iq)
            except: self.running = False

    def _t_dsp(self):
        while self.running:
            try:
                iq = self.q_iq.get(timeout=1.0)
                audio = self.demod.process(iq)
                self.ring.push(audio * self.volume)
                if self.q_spec.empty(): self.q_spec.put(iq)
            except: pass

    def _audio_cb(self, outdata, frames, time, status):
        outdata[:, 0] = self.ring.pull(frames)

    def _gui_tick(self):
        if not self.running: return
        try:
            iq = self.q_spec.get_nowait()
            psd = 10 * np.log10(np.abs(np.fft.fftshift(np.fft.fft(iq, 1024)))**2 / 1024 + 1e-12)
            fr = np.linspace(-FS_IN/2000, FS_IN/2000, 1024)
            self.line.set_data(fr, psd)
            rms = self.demod.last_rms
            db = 20 * np.log10(rms + 1e-9)
            self.vu_lbl.config(text=f"VU: {db:+.1f}dBFS")
            self.canvas.draw_idle()
        except: pass
        self.root.after(100, self._gui_tick)


if __name__ == "__main__":
    root = tk.Tk()
    app = FMRadioApp(root)
    root.protocol("WM_DELETE_WINDOW", lambda: (app._stop(), root.destroy()))
    root.mainloop()
