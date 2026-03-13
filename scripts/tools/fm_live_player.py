"""
FM Live Player - Minimal Reliable Architecture
FS=240kHz -> FM Demod -> LPF 15kHz -> Decimate 5x -> 48kHz Audio
Three decoupled threads: SDR / DSP / Audio
"""
import numpy as np
import scipy.signal as signal
from rtlsdr import RtlSdr
import sounddevice as sd
import tkinter as tk
from tkinter import ttk
import threading
import queue
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# --- Constants ---
FS_IN    = 240_000   # SDR rate: 240kHz / 5 = 48kHz audio. Simple integer ratio
FS_AUDIO = 48_000    # Audio output rate
DECIMATE = 5         # 240k / 5 = 48k
CHUNK    = 48_000    # 0.2s per IQ chunk


class FMReceiver:
    """Minimal correct FM DSP chain."""

    def __init__(self):
        # Anti-alias LPF for audio band (15kHz cutoff at FS_IN)
        self.lpf_taps  = signal.firwin(128 + 1, 15_000 / (FS_IN / 2.0))
        self.lpf_state = signal.lfilter_zi(self.lpf_taps, 1.0)

        # De-emphasis: 50us IIR
        tau   = 50e-6
        alpha = np.exp(-1.0 / (FS_AUDIO * tau))  # at audio rate after decimate
        self.b_de = np.array([1.0 - alpha])
        self.a_de = np.array([1.0, -alpha])
        self.zi_de = np.array([0.0])

        # AGC state
        self.agc_level = 0.3

    def process(self, iq: np.ndarray, volume: float) -> np.ndarray:
        # 1. FM discriminator
        diff = iq[1:] * np.conj(iq[:-1])
        fm   = np.angle(diff).astype(np.float32)

        # 2. Low-pass filter (remove pilot, RDS, and other sub-carriers)
        fm_lpf, self.lpf_state = signal.lfilter(
            self.lpf_taps, 1.0, fm, zi=self.lpf_state * fm[0])

        # 3. Decimate 5x: 240k -> 48k
        audio = fm_lpf[::DECIMATE]

        # 4. De-emphasis (vectorized IIR)
        audio, self.zi_de = signal.lfilter(
            self.b_de, self.a_de, audio, zi=self.zi_de)

        # 5. AGC: track RMS, normalize
        rms = np.sqrt(np.mean(audio ** 2)) + 1e-9
        self.agc_level = 0.95 * self.agc_level + 0.05 * rms
        audio = audio * (0.3 / self.agc_level)

        # 6. Volume + hard clip
        audio = np.clip(audio * volume, -1.0, 1.0)
        return audio.astype(np.float32)


class FMRadioApp:
    def __init__(self, root):
        self.root    = root
        self.root.title("FM Baseband Forge")
        self.root.geometry("800x500")
        self.root.configure(bg="#0d0d1a")

        self.gain    = 35.0
        self.volume  = 1.0
        self.running = False
        self.freq_hz = 94.3e6

        # Three decoupled queues
        self.q_iq    = queue.Queue(maxsize=12)
        self.q_audio = queue.Queue(maxsize=24)
        self.q_spec  = queue.Queue(maxsize=2)

        self.receiver = FMReceiver()
        self._build_ui()

    def _build_ui(self):
        s = ttk.Style()
        s.theme_use("clam")
        for w in ("TLabel", "TFrame"):
            s.configure(w, background="#0d0d1a", foreground="#cccccc", font=("Consolas", 10))
        s.configure("TLabelframe",       background="#0d0d1a", foreground="#00bfff")
        s.configure("TLabelframe.Label", background="#0d0d1a", foreground="#00bfff", font=("Consolas", 10, "bold"))
        s.configure("TButton",  background="#16213e", foreground="#00bfff", font=("Consolas", 10, "bold"))
        s.configure("Hscale.TScale", background="#0d0d1a")

        ctrl = ttk.LabelFrame(self.root, text=" CONTROLS ", padding=6)
        ctrl.pack(fill="x", padx=10, pady=6)

        # Frequency
        ttk.Label(ctrl, text="Freq (MHz)").grid(row=0, column=0, padx=6)
        self.freq_var = tk.DoubleVar(value=94.3)
        ttk.Entry(ctrl, textvariable=self.freq_var, width=7, font=("Consolas", 11)).grid(row=0, column=1, padx=4)

        # Start/Stop
        self.btn = ttk.Button(ctrl, text="  START  ", command=self._toggle)
        self.btn.grid(row=0, column=2, padx=14)

        # Gain
        ttk.Label(ctrl, text="RF Gain").grid(row=0, column=3, padx=6)
        self.gain_var = tk.DoubleVar(value=35.0)
        self.gain_lbl = ttk.Label(ctrl, text="35 dB", width=6)
        ttk.Scale(ctrl, from_=0, to=49.6, variable=self.gain_var, length=110,
                  orient="horizontal",
                  command=lambda v: [setattr(self, 'gain', float(v)),
                                     self.gain_lbl.config(text=f"{float(v):.0f} dB")]
                  ).grid(row=0, column=4, padx=2)
        self.gain_lbl.grid(row=0, column=5, padx=2)

        # Volume
        ttk.Label(ctrl, text="Volume").grid(row=0, column=6, padx=6)
        self.vol_var = tk.DoubleVar(value=1.0)
        ttk.Scale(ctrl, from_=0, to=2.0, variable=self.vol_var, length=90,
                  orient="horizontal",
                  command=lambda v: setattr(self, 'volume', float(v))
                  ).grid(row=0, column=7, padx=2)

        # Spectrum
        self.fig = plt.Figure(figsize=(7.6, 2.8), facecolor="#0d0d1a")
        self.ax  = self.fig.add_subplot(111, facecolor="#0d0d1a")
        self.ax.tick_params(colors="#555")
        for sp in self.ax.spines.values(): sp.set_edgecolor("#222")
        self.sline, = self.ax.plot([], [], color="#00bfff", lw=0.9)
        self.ax.set_ylim(-70, 10)
        self.ax.set_xlim(-FS_IN/2000, FS_IN/2000)
        self.ax.set_xlabel("kHz offset", color="#555", fontsize=8)
        self.ax.set_title("-- MHz", color="#aaa", fontsize=10)
        self.fig.tight_layout()

        cv = FigureCanvasTkAgg(self.fig, master=self.root)
        cv.get_tk_widget().pack(fill="both", expand=True, padx=10, pady=4)
        self.canvas = cv

        self.status = tk.StringVar(value="Plug in RTL-SDR and press START.")
        ttk.Label(self.root, textvariable=self.status, font=("Consolas", 9)).pack(pady=2)

    def _toggle(self):
        if self.running: self._stop()
        else:            self._start()

    def _start(self):
        try:
            self.sdr = RtlSdr()
            self.sdr.sample_rate = FS_IN
            self.sdr.center_freq = self.freq_var.get() * 1e6
            self.sdr.gain        = self.gain
        except Exception as e:
            self.status.set(f"SDR Error: {e} -> Re-plug device and retry.")
            return

        self.running = True
        self.receiver = FMReceiver()
        self.btn.config(text="  STOP   ")
        self.status.set(f"Receiving {self.freq_var.get():.1f} MHz ...")

        threading.Thread(target=self._t_sdr,   daemon=True, name="T-SDR").start()
        threading.Thread(target=self._t_dsp,   daemon=True, name="T-DSP").start()
        threading.Thread(target=self._t_audio, daemon=True, name="T-Audio").start()
        self._gui_tick()

    def _stop(self):
        self.running = False
        self.btn.config(text="  START  ")
        self.status.set("Stopped.")
        try: self.sdr.close()
        except: pass

    # ---- Thread: SDR read ----
    def _t_sdr(self):
        while self.running:
            try:
                self.sdr.gain = self.gain       # live update
                iq = self.sdr.read_samples(CHUNK)
                if self.q_iq.full():
                    try: self.q_iq.get_nowait()  # drop oldest IQ, never block
                    except: pass
                self.q_iq.put(iq)
            except Exception as e:
                self.status.set(f"SDR read error: {e}")
                self._stop(); return

    # ---- Thread: DSP (FM demod) ----
    def _t_dsp(self):
        while self.running:
            try:
                iq    = self.q_iq.get(timeout=1.0)
                audio = self.receiver.process(iq, self.volume)
                self.q_audio.put(audio)           # block if audio consumer is slow
                if self.q_spec.empty():
                    self.q_spec.put(iq)
            except queue.Empty:
                continue
            except Exception as e:
                print(f"DSP err: {e}")

    # ---- Thread: Audio output (zero computation here) ----
    def _t_audio(self):
        # Use low latency; blocksize matches our chunk: 48k/5=9600 samples
        with sd.OutputStream(samplerate=FS_AUDIO, channels=1,
                             blocksize=CHUNK // DECIMATE,
                             latency='low', dtype='float32') as s:
            silence = np.zeros(CHUNK // DECIMATE, dtype='float32')
            while self.running:
                try:
                    chunk = self.q_audio.get(timeout=0.3)
                    s.write(chunk)
                except queue.Empty:
                    s.write(silence)   # keep stream alive, no pop
                except Exception as e:
                    print(f"Audio err: {e}")

    # ---- GUI update (main thread, low priority) ----
    def _gui_tick(self):
        if not self.running: return
        try:
            iq   = self.q_spec.get_nowait()
            psd  = 10 * np.log10(
                np.abs(np.fft.fftshift(np.fft.fft(iq, 1024))) ** 2 / 1024 + 1e-10)
            freq = np.linspace(-FS_IN/2000, FS_IN/2000, 1024)
            self.sline.set_data(freq, psd)
            self.ax.set_title(f"{self.freq_var.get():.1f} MHz  |  Gain {self.gain:.0f} dB",
                              color="#aaa", fontsize=10)
            self.canvas.draw_idle()
        except queue.Empty:
            pass
        self.root.after(150, self._gui_tick)   # ~7fps, light on CPU


if __name__ == "__main__":
    root = tk.Tk()
    app  = FMRadioApp(root)
    root.protocol("WM_DELETE_WINDOW", lambda: (app._stop(), root.destroy()))
    root.mainloop()
