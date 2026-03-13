"""
FM Live Player - Callback Audio Architecture
- T-SDR:  read IQ samples -> q_iq
- T-DSP:  IQ -> FM demod -> append to ring buffer
- Audio:  sounddevice callback pulls from ring buffer at hardware rate
- GUI:    spectrum display on main thread
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

FS_IN    = 240_000
FS_AUDIO = 48_000
DECIMATE = 5            # 240k / 5 = 48k
CHUNK_IQ = 24_000       # 0.1s IQ chunk at 240kHz


class FMDemod:
    """Stateful FM demodulator."""

    def __init__(self):
        # Audio LPF: 15kHz at 240kHz rate (before decimation)
        self.lpf_b = signal.firwin(127, 15_000 / (FS_IN / 2.0))
        self.lpf_zi = signal.lfilter_zi(self.lpf_b, [1.0])
        self.lpf_zi = np.real(self.lpf_zi.copy())

        # De-emphasis 50us at 48kHz audio rate
        tau   = 50e-6
        alpha = np.exp(-1.0 / (FS_AUDIO * tau))
        self.de_b  = np.array([1.0 - alpha])
        self.de_a  = np.array([1.0, -alpha])
        self.de_zi = np.zeros(1)

        # AGC: track RMS with slow filter
        self.rms_avg = 0.1

    def process(self, iq: np.ndarray) -> np.ndarray:
        # FM discriminator
        diff = iq[1:] * np.conj(iq[:-1])
        fm   = np.angle(diff).astype(np.float64)

        # LPF to remove sub-carriers
        scaled_zi = self.lpf_zi * fm[0]
        fm_lpf, self.lpf_zi = signal.lfilter(self.lpf_b, [1.0], fm, zi=scaled_zi)
        self.lpf_zi = np.real(self.lpf_zi)

        # Decimate 5x
        audio = fm_lpf[::DECIMATE]

        # De-emphasis
        audio, self.de_zi = signal.lfilter(self.de_b, self.de_a, audio, zi=self.de_zi)

        # AGC
        rms = float(np.sqrt(np.mean(audio ** 2)) + 1e-9)
        self.rms_avg = 0.98 * self.rms_avg + 0.02 * rms
        audio = audio * (0.25 / self.rms_avg)

        return audio.astype(np.float32)


class AudioRingBuffer:
    """Thread-safe ring buffer for audio samples."""

    def __init__(self, capacity: int):
        self._buf  = collections.deque(maxlen=capacity)
        self._lock = threading.Lock()

    def push(self, samples: np.ndarray):
        with self._lock:
            self._buf.extend(samples.tolist())

    def pull(self, n: int) -> np.ndarray:
        with self._lock:
            available = len(self._buf)
            if available >= n:
                return np.array([self._buf.popleft() for _ in range(n)], dtype=np.float32)
            else:
                # Return what we have + silence padding
                aud = np.array([self._buf.popleft() for _ in range(available)], dtype=np.float32)
                return np.concatenate([aud, np.zeros(n - available, dtype=np.float32)])

    @property
    def size(self) -> int:
        return len(self._buf)


class FMRadioApp:
    RING_CAP = FS_AUDIO * 3   # 3 seconds of audio headroom

    def __init__(self, root):
        self.root    = root
        self.root.title("FM Baseband Forge - Live Player")
        self.root.geometry("820x500")
        self.root.configure(bg="#0d0d1a")

        self.gain    = 35.0
        self.volume  = 1.0
        self.running = False

        self.q_iq   = queue.Queue(maxsize=16)
        self.q_spec = queue.Queue(maxsize=2)
        self.ring   = AudioRingBuffer(self.RING_CAP)
        self.demod  = FMDemod()

        self._build_ui()

    # ---------------------------------------------------------------- UI
    def _build_ui(self):
        s = ttk.Style(); s.theme_use("clam")
        bg, fg = "#0d0d1a", "#cccccc"
        acc = "#00bfff"
        s.configure("TLabel",          background=bg, foreground=fg, font=("Consolas", 10))
        s.configure("TLabelframe",     background=bg, foreground=acc)
        s.configure("TLabelframe.Label", background=bg, foreground=acc, font=("Consolas", 10, "bold"))
        s.configure("TButton",         background="#16213e", foreground=acc, font=("Consolas", 10, "bold"))

        ctrl = ttk.LabelFrame(self.root, text=" CONTROL ", padding=6)
        ctrl.pack(fill="x", padx=10, pady=6)

        ttk.Label(ctrl, text="Freq (MHz)").grid(row=0, column=0, padx=6)
        self.freq_var = tk.DoubleVar(value=94.3)
        ttk.Entry(ctrl, textvariable=self.freq_var, width=7,
                  font=("Consolas", 11)).grid(row=0, column=1, padx=4)

        self.btn = ttk.Button(ctrl, text="  START  ", command=self._toggle)
        self.btn.grid(row=0, column=2, padx=14)

        ttk.Label(ctrl, text="RF Gain").grid(row=0, column=3, padx=6)
        self.gain_var = tk.DoubleVar(value=35.0)
        self.gain_lbl = ttk.Label(ctrl, text="35 dB", width=6)
        ttk.Scale(ctrl, from_=0, to=49.6, variable=self.gain_var, length=110,
                  orient="horizontal",
                  command=lambda v: [setattr(self, 'gain', float(v)),
                                     self.gain_lbl.config(text=f"{float(v):.0f} dB")]
                  ).grid(row=0, column=4)
        self.gain_lbl.grid(row=0, column=5, padx=4)

        ttk.Label(ctrl, text="Volume").grid(row=0, column=6, padx=6)
        self.vol_var = tk.DoubleVar(value=1.0)
        ttk.Scale(ctrl, from_=0, to=2.0, variable=self.vol_var, length=100,
                  orient="horizontal",
                  command=lambda v: setattr(self, 'volume', float(v))
                  ).grid(row=0, column=7)

        self.buf_lbl = ttk.Label(ctrl, text="buf: --")
        self.buf_lbl.grid(row=0, column=8, padx=8)

        # Spectrum
        self.fig = plt.Figure(figsize=(7.8, 2.8), facecolor="#0d0d1a")
        self.ax  = self.fig.add_subplot(111, facecolor="#0d0d1a")
        self.ax.tick_params(colors="#555")
        for sp in self.ax.spines.values(): sp.set_edgecolor("#222")
        self.sline, = self.ax.plot([], [], color="#00bfff", lw=0.9)
        self.ax.set_ylim(-70, 10)
        self.ax.set_xlim(-FS_IN/2e3, FS_IN/2e3)
        self.ax.set_xlabel("kHz offset", color="#555", fontsize=8)
        self.ax.set_title("-- MHz", color="#aaa", fontsize=10)
        self.fig.tight_layout()
        cv = FigureCanvasTkAgg(self.fig, master=self.root)
        cv.get_tk_widget().pack(fill="both", expand=True, padx=10, pady=4)
        self.canvas = cv

        self.status = tk.StringVar(value="Ready. Plug in RTL-SDR and press START.")
        ttk.Label(self.root, textvariable=self.status, font=("Consolas", 9)).pack(pady=2)

    # ---------------------------------------------------------------- Control
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
            self.status.set(f"SDR Error: {e}  ->  Re-plug USB and retry.")
            return

        self.running = True
        self.demod   = FMDemod()
        self.ring    = AudioRingBuffer(self.RING_CAP)
        self.btn.config(text="  STOP   ")
        self.status.set(f"Receiving {self.freq_var.get():.1f} MHz ...")

        # Pre-fill ring: 0.5s silence so callback won't starve on startup
        self.ring.push(np.zeros(FS_AUDIO // 2, dtype=np.float32))

        # Start audio output with hardware callback
        self._stream = sd.OutputStream(
            samplerate=FS_AUDIO,
            channels=1,
            blocksize=1024,          # small blocks = low latency
            dtype='float32',
            callback=self._audio_cb
        )
        self._stream.start()

        threading.Thread(target=self._t_sdr, daemon=True, name="T-SDR").start()
        threading.Thread(target=self._t_dsp, daemon=True, name="T-DSP").start()
        self._gui_tick()

    def _stop(self):
        self.running = False
        self.btn.config(text="  START  ")
        self.status.set("Stopped.")
        try: self._stream.stop(); self._stream.close()
        except: pass
        try: self.sdr.close()
        except: pass

    # ---------------------------------------------------------------- Threads
    def _t_sdr(self):
        while self.running:
            try:
                self.sdr.gain = self.gain
                iq = self.sdr.read_samples(CHUNK_IQ)
                if self.q_iq.full():
                    try: self.q_iq.get_nowait()
                    except: pass
                self.q_iq.put(iq)
            except Exception as e:
                self.status.set(f"SDR read err: {e}")
                self._stop(); return

    def _t_dsp(self):
        while self.running:
            try:
                iq    = self.q_iq.get(timeout=1.0)
                audio = self.demod.process(iq)
                audio = np.clip(audio * self.volume, -1.0, 1.0)
                self.ring.push(audio)
                if self.q_spec.empty():
                    self.q_spec.put(iq)
            except queue.Empty:
                continue
            except Exception as e:
                print(f"DSP err: {e}")

    # ---------------------------------------------------------------- Audio callback (runs on sounddevice's private thread)
    def _audio_cb(self, outdata, frames, time_info, status):
        chunk = self.ring.pull(frames)
        outdata[:, 0] = chunk

    # ---------------------------------------------------------------- GUI
    def _gui_tick(self):
        if not self.running: return
        try:
            iq  = self.q_spec.get_nowait()
            psd = 10 * np.log10(
                np.abs(np.fft.fftshift(np.fft.fft(iq, 1024))) ** 2 / 1024 + 1e-10)
            frq = np.linspace(-FS_IN/2e3, FS_IN/2e3, 1024)
            self.sline.set_data(frq, psd)
            self.ax.set_title(
                f"{self.freq_var.get():.1f} MHz  |  Gain {self.gain:.0f} dB  |  buf {self.ring.size}",
                color="#aaa", fontsize=10)
            self.buf_lbl.config(text=f"buf: {self.ring.size}")
            self.canvas.draw_idle()
        except queue.Empty:
            pass
        self.root.after(150, self._gui_tick)


if __name__ == "__main__":
    root = tk.Tk()
    app  = FMRadioApp(root)
    root.protocol("WM_DELETE_WINDOW", lambda: (app._stop(), root.destroy()))
    root.mainloop()
