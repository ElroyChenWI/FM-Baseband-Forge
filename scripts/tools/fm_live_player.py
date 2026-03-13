"""
FM Live Player - GNU Radio Standard Architecture
DSP Chain: IQ(250k) -> LPF+Decimate(50k) -> FM Demod -> LPF 15k -> De-emphasis -> Resample(48k) -> Audio
Reference: GNU Radio FM Receiver, rtl_fm
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

FS_IN      = 1_140_000   # SDR sample rate (rtl_fm default: 1.14 MSPS)
FS_FM      = 228_000     # After 5x decimate (FM demod rate)
FS_AUDIO   = 48_000      # Output audio rate
DECIMATE_1 = 5           # FS_IN -> FS_FM
DECIMATE_2 = FS_FM // FS_AUDIO   # FS_FM -> FS_AUDIO = 4.75 -> use resample

class FMReceiver:
    """GNU Radio style FM receive chain."""

    def __init__(self):
        # Stage 1: Anti-alias LPF before decimation (100kHz cutoff)
        self.lpf1_taps = signal.firwin(128, 100_000 / (FS_IN / 2))
        self.lpf1_state = np.zeros(len(self.lpf1_taps) - 1, dtype=complex)

        # Stage 2: Audio LPF after demod (15kHz cutoff, standard voice+music)
        self.lpf2_taps = signal.firwin(64, 15_000 / (FS_FM / 2))
        self.lpf2_state = np.zeros(len(self.lpf2_taps) - 1)

        # Stage 3: De-emphasis IIR (50us for Asia/Europe, 75us for US)
        # H(z) = (1-alpha) / (1 - alpha*z^-1), alpha = exp(-1/(fs*tau))
        tau = 50e-6
        alpha = np.exp(-1.0 / (FS_FM * tau))
        self.deemph_b = np.array([1 - alpha])
        self.deemph_a = np.array([1, -alpha])
        self.deemph_zi = np.array([0.0])

        # Stage 4: AGC
        self.agc_ref   = 0.5
        self.agc_gain  = 1.0
        self.agc_decay = 0.99

    def process(self, iq_samples: np.ndarray, vol: float) -> np.ndarray:
        # --- Stage 1: Anti-alias LPF + Decimate 5x ---
        iq_lpf, self.lpf1_state = signal.lfilter(self.lpf1_taps, 1.0, iq_samples, zi=self.lpf1_state)
        iq_dec = iq_lpf[::DECIMATE_1]                     # 1.14M -> 228k

        # --- Stage 2: FM Discriminator (standard arctan differentiator) ---
        diff = iq_dec[1:] * np.conj(iq_dec[:-1])
        fm   = np.angle(diff)                             # unit: radians/sample

        # Normalize by fs/2pi to get Hz: fm_hz = fm * (FS_FM / (2*pi))
        # But for audio, we just need relative amplitude, skip constant factor.

        # --- Stage 3: Audio LPF (0-15kHz) ---
        audio_lpf, self.lpf2_state = signal.lfilter(self.lpf2_taps, 1.0, fm, zi=self.lpf2_state)

        # --- Stage 4: De-emphasis (IIR, scipy lfilter is vectorized/fast) ---
        audio_de, self.deemph_zi = signal.lfilter(self.deemph_b, self.deemph_a, audio_lpf, zi=self.deemph_zi)

        # --- Stage 5: Resample 228k -> 48k ---
        # Use polyphase for speed (scipy.signal.resample_poly)
        audio_out = signal.resample_poly(audio_de, FS_AUDIO, FS_FM)

        # --- Stage 6: AGC (slow attack, fast decay) ---
        peak = np.max(np.abs(audio_out)) + 1e-9
        if peak * self.agc_gain > self.agc_ref:
            self.agc_gain = self.agc_ref / peak  # fast attack
        else:
            self.agc_gain = self.agc_gain * self.agc_decay + (self.agc_ref / peak) * (1 - self.agc_decay)  # slow recovery
        audio_out *= self.agc_gain

        # --- Stage 7: Volume + safety clip ---
        audio_out = np.clip(audio_out * vol, -1.0, 1.0)

        return audio_out.astype(np.float32)


class FMRadioApp:
    def __init__(self, root):
        self.root = root
        self.root.title("FM Baseband Forge - Live Radio")
        self.root.geometry("820x550")
        self.root.configure(bg="#1a1a2e")

        self.freq_mhz = 94.3
        self.sdr_gain = 30.0
        self.volume   = 0.8
        self.running  = False
        # Three separate queues: SDR -> DSP -> Audio
        self.iq_queue    = queue.Queue(maxsize=8)   # raw IQ
        self.audio_queue = queue.Queue(maxsize=16)  # processed float32 audio
        self.spec_queue  = queue.Queue(maxsize=2)   # for spectrum display only

        self.receiver = FMReceiver()
        self._setup_ui()

    def _setup_ui(self):
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TLabel", background="#1a1a2e", foreground="#e0e0e0", font=("Consolas", 10))
        style.configure("TButton", background="#16213e", foreground="#00d2ff", font=("Consolas", 10, "bold"))
        style.configure("TScale", background="#1a1a2e")
        style.configure("TLabelframe", background="#1a1a2e", foreground="#00d2ff")
        style.configure("TLabelframe.Label", background="#1a1a2e", foreground="#00d2ff")

        ctrl = ttk.LabelFrame(self.root, text="CONTROL PANEL")
        ctrl.pack(fill="x", padx=12, pady=6)

        ttk.Label(ctrl, text="Frequency (MHz)").grid(row=0, column=0, padx=8, pady=4)
        self.freq_var = tk.DoubleVar(value=94.3)
        ttk.Entry(ctrl, textvariable=self.freq_var, width=8, font=("Consolas", 11)).grid(row=0, column=1, padx=4)

        self.btn = ttk.Button(ctrl, text="[ START  ]", command=self._toggle)
        self.btn.grid(row=0, column=2, padx=14)

        ttk.Label(ctrl, text="RF Gain (dB)").grid(row=0, column=3, padx=8)
        self.gain_var = tk.DoubleVar(value=30.0)
        ttk.Scale(ctrl, from_=0, to=49.6, variable=self.gain_var, orient="horizontal", length=120, command=lambda v: setattr(self, 'sdr_gain', float(v))).grid(row=0, column=4, padx=4)
        self.gain_label = ttk.Label(ctrl, text="30 dB")
        self.gain_label.grid(row=0, column=5)

        ttk.Label(ctrl, text="Volume").grid(row=0, column=6, padx=8)
        self.vol_var = tk.DoubleVar(value=0.8)
        ttk.Scale(ctrl, from_=0.0, to=1.5, variable=self.vol_var, orient="horizontal", length=100, command=lambda v: setattr(self, 'volume', float(v))).grid(row=0, column=7, padx=4)

        # Spectrum Plot
        self.fig = plt.Figure(figsize=(8, 3), facecolor="#0d0d1a")
        self.ax  = self.fig.add_subplot(111)
        self.ax.set_facecolor("#0d0d1a")
        self.ax.tick_params(colors="#888888")
        for spine in self.ax.spines.values(): spine.set_edgecolor("#333")
        self.line, = self.ax.plot([], [], color="#00d2ff", lw=0.8)
        self.ax.set_ylim(-80, 10)
        self.ax.set_xlim(-FS_IN/2000, FS_IN/2000)
        self.ax.set_xlabel("kHz", color="#888")
        self.ax.set_ylabel("dBFS", color="#888")
        self.ax.set_title(f"Baseband Spectrum | -- MHz", color="#cccccc")
        canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        canvas.get_tk_widget().pack(fill="both", expand=True, padx=12, pady=4)
        self.canvas = canvas

        self.status_var = tk.StringVar(value="Ready. Plug in RTL-SDR and press START.")
        ttk.Label(self.root, textvariable=self.status_var, font=("Consolas", 9)).pack(pady=2)

    def _toggle(self):
        if not self.running:
            self._start()
        else:
            self._stop()

    def _start(self):
        try:
            self.sdr = RtlSdr()
            self.sdr.sample_rate = FS_IN
            self.sdr.center_freq = self.freq_var.get() * 1e6
            self.sdr.gain = self.sdr_gain
        except Exception as e:
            self.status_var.set(f"ERROR: {e}  ->  Unplug + re-plug SDR, then retry.")
            return

        self.running = True
        self.btn.config(text="[ STOP   ]")
        self.receiver = FMReceiver()  # reset DSP state on new stream

        # Launch 3 dedicated threads: SDR read / DSP process / Audio output
        threading.Thread(target=self._sdr_thread,   daemon=True, name="SDR").start()
        threading.Thread(target=self._dsp_thread,   daemon=True, name="DSP").start()
        threading.Thread(target=self._audio_thread, daemon=True, name="Audio").start()
        self._gui_loop()

    def _stop(self):
        self.running = False
        self.btn.config(text="[ START  ]")
        self.status_var.set("Stopped.")
        if hasattr(self, 'sdr'):
            try: self.sdr.close()
            except: pass

    def _sdr_thread(self):
        # Small chunk = low latency. 0.05s per read at 1.14 MSPS
        CHUNK = int(FS_IN * 0.05)
        while self.running:
            try:
                self.sdr.gain = self.sdr_gain
                samples = self.sdr.read_samples(CHUNK)
                # If IQ queue is full, drop oldest. Never block SDR reads.
                if self.iq_queue.full():
                    try: self.iq_queue.get_nowait()
                    except: pass
                self.iq_queue.put(samples)
            except Exception as e:
                self.status_var.set(f"SDR Error: {e}")
                self._stop(); break

    def _dsp_thread(self):
        # Only responsibility: read IQ, apply DSP chain, push audio frames
        while self.running:
            try:
                iq = self.iq_queue.get(timeout=0.5)
                audio = self.receiver.process(iq, self.volume)
                # Push to audio queue. Do not drop audio frames or we get glitches.
                self.audio_queue.put(audio)  # blocks if full (backpressure)
                # Push IQ snapshot for spectrum (best-effort, drops ok)
                if self.spec_queue.empty():
                    self.spec_queue.put(iq)
            except queue.Empty:
                continue
            except Exception as e:
                print(f"DSP error: {e}")

    def _audio_thread(self):
        # Only responsibility: drain audio queue and write to sounddevice.
        # NO computation here whatsoever.
        with sd.OutputStream(samplerate=FS_AUDIO, channels=1, blocksize=2048,
                             latency='low', dtype='float32') as stream:
            while self.running:
                try:
                    audio_chunk = self.audio_queue.get(timeout=0.5)
                    stream.write(audio_chunk)
                except queue.Empty:
                    # Write silence to prevent sounddevice underrun
                    stream.write(np.zeros(2048, dtype='float32'))
                except Exception as e:
                    print(f"Audio write error: {e}")

    def _gui_loop(self):
        if not self.running: return
        try:
            iq = self.spec_queue.get_nowait()
            psd = 10 * np.log10(np.abs(np.fft.fftshift(np.fft.fft(iq, 2048)))**2 / 2048 + 1e-12)
            freqs = np.linspace(-FS_IN/2000, FS_IN/2000, 2048)
            self.line.set_data(freqs, psd)
            freq_mhz = self.freq_var.get()
            self.ax.set_title(f"Baseband Spectrum | {freq_mhz:.1f} MHz  |  Gain: {self.sdr_gain:.0f} dB", color="#cccccc")
            self.gain_label.config(text=f"{self.sdr_gain:.0f} dB")
            self.canvas.draw_idle()
        except queue.Empty:
            pass
        self.root.after(100, self._gui_loop)  # 10 fps is enough for spectrum


if __name__ == "__main__":
    root = tk.Tk()
    app  = FMRadioApp(root)
    root.protocol("WM_DELETE_WINDOW", lambda: (app._stop(), root.destroy()))
    root.mainloop()
