"""
FM Live Player - Debug Edition
Prints status at every step so we can see exactly what fails.
"""
import sys
import traceback

print("[BOOT] Python OK, importing libraries...")
import numpy as np
import scipy.signal as signal
print("[BOOT] numpy/scipy OK")

try:
    from rtlsdr import RtlSdr
    print("[BOOT] rtlsdr OK")
except Exception as e:
    print(f"[BOOT] rtlsdr FAILED: {e}")
    sys.exit(1)

try:
    import sounddevice as sd
    print(f"[BOOT] sounddevice OK, default output: {sd.query_devices(kind='output')['name']}")
except Exception as e:
    print(f"[BOOT] sounddevice FAILED: {e}")
    sys.exit(1)

import tkinter as tk
from tkinter import ttk
import threading
import queue
import collections
print("[BOOT] tkinter OK")

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
print("[BOOT] matplotlib OK")
print("[BOOT] All imports successful. Starting GUI...")

FS_IN    = 240_000
FS_AUDIO = 48_000
DECIMATE = 5
CHUNK_IQ = 24_000


class FMDemod:
    def __init__(self):
        self.lpf_b  = signal.firwin(127, 15_000 / (FS_IN / 2.0))
        self.lpf_zi = np.real(signal.lfilter_zi(self.lpf_b, [1.0]).copy())
        tau   = 50e-6
        alpha = np.exp(-1.0 / (FS_AUDIO * tau))
        self.de_b  = np.array([1.0 - alpha])
        self.de_a  = np.array([1.0, -alpha])
        self.de_zi = np.zeros(1)
        # AGC: track RMS with slow filter
        self.rms_avg  = 0.1
        self.last_rms = 0.0   # expose for GUI display
        print("[DSP] FMDemod initialized OK")

    def process(self, iq):
        diff = iq[1:] * np.conj(iq[:-1])
        fm   = np.angle(diff).astype(np.float64)
        scaled_zi  = self.lpf_zi * (fm[0] if fm[0] != 0 else 1.0)
        fm_lpf, self.lpf_zi = signal.lfilter(self.lpf_b, [1.0], fm, zi=scaled_zi)
        self.lpf_zi = np.real(self.lpf_zi)
        audio = fm_lpf[::DECIMATE]
        audio, self.de_zi = signal.lfilter(self.de_b, self.de_a, audio, zi=self.de_zi)
        # AGC: target 0.5 (-6 dBFS) - strong enough to hear clearly
        rms = float(np.sqrt(np.mean(audio ** 2)) + 1e-9)
        self.rms_avg  = 0.98 * self.rms_avg + 0.02 * rms
        audio         = audio * (0.5 / self.rms_avg)
        self.last_rms = float(np.sqrt(np.mean(audio ** 2)))
        return audio.astype(np.float32)


class AudioRingBuffer:
    def __init__(self, capacity):
        self._buf  = collections.deque(maxlen=capacity)
        self._lock = threading.Lock()

    def push(self, samples):
        with self._lock:
            self._buf.extend(samples.tolist())

    def pull(self, n):
        with self._lock:
            avail = len(self._buf)
            if avail >= n:
                return np.array([self._buf.popleft() for _ in range(n)], dtype=np.float32)
            out = np.array([self._buf.popleft() for _ in range(avail)], dtype=np.float32)
            return np.concatenate([out, np.zeros(n - avail, dtype=np.float32)])

    @property
    def size(self):
        return len(self._buf)


class FMRadioApp:
    RING_CAP = FS_AUDIO * 3

    def __init__(self, root):
        self.root    = root
        self.gain    = 35.0
        self.volume  = 1.0
        self.running = False
        self.q_iq    = queue.Queue(maxsize=16)
        self.q_spec  = queue.Queue(maxsize=2)
        self.ring    = AudioRingBuffer(self.RING_CAP)
        self.demod   = FMDemod()
        self._build_ui()
        print("[GUI] UI built OK. App is ready.")

    def _build_ui(self):
        self.root.title("FM Baseband Forge")
        self.root.geometry("780x480")
        self.root.configure(bg="#0d0d1a")

        ctrl = tk.Frame(self.root, bg="#111122", bd=1, relief="sunken")
        ctrl.pack(fill="x", padx=8, pady=6)

        tk.Label(ctrl, text="Freq (MHz):", bg="#111122", fg="#cccccc",
                 font=("Consolas", 10)).grid(row=0, column=0, padx=6, pady=4)
        self.freq_var = tk.DoubleVar(value=94.3)
        tk.Entry(ctrl, textvariable=self.freq_var, width=7,
                 font=("Consolas", 11), bg="#1a1a2e", fg="#00bfff",
                 insertbackground="white").grid(row=0, column=1, padx=4)

        self.btn = tk.Button(ctrl, text="  START  ", command=self._toggle,
                             bg="#16213e", fg="#00bfff",
                             font=("Consolas", 10, "bold"),
                             activebackground="#0a3050", cursor="hand2")
        self.btn.grid(row=0, column=2, padx=12, pady=4)

        tk.Label(ctrl, text="RF Gain:", bg="#111122", fg="#cccccc",
                 font=("Consolas", 10)).grid(row=0, column=3, padx=6)
        self.gain_var = tk.DoubleVar(value=35.0)
        self.gain_lbl = tk.Label(ctrl, text="35 dB", bg="#111122", fg="#aaaaaa",
                                 font=("Consolas", 10), width=6)
        tk.Scale(ctrl, from_=0, to=49.6, variable=self.gain_var, orient="horizontal",
                 length=110, bg="#111122", fg="#00bfff", highlightthickness=0,
                 showvalue=False, troughcolor="#0d0d1a",
                 command=lambda v: [setattr(self, 'gain', float(v)),
                                    self.gain_lbl.config(text=f"{float(v):.0f} dB")]
                 ).grid(row=0, column=4)
        self.gain_lbl.grid(row=0, column=5, padx=4)

        tk.Label(ctrl, text="Vol:", bg="#111122", fg="#cccccc",
                 font=("Consolas", 10)).grid(row=0, column=6, padx=6)
        self.vol_var = tk.DoubleVar(value=2.0)  # default 2x
        tk.Scale(ctrl, from_=0, to=5.0, variable=self.vol_var, orient="horizontal",
                 length=90, bg="#111122", fg="#00bfff", highlightthickness=0,
                 showvalue=False, troughcolor="#0d0d1a",
                 command=lambda v: setattr(self, 'volume', float(v))
                 ).grid(row=0, column=7)

        self.vu_lbl = tk.Label(ctrl, text="VU: ---", bg="#111122", fg="#00ff88",
                               font=("Consolas", 9), width=10)
        self.vu_lbl.grid(row=0, column=8, padx=4)

        self.buf_lbl = tk.Label(ctrl, text="buf:--", bg="#111122", fg="#555555",
                               font=("Consolas", 9))
        self.buf_lbl.grid(row=0, column=9, padx=4)

        # spectrum
        self.fig = plt.Figure(figsize=(7.5, 2.7), facecolor="#0d0d1a")
        self.ax  = self.fig.add_subplot(111, facecolor="#0d0d1a")
        self.ax.tick_params(colors="#444")
        for sp in self.ax.spines.values(): sp.set_edgecolor("#222")
        self.sline, = self.ax.plot([], [], color="#00bfff", lw=0.9)
        self.ax.set_ylim(-70, 10)
        self.ax.set_xlim(-FS_IN/2e3, FS_IN/2e3)
        self.ax.set_xlabel("kHz offset", color="#444", fontsize=8)
        self.ax.set_title("-- MHz", color="#aaa", fontsize=10)
        self.fig.tight_layout()

        cv = FigureCanvasTkAgg(self.fig, master=self.root)
        cv.get_tk_widget().pack(fill="both", expand=True, padx=8, pady=4)
        self.canvas = cv

        self.status_var = tk.StringVar(value="Ready. Plug in RTL-SDR and press START.")
        tk.Label(self.root, textvariable=self.status_var, bg="#0d0d1a",
                 fg="#888888", font=("Consolas", 9)).pack(pady=2)

    def _toggle(self):
        print(f"[GUI] Button clicked. running={self.running}")
        if self.running:
            self._stop()
        else:
            self._start()

    def _start(self):
        print(f"[SDR] Opening device at {self.freq_var.get():.1f} MHz, gain={self.gain:.0f} dB")
        try:
            self.sdr = RtlSdr()
            self.sdr.sample_rate = FS_IN
            self.sdr.center_freq = self.freq_var.get() * 1e6
            self.sdr.gain        = self.gain
            print(f"[SDR] Opened OK. Actual gain={self.sdr.gain}")
        except Exception as e:
            msg = f"SDR Error: {e} -> Re-plug USB and retry."
            print(f"[SDR] {msg}")
            self.status_var.set(msg)
            return

        self.running = True
        self.demod   = FMDemod()
        self.ring    = AudioRingBuffer(self.RING_CAP)
        self.ring.push(np.zeros(FS_AUDIO // 2, dtype=np.float32))
        self.btn.config(text="  STOP   ")
        self.status_var.set(f"Receiving {self.freq_var.get():.1f} MHz ...")

        print("[Audio] Starting sounddevice callback stream...")
        try:
            self._stream = sd.OutputStream(
                samplerate=FS_AUDIO, channels=1, blocksize=1024,
                dtype='float32', latency='low', callback=self._audio_cb)
            self._stream.start()
            print("[Audio] Stream started OK")
        except Exception as e:
            print(f"[Audio] Stream failed: {e}")
            self.status_var.set(f"Audio error: {e}")
            self.running = False
            return

        threading.Thread(target=self._t_sdr, daemon=True, name="T-SDR").start()
        threading.Thread(target=self._t_dsp, daemon=True, name="T-DSP").start()
        print("[GUI] All threads launched. Starting GUI tick.")
        self._gui_tick()

    def _stop(self):
        print("[STOP] Stopping...")
        self.running = False
        self.btn.config(text="  START  ")
        self.status_var.set("Stopped.")
        try: self._stream.stop(); self._stream.close()
        except: pass
        try: self.sdr.close()
        except: pass
        print("[STOP] Done.")

    def _t_sdr(self):
        print("[T-SDR] Thread started")
        count = 0
        while self.running:
            try:
                self.sdr.gain = self.gain
                iq = self.sdr.read_samples(CHUNK_IQ)
                if self.q_iq.full():
                    try: self.q_iq.get_nowait()
                    except: pass
                self.q_iq.put(iq)
                count += 1
                if count % 10 == 0:
                    print(f"[T-SDR] {count} chunks read, q_iq size={self.q_iq.qsize()}")
            except Exception as e:
                print(f"[T-SDR] Error: {e}")
                self.status_var.set(f"SDR error: {e}")
                self._stop(); return
        print("[T-SDR] Thread exited")

    def _t_dsp(self):
        print("[T-DSP] Thread started")
        count = 0
        while self.running:
            try:
                iq    = self.q_iq.get(timeout=1.0)
                audio = self.demod.process(iq)
                audio = np.clip(audio * self.volume, -1.0, 1.0)
                self.ring.push(audio)
                count += 1
                if count % 10 == 0:
                    print(f"[T-DSP] {count} chunks processed, ring_buf={self.ring.size}")
                if self.q_spec.empty():
                    self.q_spec.put(iq)
            except queue.Empty:
                continue
            except Exception as e:
                print(f"[T-DSP] Error: {traceback.format_exc()}")
        print("[T-DSP] Thread exited")

    def _audio_cb(self, outdata, frames, time_info, status):
        if status:
            print(f"[Audio CB] Status: {status}")
        outdata[:, 0] = self.ring.pull(frames)

    def _gui_tick(self):
        if not self.running: return
        try:
            iq  = self.q_spec.get_nowait()
            psd = 10 * np.log10(
                np.abs(np.fft.fftshift(np.fft.fft(iq, 1024))) ** 2 / 1024 + 1e-10)
            frq = np.linspace(-FS_IN/2e3, FS_IN/2e3, 1024)
            self.sline.set_data(frq, psd)
            self.ax.set_title(
                f"{self.freq_var.get():.1f} MHz  |  Gain {self.gain:.0f} dB",
                color="#aaa", fontsize=10)
            rms = self.demod.last_rms
            db  = 20 * np.log10(rms + 1e-9)
            self.vu_lbl.config(text=f"VU: {db:+.1f}dBFS")
            self.buf_lbl.config(text=f"buf:{self.ring.size}")
            self.canvas.draw_idle()
        except queue.Empty:
            pass
        self.root.after(150, self._gui_tick)


if __name__ == "__main__":
    try:
        root = tk.Tk()
        app  = FMRadioApp(root)
        root.protocol("WM_DELETE_WINDOW", lambda: (app._stop(), root.destroy()))
        print("[MAIN] Entering mainloop...")
        root.mainloop()
        print("[MAIN] Mainloop exited normally.")
    except Exception:
        print(f"[FATAL] Unhandled exception:\n{traceback.format_exc()}")
