# FM Baseband Forge: Detailed Usage Guide (EN)

This guide provides a step-by-step walkthrough for capturing and decoding FM broadcast signals using RTL-SDR hardware and Python.

## 1. Environment Setup

Before starting, ensure your Python environment and hardware drivers are properly configured.

### Prerequisites & Libraries
Run the following command in your terminal:
```powershell
pip install pyrtlsdr pyrtlsdrlib numpy scipy matplotlib
```
*Note: `pyrtlsdrlib` is included to automatically handle the `librtlsdr.dll` driver dependency on Windows systems.*

### Hardware Connection
1. Insert the RTL-SDR receiver into a USB port.
2. Connect the FM antenna. For optimal results, place the antenna near a window to maximize the Signal-to-Noise Ratio (SNR).

---

## 2. Step 1: Exploration & Detection

The first task is to identify the strongest FM station in your local environment.

### Run Spectrum Profiler
Use `fm_profiler.py` to scan the 88-108 MHz band.
Command:
```powershell
python 1_rf_reconnaissance/fm_profiler.py
```
**Expected Outcome:** The script will generate a spectrum energy graph. Note down the frequency with the highest peak (e.g., 94.3 MHz).

---

## 3. Step 2: Audio Analysis & Partitioning

Once a frequency is selected, we begin deconstructing the analog audio components.

### Extract Harmonics (19k/38k/57k)
Command:
```powershell
python 4_spectrum_analytics/extract_harmonics.py
```
This script isolates the 19kHz Pilot Tone, 38kHz Stereo Difference (L-R), and 57kHz RDS data layer to verify the broadcast's spectral topology.

### Stereo Recovery & De-emphasis Experiment
Command:
```powershell
python 3_analog_demodulation/fm_master_comparison.py
```
This generates four comparative audio files for the target frequency:
1. Mono (Raw)
2. Mono (50μs De-emphasis filter applied)
3. Stereo (L/R separation)
4. Stereo (De-emphasis applied, optimal quality)
Output files are stored in the `data/comparison/` directory.

---

## 4. Step 3: RDS Data Decoding

Finally, attempt to extract digital text information from the baseband signal.

### Run the Final RDS Decoder
Command:
```powershell
python 5_digital_protocol_decoder/rds_final_decoder.py
```
This script initializes the four-stage digital signal processing (DSP) pipeline:
1. **Costas Loop**: Recovers carrier phase.
2. **Bit-Sync**: Aligns bit-sampling timing.
3. **Differential Decode**: Reverses the differential encoding.
4. **Frame Sync (Syndrome)**: Locks the 26-bit block boundaries.

**Debugging Tips:**
* If `🟢 [SYNC]` appears but text is garbled: Signal strength is too low, leading to CRC failures.
* No PI code found: Reposition the antenna or check the frequency accuracy.

---

## 5. Common Troubleshooting

### Access Denied (LIBUSB_ERROR_ACCESS)
This occurs when the hardware is locked by a previous process.
1. Terminate all Python terminals/sessions.
2. Unplug and re-plug the SDR device.

### File Path Errors
Ensure you are executing commands from the project root directory using relative paths: `python scripts/tools/xxx.py`.
