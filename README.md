# BasebandForge: Demystifying the RF Physical Layer

BasebandForge is a ground-up Software Defined Radio (SDR) DSP engine built entirely in Python. Rejecting black-box frameworks like GNU Radio, this project directly implements the mathematical core of telecommunications to bridge the gap between raw electromagnetic waves and digital protocols. 

---

## Core Architectural Highlights
* **RF to Baseband**: Processes 2.4 MSPS raw I/Q streams down to a 240 kSPS composite multiplex (MPX).
* **Precision DSP Filtering**: Implements custom high-order (255-tap to 961-tap) FIR brick-wall filters for surgical spectrum isolation.
* **Analog Demodulation**: Features software phase-locked loops (Costas Loop/DPLL) for 19kHz pilot tone recovery and DSB-SC stereo matrixing.
* **Digital Protocol Extraction**: Executes differential BPSK decoding and CRC-10 syndrome validation to extract Radio Data System (RDS) frames from noisy RF environments.

---

## Module Descriptions

### 1. RF Reconnaissance
Scripts dedicated to environmental scanning and edge-device spectrum monitoring. Implements dynamic frequency hopping and SNR threshold detection to identify active carriers in real-world multipath environments.

### 2. Hardware Abstraction
Interfaces directly with RTL-SDR hardware. Handles high-speed Direct Memory Access (DMA) of complex I/Q arrays (2.4 MSPS) for offline DSP debugging and algorithm validation.

### 3. Analog Demodulation Core
The mathematical core of the radio. Features custom implementations of phase discriminators (Delay-and-Multiply) and multi-tap FIR filters to extract Mono, Stereo, and NFM audio components from composite baseband signals.

### 4. Spectrum Analytics
Diagnostic tools utilizing Welch's method for Power Spectral Density (PSD) estimation. Used extensively during development to ensure surgical precision of bandpass filters targeting the 19kHz pilot tone and 57kHz RDS subcarrier.

### 5. Digital Protocol Decoder
The final digital stage transforming analog waveforms into human-readable data. Implements a Software PLL (Costas Loop) for 57kHz BPSK demodulation, differential decoding to bypass phase ambiguity, and a custom CRC-10 syndrome calculator to extract verified ISO standard PI codes and text from noisy RF environments.

---

## Showcase: The SDR Evolution

Want to see (and hear) the power of DSP in action? Experience the transformation from raw electromagnetic noise to high-fidelity stereo through our 4-stage evolution script.

### [Run the Evolution Pipeline](file:///d:/Elroy/%E5%80%8B%E4%BA%BA%E8%88%88%E8%B6%A3%E8%88%87%E7%A0%94%E7%A9%B6/SDR/3_analog_demodulation/step_by_step_evolution.py)
```powershell
python 3_analog_demodulation/step_by_step_evolution.py
```

**What happens?**
1. **Stage 0: Raw Chaos** - Listen to the unmodified RF capture (100% noise).
2. **Stage 1: Mono Whisper** - The first signs of audio emerge through FM demodulation.
3. **Stage 2: Warmth Refined** - High-frequency hiss is removed via 50μs de-emphasis.
4. **Stage 3: Stereo Emergence** - The soundstage expands as the 19kHz pilot tone unlocks the L-R difference.

Outputs are saved in `data/evolution/` with a comparative spectral analysis map.

---

## Project Structure

```text
BasebandForge/
│
├── 1_rf_reconnaissance/           # RF Scanning & Spectrum Surveillance
│   ├── power_scanner.py           # UHF band broadband scanner for signal occupancy
│   ├── fm_profiler.py             # FM band carrier profiling and SNR detection
│   ├── scan_office.py             # Multi-band (FM/FRS/IoT) PSD spectral analysis
│   └── security_scanner.py        # Real-time hopping scanner for FRS communications
│
├── 2_hardware_abstraction/        # Physical Layer Acquisition
│   └── iq_recorder.py             # Raw I/Q baseband data dumper and recorder
│
├── 3_analog_demodulation/         # Analog DSP Pipeline
│   ├── demodulate_nfm.py          # Narrowband FM (NFM) demodulator for voice
│   ├── fm_master_comparison.py    # Wideband FM stereo matrix (L+R, L-R, De-emphasis)
│   └── extract_pilot.py           # 19kHz pilot recovery and stereo reconstruction
│
├── 4_spectrum_analytics/          # DSP Filter Validation & Profiling
│   ├── fm_baseband_analyzer.py    # MPX composite baseband PSD visualization
│   ├── extract_harmonics.py       # 19k/38k/57k subcarrier isolation and extraction
│   └── extract_rds.py             # 57kHz BPSK microscopic waveform analysis
│
└── 5_digital_protocol_decoder/    # Digital Baseband & Protocol Parsing
    ├── rds_bit_extractor.py       # Costas Loop synchronization and bit stream extraction
    ├── rds_decoder_core.py        # Differential decoding and raw bitstream generation
    └── rds_final_decoder.py       # CRC-10 validation, 8-phase scanning, and PI code extraction
```

---

## Getting Started

Refer to the [Usage Guide (EN)](docs/USAGE_GUIDE_EN.md) or [使用指南 (ZH)](docs/USAGE_GUIDE_ZH.md) for detailed instructions on how to run these modules.

<img width="1200" height="1000" alt="image" src="https://github.com/user-attachments/assets/f7486b2e-41ad-403d-a699-ea7dfd7d7563" />
Fig 1. Time-domain extraction of MPX subcarriers. Demonstrating surgical isolation of the 19kHz pilot tone, the DSB-SC envelope of the 38kHz stereo difference signal, and the physical phase reversals of the 57kHz RDS BPSK data stream.
