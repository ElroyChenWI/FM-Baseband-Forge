BasebandForge: Demystifying the RF Physical Layer

BasebandForge is a ground-up Software Defined Radio (SDR) DSP engine built entirely in Python. Rejecting black-box frameworks like GNU Radio, this project directly implements the mathematical core of telecommunications to bridge the gap between raw electromagnetic waves and digital protocols. 

Core Architectural Highlights:
* RF to Baseband: Processes 2.4 MSPS raw I/Q streams down to a 240 kSPS composite multiplex (MPX).
* Precision DSP Filtering: Implements custom high-order (255-tap to 961-tap) FIR brick-wall filters for surgical spectrum isolation.
* Analog Demodulation: Features software phase-locked loops (Costas Loop/DPLL) for 19kHz pilot tone recovery and DSB-SC stereo matrixing.
* Digital Protocol Extraction: Executes differential BPSK decoding and CRC-10 syndrome validation to extract Radio Data System (RDS) frames from noisy RF environments.



<img width="1200" height="1000" alt="image" src="https://github.com/user-attachments/assets/f7486b2e-41ad-403d-a699-ea7dfd7d7563" />

