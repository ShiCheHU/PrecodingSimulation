# MU-MIMO Simulation with Block Diagonalization Precoding

This repository provides a Python-based simulation framework for a multi-user multiple-input multiple-output (MU-MIMO) downlink system with **Block Diagonalization (BD)** precoding. The code supports bit generation, digital modulation and demodulation, channel generation, BD precoding, receiver combining, signal detection, and BER evaluation under different modulation formats and detection methods.

The framework is intended for algorithm verification, link-level performance evaluation, and educational demonstration of MU-MIMO transmission and reception processing.

---

## Features

- MU-MIMO downlink simulation with multiple users and multiple spatial streams per user
- Block Diagonalization (BD) precoding for inter-user interference suppression
- Receiver combining and linear detection
  - Zero-Forcing (ZF)
  - Minimum Mean Square Error (MMSE)
- Support for multiple modulation formats
  - QPSK
  - 16QAM
  - 64QAM
- Rayleigh and Rician channel generation
- BER performance evaluation versus SNR
- Flexible and modular code structure for future extension

---

## Repository Structure

```text
.
├── main.py
├── channel.py
├── receiver.py
├── transmitter.py
├── constellation.py
├── BlockDiagonalization.py
└── README.md
```

### File Description

- **main.py**  
  Main simulation script. It configures the system parameters, runs BER simulations over a range of SNR values, and plots the final curves.

- **channel.py**  
  Channel-related processing, including channel generation, received signal construction, and effective channel computation.

- **receiver.py**  
  Receiver-side processing, including receiver combining and symbol detection using ZF or MMSE.

- **transmitter.py**  
  Transmitter-side processing, including random bit generation, modulation, and demodulation.

- **constellation.py**  
  Constellation generation for QPSK, 16QAM, and 64QAM.

- **BlockDiagonalization.py**  
  Implementation of the BD precoding algorithm and stream-wise power allocation.

---

## System Model

The simulation considers a MU-MIMO downlink system with:

- `Nt`: number of transmit antennas at the base station
- `Nr`: number of receive antennas per user
- `K`: number of users
- `S`: number of spatial streams per user

For each channel realization:

1. Random bits are generated and modulated.
2. A channel matrix is generated for all users.
3. BD precoding is computed at the transmitter.
4. The transmitted symbols pass through the channel with additive noise.
5. Receiver combining and linear detection are applied.
6. The detected symbols are demodulated back to bits.
7. BER is calculated by comparing transmitted bits and detected bits.

---

## Requirements

The code is written in Python and requires the following packages:

- `numpy`
- `matplotlib`

You can install them with:

```bash
pip install numpy matplotlib
```

---

## How to Run

Run the main simulation script:

```bash
python main.py
```

If the main script is configured to sweep multiple modulation formats and detectors, it will:

- simulate BER over a range of SNR values
- generate one BER curve for each configuration
- plot all curves in a single figure
- save the figure to a `.png` file

---

## Example Simulation Settings

Typical parameters used in the simulation are:

```python
Nt = 32   # Number of transmit antennas
Nr = 3    # Number of receive antennas per user
S  = 2    # Number of spatial streams per user
K  = 4    # Number of users
mod = '16QAM'
method = 'ZF'
```

The SNR sweep can be configured, for example, as:

```python
SNR_list = np.arange(-20, 10, 4)
```

---

## Notes on Detection and Effective Channel Modeling

In this project, special attention should be paid to consistency between:

- the transmitted symbol model
- the stream-wise power loading matrix
- the effective channel used in the receiver detector

If the receiver-side effective channel does not include the same stream-wise scaling used during transmission, high-order modulation formats such as 64QAM may exhibit an artificial BER floor even at high SNR.

For reliable results, the receiver detector should use an effective channel model consistent with the actual transmitted signal structure.

---

## Possible Extensions

This code base can be extended in several directions, for example:

- imperfect CSI
- correlated MIMO channel models
- alternative precoding methods
  - ZF precoding
  - RZF precoding
  - MMSE precoding
- soft demodulation and channel decoding
- OFDM-based wideband simulation
- coded BER or BLER performance evaluation
- adaptive modulation and coding

---

## License

This project is currently provided for academic study, research, and personal use.

If you plan to publish or redistribute this code, you may wish to add a formal open-source license such as MIT or BSD-3-Clause.

---

## Contact

If you use this repository as part of your research or technical blog, you may consider adding:

- Shicheng Hu
- University of Chinese Academy of Science
- https://github.com/ShiCheHU
- hushch2018@163.com

for easier academic or professional reference.
