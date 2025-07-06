# Getting Started

## 📦 Installation

Clone the repository:

```bash
git clone https://github.com/furkanercan/arkastone.git
cd arkastone
```

Install dependencies directly:

```bash
pip install -r requirements.txt
```

Or use a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## 🧪 Run a Simulation

```sh
python examples/tx_rx_chain_example_polar.py
```
## Example Simulation Output (Terminal)

```sh
#################################################################################
#                                                                               #
#  Successive Cancellation Decoder for Polar Codes            __                #
#  Author: Furkan Ercan                               _(\    |@@|               #
#                                                    (__/\__ \--/ __            #
#  Copyright (c) 2025 Furkan Ercan.                     \___|----|  |   __      #
#  All Rights Reserved.                                  \ }{ /\ )_ / _\ _\     #
#                                                           /\__/\ \__O (__     #
#  Version: 0.1                                            (--/\--)    \__/     #
#                                                          _)(  )(_             #
#  Licensed under the MIT License                         `---''---`            #
#  See the LICENSE file for details.                                            #
#                                                                               #
#  ASCII Art Source: https://www.asciiart.eu/                                   #
#                                                                               #
#################################################################################

SNR (dB)    BER           FER           ITER       Frames     Errors     Time
1.000e+00   2.78969e-01   7.61800e-01   1.00e+00   1.00e+04   7.62e+03   00:00:54
1.250e+00   1.90590e-01   5.67700e-01   1.00e+00   1.00e+04   5.68e+03   00:00:54
1.500e+00   1.13088e-01   3.72000e-01   1.00e+00   1.00e+04   3.72e+03   00:00:53
```


---