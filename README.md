# Communications Project

## Overview
This project is a modular and object-oriented Python framework for simulating a complete digital communication system. It includes essential components such as transmission, reception, modulation, error correction, and channel modeling. The project is designed for research, development, and deployment, with a long-term goal of optimizing it using SIMD or similar techniques.

## Features
- **Transmitter & Receiver Classes**: Implements the key components of a digital communication system.
- **Modulation**: Includes BPSK, QPSK, 16QAM, and supports additional schemes.
- **OFDM Integration**: Separates modulation functionality and passes modulated data to the OFDM module.
- **Error Correction**: Supports Polar Codes with SC, SC-Flip, and future improvements.
- **Channel Simulation**: Models real-world communication environments.
- **Performance Metrics**: Enables analysis of BER, FER, and other key metrics.
- **Testing & Validation**: Supports short and overnight test runs with GitHub integration for automation.

## Installation
To set up the project environment:
```sh
# Clone the repository
git clone <repo_url>
cd <project_directory>

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`

# Install dependencies
pip install -r requirements.txt
```

## Running Example Simulations
Example simulations for the entire TX/RX chain are located under examples/ folder.
They must be run from the root folder.
First, set the environment correctly - otherwise the local exports may not get recognized:
```sh
export PYTHONPATH=/path_to_root_folder/src:$PYTHONPATH
```
Then, run the example script:
```sh
python examples/tx_rx_chain_example_polar.py
```
Make sure that the virtual environment is set correctly. The output should look like:
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

All run logs are populated under output/ folder, tagged with date and time. In each run folder, the terminal output and a copy of the configuration file is saved for convenience.

## Running Tests
```sh
pytest tests/
```

## Roadmap
- **Extend Modulation Support**: Implement 16-QAM, 64-QAM, and other schemes.
- **Enhance Error Correction**: Add LDPC and Turbo Codes.
- **Hardware Optimization**: Implement SIMD for efficiency.
- **Deployment & Acceleration**: Move from simulation to real-time execution.

## Contributing
Contributions are welcome! Please follow these steps:

1. Fork the repository.
2. Create a feature branch.
3. Commit your changes.
4. Open a pull request.

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.

## Contact
For questions and collaborations, contact furkanercan88 [at] gmail [dot] com or https://github.com/furkanercan.
