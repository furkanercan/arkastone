# `src/tx/` — Transmitter Subsystem

This module contains all transmitter-side logic for the communication system, including encoding, modulation, and OFDM transmission.

### Structure

- `core/`: Contains the high-level transmitter chain and signal processing blocks.
- `encoders/`: Houses all encoder types following a shared interface (`BaseEncoder`).
- `nr5g/`: Contains standard-specific logic for NR 5G, including PUCCH-compatible polar encoders.

### Key Concepts

- `Encoder` objects are dynamically selected based on the `code` configuration.
- `Transmitter` coordinates the full TX path: encode → modulate → OFDM.
- Standard-specific encoders can reuse base components from `encoders/`.

### Example

```python
from src.tx.core.tx import Transmitter
tx = Transmitter(mod_config, ofdm_config, code)
tx.tx_chain(info_bits)