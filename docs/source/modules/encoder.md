# Encoder Module Architecture

This document describes the structure, responsibilities, and usage of the `Encoder` module in the communications framework.

---

## 🧱 Overview

The encoder module enables a clean, extensible, and efficient way to implement multiple channel coding schemes in the framework. It separates **interface**, **implementation**, and **selection logic** using a three-layer architecture:

```
[BaseEncoder] ⇨ [Encoder] ⇨ [DedicatedEncoder]
```

---

## 🧩 1. `BaseEncoder` (Abstract Interface)

**Location:** `framework/base/encoder.py`  
**Type:** Abstract base class (`ABC`)

### Purpose:
Defines the standard encoder interface, shared functionality, and enforces implementation of the encoding logic.

### Responsibilities:
- Store input/output lengths (`A`, `G`)
- Validate inputs
- Provide the public `.encode()` method
- Require subclasses to implement `_encode_np(...)`

### Signature:
```python
class BaseEncoder(ABC):
    def __init__(A: int, G: int, name: str = "")
    def encode(info_bits: List[int]) -> List[int]
    def _encode_np(info_bits: np.ndarray) -> np.ndarray  # abstract
    def check_input_length(...)
    def reset()  # optional
```

---

## 🧩 2. `Encoder` (Factory Wrapper)

**Location:** `tx/encoders/encoder.py`  
**Type:** Lightweight instantiator and delegate

### Purpose:
Abstracts the logic for selecting the correct encoder based on `code.type` (e.g., `"polar"`, `"uncoded"`, `"nr5g"`).

### Responsibilities:
- Initialize the correct dedicated encoder
- Forward `.encode()` calls to the selected encoder
- Expose shared properties like `A` and `G`

### Signature:
```python
class Encoder:
    def __init__(code)
    def encode(info_bits: List[int]) -> List[int]
    @property def A
    @property def G
```

---

## 🧩 3. Dedicated Encoder Implementations

**Examples:**
- `PolarEncoder` → `tx/encoders/polar_encoder.py`
- `UncodedEncoder` → `tx/encoders/uncoded_encoder.py`
- `PolarNR5GEncoder` → `schemes/nr5g/polar_nr5g_encoder.py`

### Purpose:
Each class contains the actual encoding algorithm.

### Responsibilities:
- Inherit from `BaseEncoder`
- Implement `_encode_np()` for core logic
- May contain additional helper methods (e.g., CRC, polar matrix generation)

### Template:
```python
class MyEncoder(BaseEncoder):
    def __init__(self, code): ...
    def _encode_np(self, info_bits: np.ndarray) -> np.ndarray:
        # Do the actual encoding here
```

---

## ✅ Usage Example

```python
from tx.encoders.encoder import Encoder

encoder = Encoder(code)  # Code object with .type, .len_k, .len_n, etc.
info_bits = [1, 0, 1, 1, 0, 0, 1]
encoded_bits = encoder.encode(info_bits)  # Output: [0, 1, 0, 1, ...]
```

---

## ✨ Design Highlights

| Feature                   | Benefit                              |
|---------------------------|---------------------------------------|
| Abstract base class       | Enforces consistent interface         |
| Clean factory wrapper     | No repeated "if code.type == ..."     |
| Easy to extend            | Add new encoders without touching pipelines |
| Efficient in production   | `_encode_np()` allows in-place optimization |
| Readable and testable     | Each encoder is isolated and modular  |

---

## 📚 Next Steps

- [ ] Add diagrams in `docs/architecture/diagrams/encoder_arch.drawio`
- [ ] Link this doc in project README and developer onboarding
- [ ] Apply the same structure to Decoder, Modulator, Channel models