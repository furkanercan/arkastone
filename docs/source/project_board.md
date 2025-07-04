
# Arkastone Project Board (Master Task List)

---

## ✅ DONE

- Built `PUCCHEncoder` with full pipeline
- Modularized `_pucch_encode_core()`
- Integrated `CRCEncoder` using `CRCConfig`
- Refactored stateful variables into local variables
- Created `PolarNR5GWrapper` to generate real configs
- Validated pipeline end-to-end using wrapper-generated config
- Chose `List[int]` externally, `np.ndarray` internally
- Agreed on project name: **Arkastone**
- Defined monetization models (open-core, pro tier, course, consulting)
- Refined elevator pitch (street-level, technical, visionary)

---

## 🛠️ IN PROGRESS

- Set up Notion project board
- Finalize file structure standardization
- Start documenting each component (`docs/components/...`)
- Create README.md with vision, quickstart, structure
- Draft `PUSCHEncoder` structure
- Keep a `dev-log.md` or Notion Dev Log

---

## 💡 BRAINSTORMED / IDEAS PARKING LOT

- Add SC / SCF / SCL Decoder modules
- Integrate AWGN or BSC channel models
- Build BER/BLER simulation harness
- Add test vectors using randomized and 3GPP-like inputs
- Design trace/debug hooks for pipeline steps
- Add end-to-end simulation script: TB → TX → channel → RX
- Implement MAC-layer logic (segmentation, HARQ, CB scheduling)
- Support visual flow diagrams / bit traces
- Create GUI / drag-and-drop web frontend
- Publish course: *“Build a 5G PHY from Scratch”*
- Open-source with paid tier (advanced tools, verified vectors)
- Add RTL export hooks (SystemVerilog generation)
- Build SDR-ready waveform generator (Python → GNU Radio)

---

## 🚧 TO BE DONE (TBD / Actionable)

- Write documentation for CRC Encoder (`crc_encoder.md`)
- Write documentation for PUCCH Encoder (`pucch_encoder.md`)
- Generate file `docs/architecture.md` with data flow overview
- Set up `tests/` directory with `pytest` base
- Clean up `config/` folder to use consistent names + imports
- Generalize segmentation logic for multiple TBs
- Add `channel_interleaver.md` doc
- Add CLI tool or script to run full pipeline from command line
- Create example notebook: “Simulate PUCCH transmission”
- Generate one real 3GPP-like test vector and validate output
- Tag functions with docstrings and input/output details
