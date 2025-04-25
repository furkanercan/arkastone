# from src.coding.polar.nr5g.polar_nr5g_pucch_encoder import PUCCHEncoder
# from src.coding.polar.nr5g.config.pucch_config import PUCCHConfig
# from src.coding.polar.nr5g.config.crc_config import CRCConfig # TODO: This should move to some meaningful place.
# from src.coding.polar.nr5g.polar_nr5g_wrapper import PolarNR5GWrapper

# import numpy as np

# # === Step 1: Define control params ===
# A = 40
# G = 96
# channel_type = "PUCCH"

# # === Step 2: Call wrapper ===
# wrapper = PolarNR5GWrapper(A=A, G=G, channel_type=channel_type)
# config = wrapper._generate_pucch_encoder_config()

# # === Step 3: Generate dummy input ===
# input_bits = np.random.randint(0, 2, size=A).tolist()

# # === Step 4: Run encoder ===
# encoder = PUCCHEncoder(config)
# output_bits = encoder.encode(input_bits)

# # === Step 5: Check ===
# print(f"[PUCCH] A={A}, G={G}, N={config.N}, E={config.E}")
# print(f"Input length : {len(input_bits)}")
# print(f"Output length: {len(output_bits)}")
# assert len(output_bits) == config.E
# print("✅ Passed: Output matches expected E.")

