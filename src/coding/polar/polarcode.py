# import numpy as np
# import math
# from src.configs.config_polarcode import PolarCodeConfig
# from src.configs.config_crc import CRCConfig
# from src.configs.config_polarcode_decoder import PolarDecoderConfig
# from src.configs.config_quantization import QuantizeConfig
# from src.configs.config_polarcode_fast import PolarFastConfig

# class PolarCode:
#     def __init__(self, config):
#         """
#         Initialize the PolarCode with information indices and frozen bits.
#         """
#         self.len_k = config["len_k"]

#         self.reliability_indices = config["polar"]["rel_idx"]
#         self.len_n               = config["polar"]["len_n"]
#         self.len_logn            = config["polar"]["len_logn"]
#         self.en_crc              = config["polar"]["crc"]["enable"]
#         self.len_r               = config["polar"]["crc"]["length"]
#         self.crc_mode            = config["polar"]["crc"]["mode"]
#         self.len_kr              = self.len_k + self.len_r
        
#         self.qtz_enable          = config["polar"]["quantize"]["enable"]
#         self.qtz_chn_max         = config["polar"]["quantize"]["chnl_upper"]
#         self.qtz_chn_min         = config["polar"]["quantize"]["chnl_lower"]
#         self.qtz_int_max         = config["polar"]["quantize"]["intl_max"]
#         self.qtz_int_min         = config["polar"]["quantize"]["intl_min"]

#         # These are related to decoding, they should move to decoder part.
#         self.max_flips           = config["polar"]["decoder"]["flip_max_iters"]
#         self.fast_enable         = config["polar"]["fast_enable"]
#         self.nodesize_rate0      = config["polar"]["fast_max_size"]["rate0"]
#         self.nodesize_rate1      = config["polar"]["fast_max_size"]["rate1"]
#         self.nodesize_rep        = config["polar"]["fast_max_size"]["rep"]
#         self.nodesize_spc        = config["polar"]["fast_max_size"]["spc"]
#         self.nodesize_ml_0011    = config["polar"]["fast_max_size"]["ml_0011"]
#         self.nodesize_ml_0101    = config["polar"]["fast_max_size"]["ml_0101"]

#         self.frozen_bits, self.info_indices, self.crc_indices = self.create_polar_indices()
