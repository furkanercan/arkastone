import numpy as np
from src.tx.encoders.encoder import Encoder
from src.configs.config_modulation import ModConfig
from src.tx.core.modulator import Modulator
from src.common.ofdm import OFDM
from src.tx.core.tx_ofdm import OFDMTransmitter

class Transmitter:
    def __init__(self, mod_config, ofdm_config, code):
        """
        Initialize the Transmitter with an encoder (abstraction).

        Args:
            vec_polar_info_indices (list): decoder information bit index list
            len_logn (int): log2(N), where N is the block length.
        """
        # self.encoder = PolarEncoder(vec_polar_info_indices, len_logn)
        self.encoder          = Encoder(code)
        self.modulator        = Modulator(mod_config)
        
        self.ofdm             = OFDM(ofdm_config)
        self.ofdm_transmitter = OFDMTransmitter(self.ofdm)

        self.data_length_encoder_in     = code.len_k # Not needed may mislead when crc enabled
        self.data_length_encoder_out    = code.len_n # Not needed either
        self.data_length_modulator_out  = self.get_modulated_data_length()
        self.data_shape_ofdm_out        = self.get_transmitted_data_shape()
        
        # self.encoded_data = np.empty(code.len_n, dtype=bool)

        # self.transmitted_data = np.empty(???, dtype=bool)

        self.modulation_scheme = mod_config.type.lower()

        if self.modulation_scheme == "bpsk":
            self.modulated_data = np.empty(code.len_n, dtype=int)  # Real values for BPSK
        else:
            self.modulated_data = np.empty(code.len_n // self.modulator.log_num_constellations, dtype=complex)  # Complex values for the rest

    def tx_chain(self, uncoded_data, return_as_list=False):
        if isinstance(uncoded_data, list): # Normalize input at entry
            uncoded_data = np.array(uncoded_data, dtype=np.uint8)

        encoded_data = self.encoder.encode(uncoded_data)
        self.modulator.modulate(self.modulated_data, encoded_data)
        self.transmitted_data = self.ofdm_transmitter.transmit(self.modulated_data)

        # if return_as_list:
        #     return self.transmitted_data.tolist()
        # return self.transmitted_data

    def get_modulated_data_length(self):
        """
        Calculates the total length of the modulated data.
        Each modulated data consists of a number of encoded bits, based on:
            Code length
            modulation scheme
        The code length is divided by the log modulation scheme to find out the number of mod symbols.
        A ceil function is required.
    
        Args:
            len_n (int): Code length in bits
        """
        return int(np.ceil(self.data_length_encoder_out/self.modulator.log_num_constellations))

    def get_transmitted_data_shape(self):
        """
        Calculates the shape of the transmitted data to be sent over the channel.
        Each transmitted data symbol is an OFDM symbol. The length is determined by:
            self.ofdm.num_subcarriers
            self.ofdm.cyclic_prefix_length
        The code length is divided by the OFDM symbol length to find out the number of OFDM symbols.
        A ceil function is required.
    
        Args:
            modulated_data_length (int): Code length in modulated symbols
        """
        return [int(np.ceil(self.data_length_modulator_out/self.ofdm.num_subcarriers)), int(self.ofdm.num_subcarriers + self.ofdm.cyclic_prefix_length)]