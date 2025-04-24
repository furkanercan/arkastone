import numpy as np
import math
from src.rx.decoders.decoder import Decoder
from src.rx.core.demodulator import Demodulator
from src.common.ofdm import OFDM
from src.rx.core.rx_ofdm import OFDMReceiver
from src.coding.coding import Code

class Receiver:
    def __init__(self, mod_config, ofdm_config, code: Code):
        """
        Initialize the Receiver with a decoder (abstraction).
        """
        self.len_n = code.len_n
        self.len_k = code.len_k

        self.demodulator = Demodulator(mod_config)

        self.ofdm          = OFDM(ofdm_config)
        self.ofdm_receiver = OFDMReceiver(self.ofdm, int(self.len_n/self.demodulator.log_num_constellations))

        # self.demodulator = Demodulator(mod_config)
        self.decoder = Decoder(code)
        self.decoder.initialize_decoder()
        
        self.vec_llr = np.empty(self.len_n, dtype=float)
        self.decoded_data = np.empty(self.len_k, dtype=bool)

    def rx_chain(self, channel_data, awgn_var):
        """
        Perform the receive chain, including decoding.

        Returns:
            list: Decoded data.
        """
        # Placeholder for more functionalities (e.g., channel equalization)
        self.deOFDM_data = self.ofdm_receiver.receive(channel_data)
        self.demodulator.demodulate(self.vec_llr, self.deOFDM_data, awgn_var)
        self.decoder.decode_chain(self.decoded_data, self.vec_llr)

