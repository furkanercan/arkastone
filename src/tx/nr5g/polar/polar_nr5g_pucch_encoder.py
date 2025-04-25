import numpy as np
from typing import List
from src.tx.nr5g.polar.config.pucch_config import PUCCHConfig
from src.coding.crc.crc_encoder import CRCEncoder
from src.tx.nr5g.polar.components.code_block_segmentation import segment_transport_block
from src.tx.nr5g.polar.components.subchannel_allocation import subchannel_allocation
from src.tx.nr5g.polar.components.pc_bit_generation import assign_pc_bits
from src.tx.nr5g.polar.components.polar_encoder_core import polar_encode
from src.tx.nr5g.polar.components.subblock_interleaver import subblock_interleaver
from src.tx.nr5g.polar.components.rate_matching import rate_matching
from src.tx.nr5g.polar.components.channel_interleaver import channel_interleaver
from src.tx.nr5g.polar.components.code_block_concatenation import concatenate_code_blocks

# TODO: May add other classes for some of the methods later - specifically have polar encoder in mind

class PUCCHEncoder:
    def __init__(self, config: PUCCHConfig):
        self.config = config
        self.crc_encoder = CRCEncoder(config.crc_config)

    def encode(self, input_bits: List[int]) -> List[int]:
        
        input_bits = np.array(input_bits, dtype=np.int8)

        if(self.config.seg):
            # Step 1. Code Block Segmentation
            input_bits_blocks = segment_transport_block(input_bits, self.config.A)
            # Step 2-7: PUCCH Encoder Core
            # channel_interleaved_bits = [None] * 2  # Initialize the list for two code blocks
            # channel_interleaved_bits[0] = self._pucch_encode_core(input_bits_blocks[0])
            # channel_interleaved_bits[1] = self._pucch_encode_core(input_bits_blocks[1])
            channel_interleaved_bits = [self._pucch_encode_core(block) for block in input_bits_blocks]
            # Step 8: Concatenate Code Blocks
            output_bits = concatenate_code_blocks(channel_interleaved_bits, self.config.G)
            
        else:
            output_bits = self._pucch_encode_core(input_bits)

        return output_bits.tolist()

    def _pucch_encode_core(self, input_bits: List[int]) -> List[int]:
        """
        Core encoding function for PUCCH.

        Steps:
        2. CRC attachment
        3. Subchannel allocation + PC bit generation
        4. Polar encoding
        5. Subblock interleaving
        6. Rate matching
        7. Channel interleaving
        """
        # Initialization of variables
        bits_with_crc = [0] * self.config.K
        bits_subchannel = [0] * self.config.N
        nonfrozen_subchannels = [0] * (self.config.K + len(self.config.pc_indices))
        polar_encoded_bits = [0] * self.config.N
        subblock_interleaved_bits = [0] * self.config.N
        rate_matched_bits = [0] * self.config.E
        channel_interleaved_bits = [0] * self.config.E

        # Step 2: CRC Attachment
        bits_with_crc = self.crc_encoder.encode_and_append(input_bits)
        # Step 3: Frozen and Parity Check Bit Insertion
        subchannel_allocation(bits_with_crc, self.config.info_indices, self.config.pc_indices, bits_subchannel)
        assign_pc_bits(bits_subchannel, self.config.N)
        # Step 4: Polar Encoding
        nonfrozen_subchannels = [bit for idx, bit in enumerate(bits_subchannel) if idx not in self.config.frozen_indices]
        polar_encoded_bits = polar_encode(nonfrozen_subchannels, self.config.Gmat_kxN)
        # Step 5: Subblock Interleaving
        subblock_interleaved_bits = subblock_interleaver(polar_encoded_bits, self.config.N)
        # Step 6: Rate Matching
        rate_matched_bits = rate_matching(subblock_interleaved_bits, self.config.rm, self.config.N, self.config.E)
        # Step 7: Channel Interleaving
        channel_interleaved_bits = channel_interleaver(rate_matched_bits, self.config.channel_interleaved_indices)
        return channel_interleaved_bits