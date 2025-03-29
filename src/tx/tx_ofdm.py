import numpy as np
from src.common.ofdm import OFDM

class OFDMTransmitter:
    def __init__(self, ofdm_module):
        self.ofdm_module = ofdm_module


    def transmit(self, modulated_data):
        """
        Transmit modulated data using OFDM with data fragmentation.
        Splits modulated data into chunks that fit into the subcarriers, 
        performs IFFT, and adds cyclic prefix for each chunk.
        """
        fragments = self._fragment_data(modulated_data)
        transmitted_signal = []

        for fragment in fragments:
            time_domain_signal = self.ofdm_module.perform_ifft(fragment)
            time_domain_signal *= np.sqrt(self.ofdm_module.num_subcarriers) # Normalize the power of the time-domain signal
            cyclic_prefixed_signal = self.ofdm_module.add_cyclic_prefix(time_domain_signal)
            transmitted_signal.append(cyclic_prefixed_signal)

        # Concatenate all fragments to form the complete transmitted signal
        return np.concatenate(transmitted_signal)#*np.sqrt(len(modulated_data))

    def _fragment_data(self, modulated_data):
        """
        Fragment the input modulated data into chunks that fit into the available subcarriers.
        If the number of symbols % number of subcarriers is not equal to 0, then the final fragment
        data width is less. In that case, various different techniques can (should) be employed based
        on the use case. 
        For single-user communication, which is the current (temporary) focus of this work, the unused
        subcarriers are padded with zero. However this approach may yield an impact on the average signal
        power in the time domain, so need to be careful about it.
        For multi-user communications, this would be treated differently, since it is tightly coupled to
        the concept of shared resource allocation (especially in 5G applications).
        """
        num_symbols = len(modulated_data)
        fragments = []

        # Calculate the number of fragments needed
        for start_idx in range(0, num_symbols, self.ofdm_module.num_subcarriers):
            end_idx = min(start_idx + self.ofdm_module.num_subcarriers, num_symbols)
            fragment = np.zeros(self.ofdm_module.num_subcarriers, dtype=modulated_data.dtype)
            
            # Copy data to the fragment
            fragment[:end_idx - start_idx] = modulated_data[start_idx:end_idx]
            fragments.append(fragment)

        return fragments


"""
When the modulated data exceeds the number of subcarriers available in your OFDM system, there are a few strategies you can consider to handle the data appropriately:

Resizing the Data (Padding) - CURRENT APPROACH:

If the modulated data size is larger than the number of available subcarriers, you could pad the data to fit the number of subcarriers. Padding involves adding zeros or repeating the last few symbols to ensure the data fits perfectly.
While this works in terms of fitting the data, it may reduce the overall efficiency, as padding doesn't carry any useful information.
Data Fragmentation:

If the data is too large for a single transmission, you could split the data into smaller chunks (frames) that fit within the available subcarriers. After each frame is transmitted, you can proceed with the next chunk.
This approach requires managing the sequence of frames and possibly adding synchronization or sequence numbers to ensure correct reassembly on the receiver side.
Subcarrier Allocation:

If the system supports dynamic allocation of subcarriers, you could adaptively allocate more subcarriers based on the amount of data. This would involve designing a flexible system where the number of subcarriers can vary based on the data size, but this requires careful management of bandwidth and system complexity.
Resampling or Downsampling:

If applicable to your system, resampling the data could also work. In this case, you would adjust the data rate to match the available subcarriers by reducing the number of data points (downsampling). This could be useful if you're willing to reduce the resolution of the data to fit the subcarrier count.
Frequency Domain Mapping (Bandwidth Expansion):

Another option could involve mapping the data to a larger frequency range if the system allows expanding the frequency spectrum. You could use a technique like Frequency Division Multiplexing (FDM) to expand the available bandwidth for transmission.
"""