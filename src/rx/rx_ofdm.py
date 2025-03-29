import numpy as np
from src.common.ofdm import OFDM

class OFDMReceiver:
    def __init__(self, ofdm_module, len_n):
        self.ofdm_module = ofdm_module
        self.codeword_len = len_n

    def receive(self, received_signal):
        """
        Receive and process OFDM data with data reassembly.
        Splits the received signal into fragments, processes each, 
        and concatenates the frequency domain data.
        Process received OFDM symbols and reconstruct the full codeword.
        Stop processing once codeword_len is reached.
        """
        reconstructed_codeword = []
        symbols_received = 0

        fragments = self._fragment_signal(received_signal)

        for fragment in fragments:
            # Remove cyclic prefix
            signal_no_cp = self.ofdm_module.remove_cyclic_prefix(fragment)

            # Perform FFT to get frequency domain symbols
            frequency_domain_signal = self.ofdm_module.perform_fft(signal_no_cp)

            # Collect symbols while respecting codeword_len
            remaining_space = self.codeword_len - symbols_received
            symbols_to_take = min(len(frequency_domain_signal), remaining_space)
            reconstructed_codeword.extend(frequency_domain_signal[:symbols_to_take])

            # Update the count of received symbols
            symbols_received += symbols_to_take

            # Stop if the codeword length is fully received
            if symbols_received >= self.codeword_len:
                break

        return np.array(reconstructed_codeword)
    
    # def receive(self, received_signal):
    #     """
    #     Receive and process OFDM data with data reassembly.
    #     Splits the received signal into fragments, processes each, 
    #     and concatenates the frequency domain data.
    #     """
    #     fragments = self._fragment_signal(received_signal)
    #     frequency_domain_data = []

    #     for fragment in fragments:
    #         signal_no_cp = self.ofdm_module.remove_cyclic_prefix(fragment)
    #         frequency_domain_signal = self.ofdm_module.perform_fft(signal_no_cp)
    #         frequency_domain_data.append(frequency_domain_signal)

    #     # Concatenate all fragments to reconstruct the full frequency-domain signal
    #     return np.concatenate(frequency_domain_data)

    def _fragment_signal(self, received_signal):
        """
        Fragment the input received signal into chunks based on the size of subcarriers 
        and cyclic prefix length.

        When TX sends information, it is not evident to the RX that what's padding and what's
        not. In other words, TX keeps sending data and RX must discard some of this data, because
        they are redundant. The RX must strictly know what to discard and what to keep. 

        There are several different ways, and layers, to this concept, including resource allocation,
        MAC layer, control channel information, header assignment, etc. We haven't explored any of 
        those yet. For the time being we will be FIXING the frame size, and passing the code length
        as a parameter to the OFDM receiver, to tell RX what to discard and what not to discard.

        Current scheme:
        For any given frame, all subcarriers must be used consecutively to send data. The TX OFDM 
        must pad the remaining unused subcarriers in an OFDM symbol with zeroes before passing it to 
        IFFT. Consequently, at the OFDM RX, after FFT, the excess data must be found and discarded.
        
        Example:

        Code length = 8
        Number of subcarriers = 6
        Cyclic prefix length = 2

        Code represented by n, cyclic prefix represented by p. 
        Time domain represented by t. Padding represented by 0.
        
        At TX:
        Resulting 2 OFDM symbols: nnnnnn     nn0000
        After applying IFFT, (represent by t): tttttt     tttttt
        After applying CP:  pptttttt   pptttttt

        At RX:
        Remove CP: tttttt     tttttt
        Apply FFT: nnnnnn     nnnnnn
        Discard p: nnnnnn     nn
        
        """
        chunk_size = self.ofdm_module.num_subcarriers + self.ofdm_module.cyclic_prefix_length
        num_samples = len(received_signal)

        # Split the received signal into fragments
        return [
            received_signal[i:i + chunk_size]
            for i in range(0, num_samples, chunk_size)
        ]