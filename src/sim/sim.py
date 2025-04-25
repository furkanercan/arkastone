import os
import datetime
import numpy as np
from src.configs.config_sim import SimConfig

class Simulation:
    def __init__(self, config: SimConfig, output_folder="tmp", sim_size=50):
        self.num_frames = config.loop.num_frames
        self.num_errors = config.loop.num_errors
        self.max_frames = config.loop.max_frames

        self.plot_enable   = config.save.plot_enable
        self.lutsim_enable = config.save.lutsim_enable
        self.save_output   = config.save.save_output

        self.output_path = os.path.join(output_folder, "fec_results.out")

        self.source = config.sweep_type.lower()

        self.simpoints = config.sweep_vals.simpoints
        self.simpoints = np.array(self.simpoints)
        self.lenpoints = config.sweep_vals.len_points
        
        if(self.source == "snr"):
            snr_linear = 10 ** (self.simpoints / 10) 
            self.variance = 1/snr_linear
            self.stdev = np.sqrt(self.variance)
        else:
            TypeError("Current version only supports SNR(dB) type simulation!")
        
        self.snr_points         = np.zeros(sim_size, dtype=float)
        self.count_frame        = np.zeros(sim_size, dtype=int)
        self.count_dec_steps    = np.zeros(sim_size, dtype=int)
        self.count_dec_iters    = np.zeros(sim_size, dtype=int)
        self.count_bit_error    = np.zeros(sim_size, dtype=int)
        self.count_frame_error  = np.zeros(sim_size, dtype=int)
        self.ber                = np.zeros(sim_size, dtype=float)
        self.bler               = np.zeros(sim_size, dtype=float)
        self.avg_steps          = np.zeros(sim_size, dtype=float)
        self.avg_iters          = np.zeros(sim_size, dtype=float)

        self.sim_mode = config.mode
        if(self.sim_mode == "dev"):
            self.generate_sim_header()


    def run_simulation(self, idx):
        min_frame_condition = (self.count_frame[idx] < self.num_frames)
        min_error_condition = (self.count_frame_error[idx] < self.num_errors)
        # max_frame_condition = (self.count_frame[idx] > self.max_frames)
        return (min_frame_condition or min_error_condition)# and max_frame_condition )
    

    def collect_run_stats(self, idx, dec_steps, dec_iters, info_data, decoded_data):
        self.count_frame[idx]       += 1
        self.count_dec_steps[idx]   += dec_steps
        self.count_dec_iters[idx]   += dec_iters
        self.count_frame_error[idx] += (1 if (info_data != decoded_data).any() else 0)
        self.count_bit_error[idx]   += np.sum(info_data != decoded_data)

    def get_ber(self, idx, len_k):
        return self.count_bit_error[idx]/(self.count_frame[idx] * len_k)
    
    def get_bler(self, idx):
        return self.count_frame_error[idx]/self.count_frame[idx]
    
    def get_avg_steps(self, idx):
        return self.count_dec_steps[idx]/self.count_frame[idx]
    
    def get_avg_iters(self, idx):
        return self.count_dec_iters[idx]/self.count_frame[idx]

    def get_ber_results(self, idx, len_k):
        return {
            "ber": self.get_ber(idx, len_k),
            "fer": self.get_bler(idx),
            "avg_steps": self.get_avg_steps(idx),
            "avg_iters": self.get_avg_iters(idx),
            "frames": self.count_frame[idx],
            "errors": self.count_frame_error[idx]
        }

    def update_run_results(self, idx, len_k):
        self.ber[idx]  = self.get_ber(idx,len_k)
        self.bler[idx] = self.get_bler(idx)
        self.avg_steps[idx] = self.get_avg_steps(idx)
        self.avg_iters[idx] = self.get_avg_iters(idx)

    def display_run_results_temp(self, idx, snr_point, time_elapsed, prev_status_msg):
        if(self.sim_mode == "dev"):
            status_msg = f"{snr_point:.3e}   {self.ber[idx]:.5e}   {self.bler[idx]:.5e}   {self.avg_iters[idx]:.2e}   {self.count_frame[idx]:.2e}   {self.count_frame_error[idx]:.2e}   {time_elapsed}"
            status_pad = ' ' * max(0, len(prev_status_msg) - len(status_msg))
            end_char = '\r'
            print(status_msg + status_pad, end=end_char, flush=True)
            return status_msg
        else:
            return prev_status_msg

    def display_run_results_perm(self, idx, snr_point, time_elapsed, prev_status_msg):
        if(self.sim_mode == "dev"):
            status_msg = f"{snr_point:.3e}   {self.ber[idx]:.5e}   {self.bler[idx]:.5e}   {self.avg_iters[idx]:.2e}   {self.count_frame[idx]:.2e}   {self.count_frame_error[idx]:.2e}   {time_elapsed}"
            status_pad = ' ' * max(0, len(prev_status_msg) - len(status_msg))
            end_char = '\n'
            if self.save_output == 1:
                with open(self.output_path, 'a') as file_o:
                    file_o.write(status_msg + "\n")

            print(status_msg + status_pad, end=end_char, flush=True)
            prev_status_msg = status_msg
            return status_msg
        else:
            return prev_status_msg

    def generate_sim_header(self):
        header_lines = [
            r"#################################################################################",
            r"#                                                                               #",
            r"#  Successive Cancellation Decoder for Polar Codes            __                #",
            r"#  Author: Furkan Ercan                               _(\    |@@|               #",
            r"#                                                    (__/\__ \--/ __            #",
            r"#  Copyright (c) {} Furkan Ercan.                     \___|----|  |   __      #".format(datetime.datetime.now().year),
            r"#  All Rights Reserved.                                  \ }{ /\ )_ / _\ _\     #",
            r"#                                                           /\__/\ \__O (__     #",
            r"#  Version: 0.1                                            (--/\--)    \__/     #",
            r"#                                                          _)(  )(_             #",
            r"#  Licensed under the MIT License                         `---''---`            #",
            r"#  See the LICENSE file for details.                                            #",
            r"#                                                                               #",
            r"#  ASCII Art Source: https://www.asciiart.eu/                                   #",
            r"#                                                                               #",
            r"#################################################################################",
            r"",
            r"SNR (dB)    BER           FER           ITER       Frames     Errors     Time",
        ]

        header = "\n".join(header_lines)
        print(header)
        if self.save_output == 1:
            with open(self.output_path, 'w') as file_o:
                file_o.write(header + "\n")
        return header