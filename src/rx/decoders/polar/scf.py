import numpy as np
from src.rx.decoders.polar.sc import PolarDecoder_SC  
from src.configs.config_polarcode import PolarCodeConfig
from src.coding.crc.crc_decoder import CRCDecoder

class PolarDecoder_SCF(PolarDecoder_SC):
    def __init__(self, config: PolarCodeConfig):
        super().__init__(config)
        # SCF-specific initializations
        self.len_r  = config.len_r
        self.len_k  = config.len_k
        self.len_kr = config.len_kr
        self.t_max  = config.code.decoder.flip_max_iters
        self.crc = CRCDecoder(config.crc)
        
        self.vec_decoded_kr = np.empty(self.len_kr, dtype=bool) #info+crc decoded vector, for internal use
        self.vec_leaf_mag = np.empty(self.len_kr, dtype=float)
        self.sorted_indices = np.empty(self.t_max, dtype=int)
        
    def decode_chain(self, vec_decoded, vec_llr):
        self.dec_scf(vec_llr)
        vec_decoded[:] = self.vec_decoded_kr[:self.len_k]

    def dec_scf(self, vec_llr):
        # Run initial SC decoding
        self.dec_sc_flip_init(vec_llr)
        crc_pass = self.crc.crc_decode(self.vec_decoded_kr, self.len_k)
        iter_scf = 1
        if(crc_pass == False and self.t_max > 0):
            self.sorted_indices = self.get_sorted_indices(self.vec_leaf_mag)
            # print("vec_leaf_mag:", self.vec_leaf_mag)
            flip_iter = 0
            while(crc_pass == False and flip_iter < self.t_max):
                next_flip_idx = self.sorted_indices[flip_iter]
                self.dec_sc_with_flip(next_flip_idx)
                iter_scf += 1
                crc_pass = self.crc.crc_decode(self.vec_decoded_kr, self.len_k)
                flip_iter += 1
        
    
    def get_sorted_indices(self, vector):
        return sorted(range(len(vector)), key=lambda x: vector[x])

    def dec_sc_flip_init(self, vec_llr):
        """
        IMPORTANT: This function can only be used when the fast_enable is set to False.
        Fast-SCF algorithm is TBD.
        """
        self.mem_alpha[self.len_logn][:] = vec_llr # Place LLRs to bottom row of mem_alpha
        info_ctr = 0
        for i in range(len(self.vec_dec_sch)):
            if self.vec_dec_sch[i] == 'F':
                self.dec_sc_f(self.vec_dec_sch_size[i], self.vec_dec_sch_depth[i], self.qtz_enable, self.qtz_int_max)
            elif self.vec_dec_sch[i] == 'G':
                self.dec_sc_g(self.vec_dec_sch_size[i], self.vec_dec_sch_depth[i], self.qtz_enable, self.qtz_int_max, self.qtz_int_min)
            elif self.vec_dec_sch[i] == 'C':
                self.dec_sc_c(self.vec_dec_sch_size[i], self.vec_dec_sch_depth[i], self.vec_dec_sch_dir[i])
            elif self.vec_dec_sch[i] == 'R0':
                if(self.vec_dec_sch_dir[i] == 0):
                    self.mem_beta_l[0][0] = 0
                else:
                    self.mem_beta_r[0][0] = 0
            elif self.vec_dec_sch[i] == 'R1':
                self.vec_decoded_kr[info_ctr] = 0 if self.mem_alpha[0][0] >= 0 else 1
                self.vec_leaf_mag[info_ctr] = np.abs(self.mem_alpha[0][0])
                if self.vec_dec_sch_dir[i] == 0:
                    self.mem_beta_l[0][0] = 0 if self.mem_alpha[0][0] >= 0 else 1
                else:
                    self.mem_beta_r[0][0] = 0 if self.mem_alpha[0][0] >= 0 else 1
                info_ctr += 1
        
        # self.vec_decoded_kr[:] = ((self.mem_beta_l[self.len_logn] @ self.matG_Nxk) % 2)
                

    def dec_sc_with_flip(self, next_flip_idx):
        
        info_ctr = 0
        for i in range(len(self.vec_dec_sch)):
            if self.vec_dec_sch[i] == 'F':
                self.dec_sc_f(self.vec_dec_sch_size[i], self.vec_dec_sch_depth[i], self.qtz_enable, self.qtz_int_max)
            elif self.vec_dec_sch[i] == 'G':
                self.dec_sc_g(self.vec_dec_sch_size[i], self.vec_dec_sch_depth[i], self.qtz_enable, self.qtz_int_max, self.qtz_int_min)
            elif self.vec_dec_sch[i] == 'C':
                self.dec_sc_c(self.vec_dec_sch_size[i], self.vec_dec_sch_depth[i], self.vec_dec_sch_dir[i])
            elif self.vec_dec_sch[i] == 'R0':
                if(self.vec_dec_sch_dir[i] == 0):
                    self.mem_beta_l[0][0] = 0
                else:
                    self.mem_beta_r[0][0] = 0
            elif self.vec_dec_sch[i] == 'R1':
                # flip_value = 1 if info_ctr == next_flip_idx else 0
                if(info_ctr == next_flip_idx):
                    self.vec_decoded_kr[info_ctr] = 1 if self.mem_alpha[0][0] >= 0 else 0
                else:
                    self.vec_decoded_kr[info_ctr] = 0 if self.mem_alpha[0][0] >= 0 else 1

                if self.vec_dec_sch_dir[i] == 0:
                    self.mem_beta_l[0][0] = self.vec_decoded_kr[info_ctr]
                else:
                    self.mem_beta_r[0][0] = self.vec_decoded_kr[info_ctr]
                info_ctr += 1