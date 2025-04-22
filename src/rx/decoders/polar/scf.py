# import numpy as np
# from src.rx.decoders.polar.sc import PolarDecoder_SC  
# from src.coding.crc.crc import instantiate_crcs

# class PolarDecoder_SCF(PolarDecoder_SC):
#     def __init__(self, code):
#         super().__init__(code)
#         # SCF-specific initializations
#         self.len_r = code.len_r
#         self.len_k = code.len_k
#         self.len_kr = code.len_kr
#         self.crc_poly_bin = instantiate_crcs(self.len_r)
#         self.max_flips = code.max_flips
#         self.vec_leaf_mag = np.empty(self.len_kr, dtype=float)
#         self.sorted_indices = np.empty(self.max_flips, dtype=int)
        
#     def decode_chain(self, vec_decoded, vec_llr):
#         self.dec_scf(vec_decoded, vec_llr)

#     def dec_scf(self, vec_decoded, vec_llr):
#         # Run initial SC decoding
#         self.dec_sc_flip_init(vec_decoded, vec_llr)
#         crc_pass = self.crc_decode(vec_decoded, self.crc_poly_bin, self.len_k)
#         iter_scf = 1
#         if(crc_pass == False and self.max_flips > 0):
#             self.sorted_indices = self.get_sorted_indices(self.vec_leaf_mag)
#             flip_iter = 0
#             while(crc_pass == False and flip_iter < self.max_flips):
#                 next_flip_idx = self.sorted_indices[flip_iter]
#                 self.dec_sc_with_flip(vec_decoded, vec_llr)
#                 iter_scf += 1
#                 crc_pass = self.crc_decode(vec_decoded, self.crc_poly_bin, self.len_k)
#                 flip_iter += 1
        
#     def crc_decode(self, vec_decoded, CRC_bin, len_k):
#         vec_dec_crc = vec_decoded.copy()
#         for i in range(len_k):
#             if vec_dec_crc[i] != 0:
#                 for j in range(len(CRC_bin)):
#                     vec_dec_crc[i + j] ^= CRC_bin[j]
        
#         crc_pass = all(x == 0 for x in vec_dec_crc)
#         return crc_pass
    
#     def get_sorted_indices(vector):
#         return sorted(range(len(vector)), key=lambda x: vector[x])

#     # def dec_sc_h_flip(self, llr, stage_dir):
#     #     if(stage_dir == 0):
#     #         self.mem_beta_l[0][0] = 0 if llr < 0 else 1
#     #     else:
#     #         self.mem_beta_r[0][0] = 0 if llr < 0 else 1

#     def dec_sc_flip_init(self, vec_decoded, vec_llr):
#         """
#         IMPORTANT: This function can only be used when the fast_enable is set to False.
#         Fast-SCF algorithm is TBD.
#         """
#         self.mem_alpha[self.len_logn][:] = vec_llr # Place LLRs to bottom row of mem_alpha
#         info_ctr = 0
#         for i in range(len(self.vec_dec_sch)):
#             if self.vec_dec_sch[i] == 'F':
#                 self.dec_sc_f(self.vec_dec_sch_size[i], self.vec_dec_sch_depth[i], self.qtz_enable, self.qtz_int_max)
#             elif self.vec_dec_sch[i] == 'G':
#                 self.dec_sc_g(self.vec_dec_sch_size[i], self.vec_dec_sch_depth[i], self.qtz_enable, self.qtz_int_max, self.qtz_int_min)
#             elif self.vec_dec_sch[i] == 'C':
#                 self.dec_sc_c(self.vec_dec_sch_size[i], self.vec_dec_sch_depth[i], self.vec_dec_sch_dir[i])
#             elif self.vec_dec_sch[i] == 'R0':
#                 if(self.vec_dec_sch_dir[i] == 0):
#                     self.mem_beta_l[0][0] = 0
#                 else:
#                     self.mem_beta_r[0][0] = 0
#             elif self.vec_dec_sch[i] == 'R1':
#                 self.vec_leaf_mag[info_ctr] = np.abs(self.mem_alpha[0][0])
#                 if(self.vec_dec_sch_dir[i] == 0):
#                     vec_decoded[info_ctr] = self.mem_beta_l[0][0]
#                 else:
#                     vec_decoded[info_ctr] = self.mem_beta_r[0][0]
#                 info_ctr += 1
                



#     # def attempt_flips(self, llr):
#     #     # Implement SCF-specific flipping logic
#     #     pass

#     # def is_valid(self, decoded_bits):
#     #     # e.g., CRC check
#     #     pass
