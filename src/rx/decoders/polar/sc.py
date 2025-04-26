import numpy as np
import math
from numba import njit
from src.configs.config_polarcode import PolarCodeConfig

class PolarDecoder_SC():
    def __init__(self, config: PolarCodeConfig):
        self.len_logn = int(config.len_logn)
        self.vec_polar_isfrozen = config.frozen_bits
        self.info_indices = config.info_indices

        self.qtz_enable  = config.quantize.enable
        self.qtz_int_max = config.quantize.intl_max
        self.qtz_int_min = config.quantize.intl_min
        
        self.fast_enable      = config.fast_mode.enable
        self.nodesize_rate0   = config.fast_mode.max_rate0   if self.fast_enable else 0
        self.nodesize_rate1   = config.fast_mode.max_rate1   if self.fast_enable else 0
        self.nodesize_rep     = config.fast_mode.max_rep     if self.fast_enable else 0
        self.nodesize_spc     = config.fast_mode.max_spc     if self.fast_enable else 0
        self.nodesize_ml_0011 = config.fast_mode.max_ml_0011 if self.fast_enable else 0
        self.nodesize_ml_0101 = config.fast_mode.max_ml_0101 if self.fast_enable else 0
        
        self.mem_alpha = None
        self.mem_beta_l = None
        self.mem_beta_r = None
        self.vec_dec_sch = []
        self.vec_dec_sch_size = []
        self.vec_dec_sch_depth = []
        self.vec_dec_sch_dir = []
        
        self.use_optimized = 1
        self.create_Gmat()

    def create_Gmat(self):
        """
        Creates the polar matrices:
            - matG_Nxk: The generator matrix in N-by-N form.
        Raises:
            TypeError: If len_logn is not a positive integer.
        """
        matG_core = np.array([[1, 0], [1, 1]])
        matG = matG_core  # Core matrix as the initial value
        for _ in range(self.len_logn-1):
            matG = np.kron(matG, matG_core)

        self.matG_Nxk = matG[:, sorted(self.info_indices)]  # Capture k columns corresponding to info indices

    def initialize_decoder(self):
        if not self.vec_dec_sch:
            self.create_decoding_schedule()
            # raise ValueError("Decoder scheduling is not created - it must be created to continue decoding")
        self.mem_alpha =  [np.zeros((2**i)) for i in range(self.len_logn + 1)]
        self.mem_beta_l = [np.zeros((2**i), dtype=np.bool_) for i in range(self.len_logn + 1)]
        self.mem_beta_r = [np.zeros((2**i), dtype=np.bool_) for i in range(self.len_logn + 1)]

    def decode_chain(self, vec_decoded, vec_llr):
        self.dec_sc(vec_decoded, vec_llr)

    def create_decoding_schedule(self):
        sch_limit = self.len_logn
        vec_dec_sch_init = ['F', 'H', 'G', 'H', 'C'] # Core scheduling
        self.vec_dec_sch = []
        # Create initial decoding schedule (F, H, G, H, C)
        self.call_decoding_schedule(vec_dec_sch_init, sch_limit)
        # Embed frozen and info indices to the schedule: H -> R0/R1
        self.embed_frozen_nodes()
        # Create key special nodes of length 2: (R0, R1 and REP only)
        self.create_key_special_nodes()
        self.create_decoding_stages()
        self.create_special_nodes()
        self.create_decoding_direction_fast()


    def call_decoding_schedule(self, base_vector, sch_limit):
        for element in base_vector:
            if(element == 'H' and sch_limit > 1):
                self.call_decoding_schedule(base_vector, sch_limit-1)
            else:
                self.vec_dec_sch.append(element)

    def embed_frozen_nodes(self):
        j = 0 
        for i in range(len(self.vec_dec_sch)):
            if(self.vec_dec_sch[i] == 'H'):
                self.vec_dec_sch[i] = 'R1' if self.vec_polar_isfrozen[j] == 0 else 'R0'
                j+=1 

    def create_decoding_stages(self):
        sch_limit = self.len_logn
        self.vec_dec_sch_size = []
        self.vec_dec_sch_depth = []
        current_stagesize = np.power(2,sch_limit)
        for i in range(len(self.vec_dec_sch)):
            if(self.vec_dec_sch[i] == 'F'):
                self.vec_dec_sch_size.append(int(current_stagesize))
                self.vec_dec_sch_depth.append(int(np.log2(current_stagesize)))
                current_stagesize /= 2
            elif(self.vec_dec_sch[i] == 'G'):
                if(self.vec_dec_sch[i-1] == 'C' or self.isLeaf(self.vec_dec_sch[i-1])):
                    self.vec_dec_sch_size.append(int(current_stagesize*2))
                    self.vec_dec_sch_depth.append(int(np.log2(current_stagesize*2)))
                else:
                    self.vec_dec_sch_size.append(int(current_stagesize))
                    self.vec_dec_sch_depth.append(int(np.log2(current_stagesize)))
                    current_stagesize *= 2
            elif(self.vec_dec_sch[i] == 'C'):
                self.vec_dec_sch_size.append(int(current_stagesize))
                self.vec_dec_sch_depth.append(int(np.log2(current_stagesize)))
                current_stagesize *= 2
            else:
                self.vec_dec_sch_size.append(int(current_stagesize))
                self.vec_dec_sch_depth.append(int(np.log2(current_stagesize)))

    def create_decoding_direction(self):
        sch_limit = self.len_logn
        combine_ctr =  [0] * (sch_limit + 1)
        hard_dec_ctr = [0] * (sch_limit + 1)
        sc_direction = []

        for i in range(len(self.vec_dec_sch)):
            if self.vec_dec_sch[i] == 'C':
                sc_direction.append(combine_ctr[math.floor(math.log2(self.vec_dec_sch_size[i]))] % 2)
                combine_ctr[math.floor(math.log2(self.vec_dec_sch_size[i]))] += 1
            elif (self.vec_dec_sch[i] == 'R0' or self.vec_dec_sch[i] == 'R1'):
                sc_direction.append(hard_dec_ctr[math.floor(math.log2(self.vec_dec_sch_size[i]))] % 2)
                hard_dec_ctr[math.floor(math.log2(self.vec_dec_sch_size[i]))] += 1
            else:
                sc_direction.append(0)
        
        self.vec_dec_sch_dir = sc_direction

    def create_key_special_nodes(self):
        i = 0
        vec_dec_sch_fast = []
        while i < len(self.vec_dec_sch):
            pattern = ''.join(self.vec_dec_sch[i:i+5])

            if (pattern == "FR0GR0C" and self.nodesize_rate0 >= 4):  
                vec_dec_sch_fast.append("R0")
                i += 4
            elif (pattern == "FR1GR1C" and self.nodesize_rate1 >= 4): 
                vec_dec_sch_fast.append("R1")
                i += 4
            elif (pattern == "FR0GR1C" and self.nodesize_rep >= 4): 
                vec_dec_sch_fast.append("REP")
                i += 4
            else:
                vec_dec_sch_fast.append(self.vec_dec_sch[i])
            i += 1
        # return vec_dec_sch_fast
        self.vec_dec_sch = vec_dec_sch_fast

    def create_special_nodes(self):
        iterator_schedule = list(self.vec_dec_sch)
        self.vec_dec_sch.clear()

        iterator_stagesize = list(self.vec_dec_sch_size)
        self.vec_dec_sch_size.clear()

        iterator_stageidx = list(self.vec_dec_sch_depth)
        self.vec_dec_sch_depth.clear()

        i = 0
        while i < len(iterator_schedule):
            pattern = ''.join(iterator_schedule[i:i+5])

            self.vec_dec_sch_size.append(iterator_stagesize[i])
            self.vec_dec_sch_depth.append(iterator_stageidx[i])

            if pattern == "FR0GREPC" and self.nodesize_rep >= 4:
                self.vec_dec_sch.append("REP")
                i += 4
            elif pattern == "FR0GR0C" and self.nodesize_rate0 >= 4:
                self.vec_dec_sch.append("R0")
                i += 4
            elif pattern == "FR1GR1C" and self.nodesize_rate1 >= 4:
                self.vec_dec_sch.append("R1")
                i += 4
            elif pattern == "FREPGR1C" and self.nodesize_spc >= 4:
                self.vec_dec_sch.append("SPC")
                i += 4
            elif pattern == "FR0GR1C" and self.nodesize_ml_0011 == 4:
                self.vec_dec_sch.append("ML_0011")
                i += 4
            elif pattern == "FREPGREPC" and self.nodesize_ml_0101 == 4:
                self.vec_dec_sch.append("ML_0101")
                i += 4
            else:
                self.vec_dec_sch.append(iterator_schedule[i])

            i += 1

        there_is_still_hope = True

        while there_is_still_hope:
            istherehope = False

            iterator_schedule = list(self.vec_dec_sch)
            self.vec_dec_sch.clear()

            iterator_stagesize = list(self.vec_dec_sch_size)
            self.vec_dec_sch_size.clear()

            iterator_stageidx = list(self.vec_dec_sch_depth)
            self.vec_dec_sch_depth.clear()

            i = 0
            while i < len(iterator_schedule):
                pattern = ''.join(iterator_schedule[i:i+5])

                self.vec_dec_sch_size.append(iterator_stagesize[i])
                self.vec_dec_sch_depth.append(iterator_stageidx[i])

                if pattern == "FR0GREPC" and self.nodesize_rep >= iterator_stagesize[i]:
                    self.vec_dec_sch.append("REP")
                    i += 4
                    istherehope = True
                elif pattern == "FR0GR0C" and self.nodesize_rate0 >= iterator_stagesize[i]:
                    self.vec_dec_sch.append("R0")
                    i += 4
                    istherehope = True
                elif pattern == "FSPCGR1C" and self.nodesize_spc >= iterator_stagesize[i]:
                    self.vec_dec_sch.append("SPC")
                    i += 4
                    istherehope = True
                elif pattern == "FR1GR1C" and self.nodesize_rate1 >= iterator_stagesize[i]:
                    self.vec_dec_sch.append("R1")
                    i += 4
                    istherehope = True
                else:
                    self.vec_dec_sch.append(iterator_schedule[i])

                i += 1

            if not istherehope:
                there_is_still_hope = False

    def create_decoding_direction_fast(self):
        
        self.vec_dec_sch_dir = []
        directionStack = [0]

        for iteration, item in enumerate(self.vec_dec_sch):
            if item == "F":
                directionStack.append(0)
                self.vec_dec_sch_dir.append(0)
            elif item == "G":
                directionStack.append(1)
                self.vec_dec_sch_dir.append(1)
            elif self.isLeaf(item) or item == "C":
                self.vec_dec_sch_dir.append(directionStack.pop())
        
        return self.vec_dec_sch_dir

    def isLeaf(self, node):
        return node == 'R0' or node == 'R1' or node == 'REP' or node == 'SPC' or node == 'ML_0011' or node == 'ML_0101'
    
    def dec_sc_f(self, stage_size, stage_depth, is_quantized, max_value):
        # if self.use_optimized:
            return self.dec_sc_f_optimized(stage_size, stage_depth, is_quantized, max_value)
        # else:
        #     return self.dec_sc_f_legacy(stage_size, stage_depth, is_quantized, max_value)

    
    def dec_sc_f_optimized(self, stage_size, stage_depth, is_quantized, max_value):
        """
        Perform F-node (function node) computation.

        Args:
            stage_size (int): The size of the current stage.
            stage_depth (int): The depth of the current stage in the decoding tree.
        """
        mid_point = stage_size // 2
        llr_a = self.mem_alpha[stage_depth][:mid_point]
        llr_b = self.mem_alpha[stage_depth][mid_point:]

        # Use the Numba-optimized function
        result = dec_sc_f_numba(llr_a, llr_b, is_quantized, max_value)

        # Store the result back
        self.mem_alpha[stage_depth - 1][:mid_point] = result

    def dec_sc_f_legacy(self, stage_size, stage_depth, is_quantized, max_value):    
        """
        Perform F-node (function node) computation.

        Args:
            stage_size (int): The size of the current stage.
            stage_depth (int): The depth of the current stage in the decoding tree.
        """
        mid_point = stage_size // 2
        llr_a = self.mem_alpha[stage_depth][:mid_point]
        llr_b = self.mem_alpha[stage_depth][mid_point:]
        
        abs_llr = np.minimum(np.abs(llr_a), np.abs(llr_b))
        sign = np.sign(llr_a * llr_b)
        result = abs_llr * sign

        if is_quantized: #Only possible breach is when the result is 2^q and in 2s complement form.
            result = np.minimum(max_value, result)
        
        self.mem_alpha[stage_depth - 1][:mid_point] = result

    def dec_sc_g(self, stage_size, stage_depth, is_quantized, max_value, min_value):
        # if self.use_optimized:
            return self.dec_sc_g_optimized(stage_size, stage_depth, is_quantized, max_value, min_value)
        # else:
        #     return self.dec_sc_g_legacy(stage_size, stage_depth, is_quantized, max_value, min_value)


    def dec_sc_g_legacy(self, stage_size, stage_depth, is_quantized, max_value, min_value):
        """
        Perform G-node (function node) computation.

        Args:
            stage_size (int): The size of the current stage.
            stage_depth (int): The depth of the current stage in the decoding tree.
            is_quantized (bool): Whether the LLRs are quantized
            max_value (int): Max quantization value
            min_value (int): Min quantization value
        """
        mid_point = stage_size // 2

        llr_a = self.mem_alpha[stage_depth][:mid_point]
        llr_b = self.mem_alpha[stage_depth][mid_point:]

        mem_beta_slice = self.mem_beta_l[stage_depth - 1][:mid_point]
        mem_alpha_slice = np.where(mem_beta_slice == 0, llr_b + llr_a, llr_b - llr_a)        

        if is_quantized:
            mem_alpha_slice = np.clip(mem_alpha_slice, min_value, max_value)

        self.mem_alpha[stage_depth - 1][:mid_point] = mem_alpha_slice

    def dec_sc_g_optimized(self, stage_size, stage_depth, is_quantized, max_value, min_value):
        mid_point = stage_size // 2
        llr_a = self.mem_alpha[stage_depth][:mid_point]
        llr_b = self.mem_alpha[stage_depth][mid_point:]
        mem_beta_slice = self.mem_beta_l[stage_depth - 1][:mid_point]

        # Use the Numba-compiled function
        self.mem_alpha[stage_depth - 1][:mid_point] = dec_sc_g_numba(
            llr_a, llr_b, mem_beta_slice, is_quantized, max_value, min_value
        )

    def dec_sc_c(self, stage_size, stage_depth, stage_dir):
        """
        Perform C-node (function node) computation.

        Args:
            stage_size (int): The size of the current stage.
            stage_depth (int): The depth of the current stage in the decoding tree.
            stage_dir (bool): The direction of the current stage.
        """
        beta_src1 = self.mem_beta_l[stage_depth][:stage_size]
        beta_src2 = self.mem_beta_r[stage_depth][:stage_size]
        beta_src1_int = beta_src1.astype(int)
        beta_src2_int = beta_src2.astype(int)
        
        if(stage_dir == 0):
            self.mem_beta_l[stage_depth + 1][:stage_size] = np.bitwise_xor(beta_src1_int, beta_src2_int)
            self.mem_beta_l[stage_depth + 1][stage_size:] = beta_src2_int
        else:
            self.mem_beta_r[stage_depth + 1][:stage_size] = np.bitwise_xor(beta_src1_int, beta_src2_int)
            self.mem_beta_r[stage_depth + 1][stage_size:] = beta_src2_int

    def dec_sc_h(self, llr, stage_dir):
        """
        Perform H-node (function node) computation.

        Args:
            llr (double): The LLR value to decode.
            stage_dir (bool): The direction of the current stage.
        """
        if(stage_dir == 0):
            self.mem_beta_l[0][0] = 1 if llr < 0 else 0
        else:
            self.mem_beta_r[0][0] = 1 if llr < 0 else 0

    def dec_fastssc_r0(self, stage_depth, stage_dir):
        if stage_dir == 0:
            self.mem_beta_l[stage_depth][:] = 0
        else:
            self.mem_beta_r[stage_depth][:] = 0

    def dec_fastssc_rep(self, stage_depth, sch_dir):
        rep_sum = np.sum(self.mem_alpha[stage_depth][:])
        rep_sgn = rep_sum < 0
        if sch_dir == 0:
            self.mem_beta_l[stage_depth][:] = rep_sgn
        else:
            self.mem_beta_r[stage_depth][:] = rep_sgn

    def dec_fastssc_r1(self, stage_depth, sch_dir):
        if sch_dir == 0:
            self.mem_beta_l[stage_depth][:] = np.where(self.mem_alpha[stage_depth][:] < 0, 1, 0)
        else:
            self.mem_beta_r[stage_depth][:] = np.where(self.mem_alpha[stage_depth][:] < 0, 1, 0)

    def dec_fastssc_spc(self, stage_depth, sch_dir):
        hard_decisions = np.where(self.mem_alpha[stage_depth] < 0, 1, 0)
        parity = np.sum(hard_decisions) % 2

        if parity:
            min_entry_index = np.argmin(np.abs(self.mem_alpha[stage_depth][:]))
            hard_decisions[min_entry_index] ^= 1

        if sch_dir == 0:
            self.mem_beta_l[stage_depth][:] = hard_decisions
        else:
            self.mem_beta_r[stage_depth][:] = hard_decisions

    def dec_sc(self, vec_decoded, vec_llr):
        """
        Perform the decoding process using the SC algorithm.
        Args:
            vec_decoded (np.ndarray): The decoded bits.
            vec_llr (np.ndarray): The log-likelihood ratios.
            matG_full (np.ndarray): The full generator matrix.
            vec_polar_info_indices (np.ndarray): The information bit indices.
        """
        self.mem_alpha[self.len_logn][:] = vec_llr # Place LLRs to bottom row of mem_alpha
        for i in range(len(self.vec_dec_sch)):
            if self.vec_dec_sch[i] == 'F':
                self.dec_sc_f(self.vec_dec_sch_size[i], self.vec_dec_sch_depth[i], self.qtz_enable, self.qtz_int_max)
            elif self.vec_dec_sch[i] == 'G':
                self.dec_sc_g(self.vec_dec_sch_size[i], self.vec_dec_sch_depth[i], self.qtz_enable, self.qtz_int_max, self.qtz_int_min)
            elif self.vec_dec_sch[i] == 'C':
                self.dec_sc_c(self.vec_dec_sch_size[i], self.vec_dec_sch_depth[i], self.vec_dec_sch_dir[i])
            elif self.vec_dec_sch[i] == 'R0':
                self.dec_fastssc_r0(self.vec_dec_sch_depth[i], self.vec_dec_sch_dir[i])
            elif self.vec_dec_sch[i] == 'REP':
                self.dec_fastssc_rep(self.vec_dec_sch_depth[i], self.vec_dec_sch_dir[i])
            elif self.vec_dec_sch[i] == 'R1':
                self.dec_fastssc_r1(self.vec_dec_sch_depth[i], self.vec_dec_sch_dir[i])
            elif self.vec_dec_sch[i] == 'SPC':
                self.dec_fastssc_spc(self.vec_dec_sch_depth[i], self.vec_dec_sch_dir[i])

        vec_decoded[:] = ((self.mem_beta_l[self.len_logn] @ self.matG_Nxk) % 2)
    






@njit
def dec_sc_f_numba(llr_a, llr_b, is_quantized, max_value):
    """
    Perform F-node (function node) computation using JIT compilation.

    Args:
        llr_a (np.ndarray): First set of LLRs (log-likelihood ratios).
        llr_b (np.ndarray): Second set of LLRs.
        is_quantized (bool): Whether quantization is applied.
        max_value (float): Maximum value allowed during quantization.

    Returns:
        np.ndarray: Computed F-node results.
    """
    mid_point = llr_a.shape[0]
    result = np.empty(mid_point, dtype=llr_a.dtype)

    for i in range(mid_point):
        abs_llr = min(abs(llr_a[i]), abs(llr_b[i]))
        sign = np.sign(llr_a[i] * llr_b[i])  # No need to multiply here. Profile later.
        result[i] = abs_llr * sign

        if is_quantized:
            result[i] = min(max_value, result[i])

    return result


@njit
def dec_sc_g_numba(llr_a, llr_b, mem_beta_slice, is_quantized, max_value, min_value):
    mid_point = llr_a.shape[0]
    mem_alpha_slice = np.empty(mid_point, dtype=llr_a.dtype)

    for i in range(mid_point):
        if mem_beta_slice[i] == 0:
            mem_alpha_slice[i] = llr_b[i] + llr_a[i]
        else:
            mem_alpha_slice[i] = llr_b[i] - llr_a[i]

        if is_quantized:
            # In-place quantization
            if mem_alpha_slice[i] > max_value:
                mem_alpha_slice[i] = max_value
            elif mem_alpha_slice[i] < min_value:
                mem_alpha_slice[i] = min_value

    return mem_alpha_slice




