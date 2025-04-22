from dataclasses import dataclass
from typing import Union, Optional
from src.configs.config_polarcode import PolarCodeConfig
from src.configs.config_polarcode_decoder import PolarDecoderConfig
from src.configs.config_polarcode_fast import PolarFastConfig
from src.configs.config_quantization import QuantizeConfig
from src.configs.config_crc import CRCConfig
from src.configs.config_uncoded import UncodedConfig

@dataclass
class CodeConfig:
    type: str # Type of coding, i.e. Polar, LDPC, uncoded, etc.
    len_k: int # number of information bits to be passed to the encoder
    config: Union[PolarCodeConfig, None] = None #Add new code configs here

    @staticmethod
    def from_dict(data: dict) -> 'CodeConfig':
        type_ = data["type"]
        len_k = data["len_k"]

        if type_ == "POLAR":
            polar_data = data["polar"]

            # Flatten fast_mode block
            fast_mode = PolarFastConfig(
                enable      = polar_data["fast_mode"]["enable"],
                max_rate0   = polar_data["fast_mode"]["max_rate0"],
                max_rate1   = polar_data["fast_mode"]["max_rate1"],
                max_rep     = polar_data["fast_mode"]["max_rep"],
                max_spc     = polar_data["fast_mode"]["max_spc"],
                max_ml_0011 = polar_data["fast_mode"]["max_ml_0011"],
                max_ml_0101 = polar_data["fast_mode"]["max_ml_0101"]
            )

            polar_config = PolarCodeConfig(
                len_k=len_k,
                polar_file=polar_data["polar_file"],
                crc=CRCConfig(**polar_data["crc"]),
                decoder=PolarDecoderConfig(**polar_data["decoder"]),
                quantize=QuantizeConfig(**polar_data["quantize"]),
                fast_mode=fast_mode
            )
            return CodeConfig(type=type_, len_k=len_k, config=polar_config)

        elif type_ == "UNCODED":
            uncoded_config = UncodedConfig(
                len_k=data["len_k"],
                len_n=data.get("len_n", None)
            )
            return CodeConfig(type=type_, len_k=len_k, config=uncoded_config)

        else:
            raise ValueError(f"Unsupported code type: {type_}")