import numpy as np
from src.rx.decoders.polar.sc import PolarDecoder_SC
from src.rx.decoders.polar.scf import PolarDecoder_SCF
from src.rx.decoders.uncoded_decoder import UncodedDecoder

def create_decoder(code):
    if code.type == "polar":
        if code.decoder == "SC".lower():
            return PolarDecoder_SC(code)
        elif code.decoder == "SCF".lower():
            return PolarDecoder_SCF(code)
        else:
            raise ValueError(f"Unsupported polar decoder type: {code.decoder}")    
    elif code.type == "uncoded":
        return UncodedDecoder()
    else:
        raise ValueError(f"Unsupported code type: {code.type}")


class Decoder:
    def __init__(self, code):
        self.decoder_type = code.type
        self.decoder = create_decoder(code)

    def __getattr__(self, name):
        # Forward method calls and attribute access to the decoder instance
        return getattr(self.decoder, name)
