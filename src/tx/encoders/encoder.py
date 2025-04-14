from src.tx.encoders.polar_encoder import PolarEncoder
from src.tx.encoders.uncoded_encoder import UncodedEncoder

class Encoder:
    """
    A lightweight factory-wrapper that creates the correct encoder based on `code.type`.

    This class does not implement encoding itself — it delegates to an actual encoder instance.
    It exists only to simplify usage in higher-level code like `Transmitter`.

    Example:
        encoder = Encoder(code)
        output = encoder.encode(input_bits)

    Internally:
        - If code.type == 'polar', uses PolarEncoder
        - If code.type == 'uncoded', uses UncodedEncoder
        - You can extend this with new encoder types (e.g., nr5g)
    """

    def __init__(self, code):
        encoder_type = code.type.lower()
        if encoder_type == "polar":
            self.encoder = PolarEncoder(code)
        elif encoder_type == "uncoded":
            self.encoder = UncodedEncoder()
        else:
            raise ValueError(f"Unsupported encoder type: {encoder_type}")

    def encode(self, info_bits):
        return self.encoder.encode(info_bits)

    @property
    def A(self):
        return self.encoder.A

    @property
    def G(self):
        return self.encoder.G
