from src.configs.config_code import CodeConfig
from src.coding.uncoded import Uncoded

class Code:
    def __init__(self, config: CodeConfig):
        self.type = config.type.lower()
        self.len_k = config.len_k
        self.decoder = None

        if self.type == "polar":
            self.code = config.config  # This is a PolarCodeConfig instance
            self.decoder = self.code.decoder.algorithm.lower()

        elif self.type == "uncoded":
            # You can define UncodedConfig later or pass len_k
            self.code = Uncoded(len_k=self.len_k)

        else:
            raise ValueError(f"Unsupported code type: {self.type}")

    def __getattr__(self, name):
        return getattr(self.code, name)
