from ._base_layer import base_layer
from .. import Tensor

class Input(base_layer):
    def __init__(self, shape: tuple) -> None:
        super().__init__()
        self.shape = shape

    def build(self, input_shape: tuple) -> None:
        pass

    def call(self, X: Tensor) -> Tensor:
        if X.shape != self.shape:
            raise ValueError(f"Input data shape mismatch. Expected shape: {self.shape}, got: {X.shape}")
        return X