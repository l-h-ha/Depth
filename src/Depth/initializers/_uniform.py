from ._base_initializer import base_initializer

import numpy as np

class Uniform(base_initializer):
    def __init__(self, low: float=-0.05, high: float=0.05, requires_grad: bool=True, dtype: np.typing.DTypeLike=np.float32) -> None:
        super().__init__(requires_grad=requires_grad, dtype=dtype)
        self.low = low
        self.high = high

    def call(self, input_shape: tuple[int, ...]) -> np.ndarray:
        return np.random.uniform(low=self.low, high=self.high, size=input_shape)