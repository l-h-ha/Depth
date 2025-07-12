from ._base_initializer import base_initializer

import numpy as np

class Xavier(base_initializer):
    def __init__(self, requires_grad: bool=True, dtype: np.typing.DTypeLike=np.float32) -> None:
        super().__init__(requires_grad=requires_grad, dtype=dtype)

    def call(self, input_shape: tuple[int, ...]) -> np.ndarray:
        return np.random.randn(*input_shape) * np.sqrt(2 / (input_shape[0] + input_shape[1]))