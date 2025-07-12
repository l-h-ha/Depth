from ._base_initializer import base_initializer

import numpy as np

class Xavier(base_initializer):
    def __init__(self, requires_grad: bool=True, dtype: np.typing.DTypeLike=np.float32) -> None:
        super().__init__(requires_grad=requires_grad, dtype=dtype)

    def call(self, input_shape: tuple[int, ...]) -> np.ndarray:
        limit = np.sqrt(6.0 / (sum(input_shape)))
        return np.random.uniform(low=-limit, high=limit, size=input_shape)