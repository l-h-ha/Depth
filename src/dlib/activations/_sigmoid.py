from .. import Tensor
from ._base_activation import base_activation

import numpy as np

class Sigmoid(base_activation):
    def __init__(self) -> None:
        super().__init__()

    def call(self, X: Tensor) -> Tensor:
        return Tensor(data=1 / (1 + np.exp(-X.data)), prev=(X,), requires_grad=X.requires_grad, dtype=X.dtype)

    def backward(self, preactivation: Tensor, grad: np.ndarray) -> np.ndarray:
        if self.activated is None:
            raise RuntimeError("Loss function must be called before differentiating.")
        return grad * self.activated.data * (1 - self.activated.data)