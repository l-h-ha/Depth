from .. import Tensor
from ._base_activation import base_activation

import numpy as np

class ReLU(base_activation):
    def call(self, X: Tensor) -> Tensor:
        return Tensor(data=np.maximum(0, X.data), prev=(X,), requires_grad=X.requires_grad, dtype=X.dtype)

    def backward(self, preactivation: Tensor, grad: np.ndarray) -> np.ndarray:
        return grad * np.where(preactivation.data > 0, 1, 0)