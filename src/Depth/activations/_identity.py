from .. import Tensor
from ._base_activation import base_activation

import numpy as np

class Identity(base_activation):
    def call(self, X: Tensor) -> Tensor:
        return Tensor(data=X.data, prev=(X,), requires_grad=X.requires_grad, dtype=X.dtype)

    def backward(self, preactivation: Tensor, grad: np.ndarray) -> np.ndarray:
        return grad * 1