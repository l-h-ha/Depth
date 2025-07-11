from typing import Optional

from .. import Tensor
from ._base_activation import base_activation

import numpy as np

class Softmax(base_activation):
    def __init__(self, axis: Optional[int]=None, stable: bool=True):
        super().__init__()
        self.axis = axis
        self.stable = stable

    def call(self, X: Tensor) -> Tensor:
        axis = self.axis if self.axis is not None else -1
        e_x = np.exp(X.data) if not self.stable else np.exp(X.data - np.max(X.data, axis=axis, keepdims=True))
        return Tensor(data=e_x / np.sum(e_x, axis=axis, keepdims=True), prev=(X,), requires_grad=X.requires_grad, dtype=X.dtype)

    def backward(self, preactivation: Tensor, grad: np.ndarray) -> np.ndarray:
        s = self.activated
        if s is None:
            raise RuntimeError("Loss function must be called before differentiating.")
        axis = self.axis if self.axis is not None else len(s.shape) - 1
        return s * (grad - np.sum(grad * s, axis=axis, keepdims=True))