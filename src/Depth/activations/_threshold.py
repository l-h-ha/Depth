from .. import Tensor
from ._base_activation import base_activation

import numpy as np

class Threshold(base_activation):
    def __init__(self, threshold: float, value: float):
        self.threshold = threshold
        self.value = value

    def call(self, X: Tensor) -> Tensor:
        return Tensor(data=np.where(X.data > self.threshold, X.data, self.value), prev=(X,), requires_grad=X.requires_grad, dtype=X.dtype)

    def backward(self, preactivation: Tensor, grad: np.ndarray) -> np.ndarray:
        return grad * np.where(preactivation.data > self.threshold, 1, 0)