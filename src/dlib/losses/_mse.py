from .._tensor import Tensor
from ._base_loss import base_loss

import numpy as np

class MSE(base_loss):
    def call(self, y_true: Tensor, y_pred: Tensor) -> Tensor:
        return Tensor(
            data=np.mean(np.square(y_true.data - y_pred.data)),
            prev=(y_true, y_pred), requires_grad=y_true.requires_grad or y_pred.requires_grad, dtype=y_true.dtype
        )
    
    def backward(self, y_true: Tensor, y_pred: Tensor) -> tuple[np.ndarray, np.ndarray]:
        L = y_true.data - y_pred.data
        return (2 / y_true.size) * L, (2 / y_pred.size) * -L 