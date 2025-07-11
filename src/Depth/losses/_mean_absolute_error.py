from .. import Tensor
from ._base_loss import base_loss

import numpy as np

class MeanAbsoluteError(base_loss):
    def call(self, y_true: Tensor, y_pred: Tensor) -> Tensor:
        return Tensor(
            data=np.sum(np.abs(y_true.data - y_pred.data), axis=-1),
            prev=(y_true, y_pred), requires_grad=y_true.requires_grad or y_pred.requires_grad, dtype=y_true.dtype
        )
    
    def backward(self, y_true: Tensor, y_pred: Tensor) -> np.ndarray:
        return np.sign(y_pred.data)