from .. import Tensor
from ._base_loss import base_loss

from ..constants import EPSILON

import numpy as np

class CategorialCrossEntropy(base_loss):
    def call(self, y_true: Tensor, y_pred: Tensor) -> Tensor:
        y_pred_clipped = np.clip(y_pred.data, EPSILON, 1.-EPSILON)
        return Tensor(
            data=-np.sum((y_true.data * np.log(y_pred_clipped)), axis=-1),
            prev=(y_true, y_pred), requires_grad=y_true.requires_grad or y_pred.requires_grad, dtype=y_true.dtype
        )
    
    def backward(self, y_true: Tensor, y_pred: Tensor) -> np.ndarray:
        y_pred_clipped = np.clip(y_pred.data, EPSILON, 1.-EPSILON)
        return -(y_true.data / y_pred_clipped)