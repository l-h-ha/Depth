from typing import Any
from DepthTensor import Tensor

from .base_loss import BaseLoss

class MeanSquaredError(BaseLoss):
    def __init__(self) -> None:
        super().__init__()

    def call(self, y_true: Tensor, y_pred: Tensor) -> Tensor:
        return (y_true - y_pred)**2
    
    def backward(self) -> Any:
        pass