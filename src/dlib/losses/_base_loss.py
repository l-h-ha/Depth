from abc import ABC, abstractmethod
from typing import Optional

from .. import Tensor
from .._tensor import _sum_to_shape

import numpy as np

class base_loss(ABC):
    def __call__(self, y_true: Tensor, y_pred: Tensor) -> Tensor:
        L = self.call(y_true, y_pred)

        if L.requires_grad:
            def _backward() -> None:
                true_grad, pred_grad = self.backward(y_true, y_pred)
                
                if y_true.requires_grad:
                    y_true.grad += _sum_to_shape(L.grad * true_grad, y_true.shape)

                if y_pred.requires_grad:
                    y_pred.grad += _sum_to_shape(L.grad * pred_grad, y_pred.shape)
            L._backward = _backward

        return L
    
    @abstractmethod
    def call(self, y_true: Tensor, y_pred: Tensor) -> Tensor:
        raise NotImplementedError
    
    @abstractmethod
    def backward(self, y_true: Tensor, y_pred: Tensor) -> tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError