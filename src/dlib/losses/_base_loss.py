from abc import ABC, abstractmethod

from .._tensor import Tensor, _sum_to_shape

import numpy as np

class base_loss(ABC):
    def __call__(self, y_true: Tensor, y_pred: Tensor) -> Tensor:
        L = self.call(y_true, y_pred)

        if L.requires_grad:
            def _backward() -> None:
                pred_grad = self.backward(y_true, y_pred)
                if y_pred.requires_grad:
                    y_pred.grad += _sum_to_shape(L.grad.reshape((L.shape[0], 1)) * pred_grad, y_pred.shape)
            L._backward = _backward

        return L
    
    @abstractmethod
    def call(self, y_true: Tensor, y_pred: Tensor) -> Tensor:
        raise NotImplementedError
    
    @abstractmethod
    def backward(self, y_true: Tensor, y_pred: Tensor) -> np.ndarray:
        raise NotImplementedError