'''
https://arxiv.org/pdf/1708.02002 "Focal Loss for Dense Object Detection"
'''

from .. import Tensor, EPSILON
from ._base_loss import base_loss

import numpy as np

class FocalLoss(base_loss):
    def __init__(self, alpha: float=0.25, gamma: float=2, axis: int=-1):
        self.alpha = alpha
        self.gamma = gamma
        self.axis = axis

    def call(self, y_true: Tensor, y_pred: Tensor) -> Tensor:
        p_t = np.sum(y_true.data * y_pred.data, axis=self.axis, keepdims=True)
        p_t = np.clip(p_t, EPSILON, 1.-EPSILON)
        
        '''
        FL = -a(1 - p_t)^g * ln(p_t)
        '''
        return Tensor(
            data=-self.alpha * (1 - p_t)**self.gamma * np.log(p_t),
            prev=(y_true, y_pred), requires_grad=y_true.requires_grad or y_pred.requires_grad, dtype=y_true.dtype
        )
    
    def backward(self, y_true: Tensor, y_pred: Tensor) -> np.ndarray:
        '''
        The paper never diffrentiated dFL/dp_t, only dFL/dx (logits).
        Deriving the former we get:
        dFL/dp_t =a(1-p)**(g-1)(g*log(p)- (1-p)/p)
        '''
        p_t = np.sum(y_true.data * y_pred.data, axis=self.axis, keepdims=True)
        p_t = np.clip(p_t, EPSILON, 1.-EPSILON)
        return y_true * self.alpha * (1 - p_t)**(self.gamma - 1) * (self.gamma * np.log(p_t) - (1 - p_t) / p_t)