from ._base_model import base_model
from .. import Tensor

from .. import typing as typ

class Stack(base_model):
    def __init__(self, layers: list) -> None:
        super().__init__()
        self.layers = layers

    def forward(self, X: Tensor):
        for layer in self.layers:
            X = layer(X)
        return X
    
    def backward(self, Y_true: Tensor, Y_pred: Tensor, loss: typ.LossLike, learning_rate: float) -> Tensor:
        self.reset_grads()
        L = loss(Y_true, Y_pred)
        L.backward()
        self.step(learning_rate)
        return L
    
    def step(self, learning_rate: float):
        for layer in self.layers:
            for param in layer.params:
                param.step(-learning_rate)
    
    def reset_grads(self) -> None:
        for layer in self.layers:
            for param in layer.params:
                param.reset_grad()