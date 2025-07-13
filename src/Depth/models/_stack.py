from ._base_model import base_model
from .. import Tensor

from ..typedef import LossLike

import numpy as np

class Stack(base_model):
    def __init__(self, layers: list) -> None:
        super().__init__()
        self.layers = layers

    def reset_grads(self) -> None:
        for layer in self.layers:
            for param in layer.params:
                param.reset_grad()

    def call(self, X: Tensor) -> Tensor:
        return self.forward(X)

    def fit(self, X_train: np.ndarray, Y_train: np.ndarray, loss: LossLike, epoch: int, learning_rate: float):
        batch_num = X_train.shape[0]

        for epoch in range(epoch):
            for batch_n in range(batch_num):
                batch_train_x = X_train[batch_n]
                batch_train_x= Tensor(data=batch_train_x, requires_grad=True, dtype=batch_train_x.dtype)
                
                batch_train_y = Y_train[batch_n]
                batch_train_Y = Tensor(data=batch_train_y, dtype=batch_train_y.dtype)

                y_pred = self.call(batch_train_x)
                loss_tensor = self.backward(batch_train_Y, y_pred, loss, learning_rate)
                print(f"[Epoch {epoch} / Batch {batch_n}] Loss: {loss_tensor.mean()[0]}")


    def forward(self, X: Tensor) -> Tensor:
        for layer in self.layers:
            X = layer(X)
        return X
    
    def backward(self, Y_true: Tensor, Y_pred: Tensor, loss: 'LossLike', learning_rate: float) -> Tensor:
        self.reset_grads()
        L = loss(Y_true, Y_pred)
        L.backward()
        self.step(learning_rate)
        return L
    
    def step(self, learning_rate: float):
        for layer in self.layers:
            for param in layer.params:
                param.step(-learning_rate)
    