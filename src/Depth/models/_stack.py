from ._base_model import base_model
from .. import Tensor

from ..typedef import LossLike, OptimizerLike, LayerLike
from ..exceptions import BatchCalculationError

import numpy as np

class Stack(base_model):
    def __init__(self, layers: list[LayerLike]) -> None:
        super().__init__()
        self.layers = layers

    def call(self, X: Tensor) -> Tensor:
        return self.forward(X)

    def build(self) -> None:
        for layer in self.layers:
            self.parameters.extend(layer.parameters)

    def fit(self, X: np.ndarray, Y: np.ndarray, loss: LossLike, optimizer: OptimizerLike, batch_size: int, epochs: int=1, ignore_remainder: bool=True):
        num_samples = X.shape[0]

        if batch_size == -1:
            batch_size = num_samples

        if num_samples % batch_size != 0 and not ignore_remainder:
            raise BatchCalculationError(f"The amount of samples in the dataset ({num_samples}) must be divisible by batch_size ({batch_size}).")
        if Y.shape[0] % batch_size != 0 and not ignore_remainder:
            raise BatchCalculationError(f"The amount of labels in the dataset ({Y.shape[0]}) must be divisible by batch_size ({batch_size}).")

        if not self._built:
            self._built = True
            self.build()

        for epoch_n in range(epochs):
            epoch_loss = Tensor([0])
            batch_num = 0

            for batch_n in range(0, num_samples, batch_size):
                end = batch_n + batch_size
                batch_X, batch_Y = X[batch_n:end], Y[batch_n:end]

                if ignore_remainder and (batch_X.shape[0] != batch_size or batch_Y.shape[0] != batch_size):
                    continue
                batch_num += 1

                y_pred = self.call(Tensor(batch_X, dtype=batch_X.dtype))
                y_true = Tensor(data=batch_Y, dtype=batch_Y.dtype)

                L = loss(y_true, y_pred)
                optimizer.zero_grad()
                optimizer(L.backward())
                epoch_loss = epoch_loss + L

            if batch_num > 0:
                epoch_loss = epoch_loss / batch_num
                print(f"[Epoch {epoch_n} | Batch count {batch_num}] Loss: {epoch_loss.mean()[0]}")
            

    ###
    ###
    ###

    def forward(self, X: Tensor) -> Tensor:
        for layer in self.layers:
            X = layer(X)
        return X