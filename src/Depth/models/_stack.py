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

    def fit(self, X: np.ndarray, Y: np.ndarray, loss: LossLike, optimizer: OptimizerLike, batch_size: int, learning_rate: float, epochs: int=1, ignore_remainder: bool=True, logging: bool=True):
        num_samples = X.shape[0]
        num_labels = Y.shape[0]

        if num_labels != num_samples:
            raise BatchCalculationError(f"The number of samples ({num_samples}) and labels ({num_labels}) must match.")
        if num_samples % batch_size != 0 and not ignore_remainder:
            raise BatchCalculationError(f"The amount of samples in the dataset ({num_samples}) must be divisible by batch_size ({batch_size}).")
        if num_labels % batch_size != 0 and not ignore_remainder:
            raise BatchCalculationError(f"The amount of labels in the dataset ({num_labels}) must be divisible by batch_size ({batch_size}).")

        batch_num = num_samples // batch_size

        for epoch in range(epochs):
            permutation = np.random.permutation(num_samples)
            X_shuffled, Y_shuffled = X[permutation], Y[permutation]

            epoch_loss = Tensor([0])

            for batch_n in range(batch_num):
                start = batch_n * batch_size
                end = start + batch_size
                batch_X, batch_Y = X_shuffled[start:end], Y_shuffled[start:end]

                if ignore_remainder and (batch_X.shape[0] != batch_size or batch_Y.shape[0] != batch_size):
                    continue

                y_pred = self.forward(Tensor(batch_X, dtype=batch_X.dtype))
                y_true = Tensor(data=batch_Y, dtype=batch_Y.dtype)

                L = loss(y_true, y_pred)
                parameters = L.backward()
                optimizer(parameters, learning_rate)
                epoch_loss += L
            
            processed_batch_n = batch_num if not ignore_remainder else (num_samples // batch_size)
            epoch_loss /= processed_batch_n
            if logging:
                print(f"[Epoch: {epoch + 1}/{epochs} | Batch processed: {processed_batch_n}] Loss: {epoch_loss.mean()[0]}")
            

    ###
    ###
    ###

    def forward(self, X: Tensor) -> Tensor:
        for layer in self.layers:
            X = layer(X)
        return X