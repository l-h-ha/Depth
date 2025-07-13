from abc import ABC, abstractmethod

from .. import Tensor
from ..typedef import LossLike

import numpy as np

class base_model(ABC):
    @abstractmethod
    def reset_grads(self) -> None:
        raise NotImplementedError
    
    @abstractmethod
    def call(self, X: Tensor) -> Tensor:
        raise NotImplementedError

    @abstractmethod
    def fit(self, X_train: np.ndarray, Y_train: np.ndarray, loss: LossLike, epoch: int, learning_rate: float) -> None:
        pass