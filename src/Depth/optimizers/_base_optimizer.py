from abc import ABC, abstractmethod
from typing import Optional

from .. import Tensor

class base_optimizer(ABC):
    def __init__(self, learning_rate: float=0.01) -> None:
        self.learning_rate = learning_rate
        self.parameters: Optional[list[Tensor]] = None

    @abstractmethod
    def call(self, parameters: list[Tensor]) -> None:
        pass

    def zero_grad(self):
        if self.parameters is None:
            return
        for param in self.parameters:
            param.reset_grad()

    def __call__(self, parameters: list[Tensor]) -> None:
        self.call(parameters)