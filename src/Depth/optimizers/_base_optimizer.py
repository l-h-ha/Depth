from abc import ABC, abstractmethod

from .. import Tensor
class base_optimizer(ABC):
    @abstractmethod
    def call(self, parameters: list[Tensor], learning_rate: float) -> None:
        raise NotImplementedError

    def __call__(self, parameters: list[Tensor], learning_rate: float) -> None:
        self.call(parameters, learning_rate)