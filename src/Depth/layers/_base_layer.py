from abc import ABC, abstractmethod
from .. import Tensor

class base_layer(ABC):
    def __init__(self) -> None:
        self._built = False
        self.params: list[Tensor] = []
    
    @abstractmethod
    def build(self, input_shape: tuple) -> None:
        raise NotImplementedError

    @abstractmethod
    def call(self, X: Tensor) -> Tensor:
        raise NotImplementedError

    def __call__(self, X: Tensor) -> Tensor:
        if not self._built:
            self.build(X.shape)
            self._built = True
        return self.call(X)