from abc import ABC, abstractmethod
from typing import Optional

from .. import Tensor
from ..exceptions import ComponentNotBuiltError

class base_layer(ABC):
    def __init__(self) -> None:
        self.parameters: list[Tensor] = []
        self._built = False
    
    @abstractmethod
    def build(self, in_shape: tuple[int, ...]) -> None:
        raise NotImplementedError

    @abstractmethod
    def call(self, X: Tensor) -> Tensor:
        raise NotImplementedError

    def __call__(self, X: Tensor) -> Tensor:
        if not self._built:
            self._built = True
            self.build(X.shape)
        return self.call(X)