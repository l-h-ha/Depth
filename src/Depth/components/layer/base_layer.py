from abc import ABC, abstractmethod
from typing import (Any)
from DepthTensor import Tensor

from ..base_component import BaseComponent

class BaseLayer(BaseComponent, ABC):
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def build(self, *args: Any, **kwds: Any) -> Any:
        raise NotImplementedError

    @abstractmethod
    def call(self, *args: Any, **kwds: Any) -> Any:
        raise NotImplementedError
    
    def __call__(self, X: Tensor) -> Tensor:
        if not self._build:
            self._build = True
            self.build(X)
        return self.call(X)