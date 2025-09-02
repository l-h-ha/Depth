from abc import ABC, abstractmethod
from typing import (Any)
from DepthTensor import Tensor

from ..base_component import BaseComponent

class BaseActivation(BaseComponent, ABC):
    def __init__(self) -> None:
        super().__init__()

    def build(self):
        pass

    @abstractmethod
    def call(self, *args: Any, **kwds: Any) -> Any:
        raise NotImplementedError
    
    def __call__(self, X: Tensor) -> Tensor:
        return self.call(X)