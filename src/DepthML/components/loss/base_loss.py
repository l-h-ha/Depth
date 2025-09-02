from abc import ABC, abstractmethod
from typing import Any

from DepthTensor import Tensor

from ..base_component import BaseComponent

class BaseLoss(BaseComponent, ABC):
    def __init__(self) -> None:
        super().__init__()
    
    @abstractmethod
    def call(self, y_true: Tensor, y_pred: Tensor) -> Tensor:
        raise NotImplementedError
    
    @abstractmethod
    def backward(self, *args: Any, **kwds: Any) -> Any:
        raise NotImplementedError
    
    def build(self) -> Any:
        pass

    def __call__(self, y_true: Tensor, y_pred: Tensor) -> Tensor:
        return self.call(y_true, y_pred)