from abc import ABC, abstractmethod

from .. import Tensor

class base_model(ABC):
    def __init__(self) -> None:
        self.parameters: list[Tensor] = []
    
    @abstractmethod
    def call(self, X: Tensor) -> Tensor:
        raise NotImplementedError
    
    @abstractmethod
    def fit(self, *args, **kwargs) -> None:
        raise NotImplementedError

    def __call__(self, X: Tensor) -> Tensor:
        return self.call(X)