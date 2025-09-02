from abc import ABC, abstractmethod
from typing import (Any)

class BaseModel(ABC):
    def __init__(self) -> None:
        super().__init__()
        self._build = False

    @abstractmethod
    def build(self, *args, **kwds: Any) -> Any:
        raise NotImplementedError

    @abstractmethod
    def call(self, *args: Any, **kwds: Any) -> Any:
        raise NotImplementedError

    @abstractmethod
    def fit(self, *args: Any, **kwds: Any) -> Any:
        raise NotImplementedError

    @abstractmethod
    def __call__(self, *args: Any, **kwds: Any) -> Any:
        raise NotImplementedError