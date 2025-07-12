from abc import ABC, abstractmethod

from .. import Tensor

import numpy as np

from ..typedef import DTypeLike, float32
    
class base_initializer(ABC):
    def __init__(self, requires_grad: bool=True, dtype: DTypeLike=float32) -> None:
        self.requires_grad = requires_grad
        self.dtype = dtype

    def set_dtype(self, dtype: DTypeLike) -> None:
        self.dtype = dtype

    @abstractmethod
    def call(self, input_shape: tuple[int, ...]) -> np.ndarray:
        raise NotImplementedError

    def __call__(self, input_shape: tuple[int, ...]) -> Tensor:
        return Tensor(data=self.call(input_shape), requires_grad=self.requires_grad, dtype=self.dtype)