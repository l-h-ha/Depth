from DepthTensor import Tensor, random

from ...typing import (
    InitializerLike
)

from ...initializers import Uniform
from .base_layer import BaseLayer

class Affine(BaseLayer):
    def __init__(self, units: int, initializer: InitializerLike = Uniform()) -> None:
        super().__init__()
        self.units = units
        self.initializer = initializer

    def build(self, X: Tensor) -> None:
        self.w = self.initializer((X.shape[-1], self.units), X.device)
        #print(hash(self.w))
        self.b = random.uniform(-1, 1, (self.units,), device=X.device, requires_grad=True)
        #print(self.b.device)
        #print(type(self.b.grad))
        #print(hash(self.b))

    def call(self, X: Tensor) -> Tensor:
        return X @ self.w + self.b