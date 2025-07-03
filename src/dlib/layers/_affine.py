from ._base_layer import base_layer
from .. import Tensor

from .. import _typing as typ

class Affine(base_layer):
    def __init__(self, units: int, dtype: typ.DTypeLike=typ.float32) -> None:
        super().__init__()
        self.units = units
        self.dtype = dtype

    def build(self, input_shape: tuple) -> None:
        self.w = Tensor.rand(input_shape[1], self.units, requires_grad=True, dtype=self.dtype)
        self.b = Tensor.rand(input_shape[0], self.units, requires_grad=True, dtype=self.dtype)
        self.params.append(self.w)
        self.params.append(self.b)

    def call(self, X: Tensor) -> Tensor:
        return X @ self.w + self.b