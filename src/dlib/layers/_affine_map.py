from ._base_layer import base_layer
from .. import Tensor

from .. import _typing as typ

class AffineMap(base_layer):
    def __init__(self, units: int, activation: typ.ActivationLike, dtype: typ.DTypeLike=typ.float32) -> None:
        super().__init__()
        self.units = units
        self.dtype = dtype
        self.activation = activation

    def build(self, input_shape: tuple) -> None:
        self.w = Tensor.rand(input_shape[1], self.units, dtype=self.dtype, requires_grad=True)
        self.b = Tensor.rand(input_shape[0], self.units, dtype=self.dtype, requires_grad=True)
        self.params.append(self.w)
        self.params.append(self.b)

    def call(self, X: Tensor) -> Tensor:
        return self.activation(X @ self.w + self.b)