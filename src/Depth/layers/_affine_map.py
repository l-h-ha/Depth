from ..initializers import Uniform
from ._base_layer import base_layer

from .. import Tensor

from ..typedef import ActivationLike, InitializerLike, DTypeLike, float32

class AffineMap(base_layer):
    def __init__(self, units: int, activation: 'ActivationLike', initializer: 'InitializerLike'=Uniform(), dtype: 'DTypeLike'=float32) -> None:
        super().__init__()
        self.units = units
        self.dtype = dtype
        self.activation = activation
        self.initializer = initializer
        initializer.set_dtype(dtype=dtype)

    def build(self, input_shape: tuple[int, ...]) -> None:
        self.w = self.initializer(input_shape=(input_shape[0], self.units))
        self.b = Tensor.zeros(shape=(self.units,), dtype=self.dtype, requires_grad=True)
        self.params.append(self.w)
        self.params.append(self.b)

    def call(self, X: Tensor) -> Tensor:
        return self.activation(X @ self.w + self.b)