from ._base_layer import base_layer
from ..initializers import Uniform

from .. import Tensor

from ..typedef import InitializerLike, DTypeLike, float32

class Affine(base_layer):
    def __init__(self, units: int, initializer: InitializerLike=Uniform(), dtype: DTypeLike=float32) -> None:
        super().__init__()
        self.units = units
        self.dtype = dtype
        self.initializer = initializer
        initializer.set_dtype(dtype=dtype)

    def build(self, input_shape: tuple[int, ...]) -> None:
        self.w = self.initializer(input_shape=(input_shape[0], self.units))
        self.b = Tensor.zeros(shape=(self.units,), requires_grad=True, dtype=self.dtype)
        self.params.append(self.w)
        self.params.append(self.b)

    def call(self, X: Tensor) -> Tensor:
        return X @ self.w + self.b