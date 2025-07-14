from typing import Optional

from ._base_layer import base_layer
from .. import Tensor

from ..exceptions import ComponentNotBuiltError, ComputationError

class Input(base_layer):
    def __init__(self, shape: tuple[int, ...]) -> None:
        super().__init__()
        self.shape = shape
        self.batch_size: Optional[int] = None

    def build(self, in_shape: tuple[int, ...]) -> None:
        self.batch_size = in_shape[0]

    def call(self, X: Tensor) -> Tensor:
        if self.batch_size is None:
            raise ComponentNotBuiltError("Layer component has not been built as batch_size attribute is None.")
        if X.shape[1:] != self.shape or X.shape[0] != self.batch_size:
            raise ComputationError(f"Input data shape mismatch. Expected shape: {tuple([self.batch_size, *self.shape])}, got: {X.shape}")
        return X