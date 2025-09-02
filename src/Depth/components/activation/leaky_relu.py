from typing import (
    Callable,
    Any
)

import DepthTensor as DTensor
from DepthTensor import (
    Tensor, 
    NDArrayLike, 
    DeviceLike,
    create_1in_1out
)

from .base_activation import BaseActivation

def op(x: NDArrayLike, *, device: DeviceLike, alpha: float = 0.01, **kwds: Any) -> Tensor:
    return DTensor.maximum(alpha * x, x, requires_grad=True, device=device)

def diff(result: Tensor, x: NDArrayLike, alpha: float = 0.01, **kwds: Any) -> Callable[[], NDArrayLike]:
    def dx() -> NDArrayLike:
        return DTensor.where(x > 0.0, 1.0, alpha).data
    return dx

func = create_1in_1out(op, diff)

class LeakyReLU(BaseActivation):
    def __init__(self, alpha: float = 0.01) -> None:
        super().__init__()
        self.alpha = alpha

    def call(self, X: Tensor) -> Tensor:
        return func(X, alpha=self.alpha)