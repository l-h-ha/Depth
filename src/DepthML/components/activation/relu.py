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

def relu_op(x: NDArrayLike, *, device: DeviceLike, **kwds: Any) -> Tensor:
    return DTensor.maximum(0.0, x, requires_grad=True, device=device)

def relu_diff(result: Tensor, x: NDArrayLike, **kwds: Any) -> Callable[[], NDArrayLike]:
    def dx() -> NDArrayLike:
        return DTensor.where(x > 0.0, 1.0, 0.0).data
    return dx

relu_func = create_1in_1out(relu_op, relu_diff)

class ReLU(BaseActivation):
    def call(self, X: Tensor) -> Tensor:
        return relu_func(X)