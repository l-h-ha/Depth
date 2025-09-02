from typing import (
    Any,
    Callable
)

from DepthTensor import (
    Tensor,
    NDArrayLike,
    DeviceLike,
    create_1in_1out
)

from .base_activation import BaseActivation

def op(x: NDArrayLike, *, device: DeviceLike, alpha: float = 0.01, **kwds: Any) -> Tensor:
    return Tensor(x, device=device)

def diff(result: Tensor, x: NDArrayLike, alpha: float = 0.01, **kwds: Any) -> Callable[[], NDArrayLike]:
    def dx() -> NDArrayLike:
        return 1.0
    return dx

func = create_1in_1out(op, diff)

class Identity(BaseActivation):
    def call(self, X: Tensor) -> Tensor:
        return func(X)