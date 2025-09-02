from DepthTensor import Tensor, random
from DepthTensor.typing import (
    ShapeLike
)

from .base_initializer import BaseInitializer

class Uniform(BaseInitializer):
    def __init__(self, low: float = -0.05, high: float = 0.05) -> None:
        super().__init__()
        self.low = low
        self.high = high

    def call(self, fan_inout: ShapeLike) -> Tensor:
        return random.uniform(low=self.low, high=self.high, size=fan_inout, device=self.device, requires_grad=True)