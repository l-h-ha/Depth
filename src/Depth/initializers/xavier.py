from DepthTensor import Tensor, random
from DepthTensor.typing import (
    ShapeLike
)

from .base_initializer import BaseInitializer

import math

class Xavier(BaseInitializer):
    def __init__(self) -> None:
        super().__init__()

    def call(self, fan_inout: ShapeLike) -> Tensor:
        limit = math.sqrt(6 / sum(fan_inout))
        return random.uniform(low=-limit, high=limit, size=fan_inout, device=self.device, requires_grad=True)