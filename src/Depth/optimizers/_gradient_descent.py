from .. import Tensor

from ._base_optimizer import base_optimizer

class GradientDescent(base_optimizer):
    def call(self, parameters: list[Tensor], learning_rate: float) -> None:
        for param in parameters:
            param.data -= param.grad * learning_rate