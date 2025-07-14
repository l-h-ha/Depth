from .. import Tensor

from ._base_optimizer import base_optimizer

class GradientDescent(base_optimizer):
    def __init__(self, learning_rate: float=0.01) -> None:
        super().__init__(learning_rate)
        self.learning_rate = learning_rate

    def call(self, parameters: list[Tensor]) -> None:
        for param in parameters:
            param.data -= param.grad * self.learning_rate