from __future__ import annotations
from typing import Any, Sequence, List
from DepthTensor import Tensor, differentiate

from DepthTensor.typing import (
    DeviceLike
)

from ..typing import (
    ComponentLike,
    LossLike
)

from .base_model import BaseModel

class Stack(BaseModel):
    def __init__(self, components: Sequence[ComponentLike]) -> None:
        super().__init__()
        self.components = components

    def build(self) -> Any:
        pass

    def call(self, X: Tensor) -> Tensor:
        for component in self.components:
            X = component(X)
        return X
    
    def fit(self: Stack, X: List[Tensor], Y: List[Tensor], loss: LossLike, epoch: int = 1, learning_rate: float = 0.01):
        samples_n = len(X)
        
        for epoch_n in range(epoch):
            for (X_ind, X_val), (Y_ind, Y_val) in zip(enumerate(X), enumerate(Y)):
                y_tr = self(X_val)
                L = loss(Y_val, y_tr)
                parameters = differentiate(L)

                for t in parameters:
                    t.data -= learning_rate * t.grad

                progress = max(0, min(10, 10*(X_ind+1)//samples_n))
                print(
                    "[Epoch: {epoch_n}/{epoch_max}] Progress: [{progress_str}] ({progress_percent}%) Loss: {loss_scalar}".format(
                        epoch_n=epoch_n + 1,
                        epoch_max=epoch,
                        progress_str="#"*progress + "_"*(10-progress),
                        progress_percent=100*(X_ind+1)/samples_n,
                        loss_scalar=L.data.mean()
                    ),
                    end="\r" if X_ind + 1 < samples_n else "\n"
                )
    
    def __call__(self, X: Tensor) -> Tensor:
        if not self._build:
            self._build = True
            self.build()
        return self.call(X)