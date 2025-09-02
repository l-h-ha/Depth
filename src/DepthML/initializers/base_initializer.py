from abc import ABC, abstractmethod
from typing import (
    Any,
    Optional
)

from DepthTensor.typing import (
    ShapeLike,
    DeviceLike
)

from DepthTensor import Tensor

class BaseInitializer(ABC):
    def __init__(self) -> None:
        super().__init__()
        self._build = False
        
    @abstractmethod
    def call(self, *args: Any, **kwds: Any) -> Any:
        raise NotImplementedError

    def build(self, device: DeviceLike) -> Any:
        self.device: DeviceLike = device
    
    def __call__(self, fan_inout: ShapeLike, device: Optional[DeviceLike] = None) -> Tensor:
        if not self._build:
            self._build = True
            if device is not None:
                self.build(device)
            else:
                raise RuntimeError("In order to be called, initializer must be built first. Insufficient arguments available to initiate the process: argument device is None.")
        return self.call(fan_inout)