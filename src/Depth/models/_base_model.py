from abc import ABC, abstractmethod

class base_model(ABC):
    @abstractmethod
    def reset_grads(self) -> None:
        raise NotImplementedError