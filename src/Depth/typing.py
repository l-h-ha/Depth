from typing import (
    TYPE_CHECKING,
    TypeAlias
)

if TYPE_CHECKING:
    from .models.base_model import BaseModel
    from .initializers.base_initializer import BaseInitializer
    from .components.base_component import BaseComponent
    from .components.activation.base_activation import BaseActivation
    from .components.layer.base_layer import BaseLayer
    from .components.loss.base_loss import BaseLoss

ModelLike: TypeAlias = "BaseModel"
InitializerLike: TypeAlias = "BaseInitializer"
ComponentLike: TypeAlias = "BaseComponent"
TransformerLike: TypeAlias = "BaseLayer"
ActivationLike: TypeAlias = "BaseActivation"
LossLike: TypeAlias = "BaseLoss"