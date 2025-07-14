from __future__ import annotations
from typing import TypeAlias, TYPE_CHECKING

if TYPE_CHECKING:
    from .activations._base_activation import base_activation
    from .losses._base_loss import base_loss
    from .initializers._base_initializer import base_initializer
    from .optimizers._base_optimizer import base_optimizer
    from .layers._base_layer import base_layer

LayerLike: TypeAlias = 'base_layer'
ActivationLike: TypeAlias = 'base_activation'
LossLike: TypeAlias = 'base_loss'
InitializerLike: TypeAlias = 'base_initializer'
OptimizerLike: TypeAlias = 'base_optimizer'

import numpy as np
DTypeLike: TypeAlias = np.typing.DTypeLike

int8: TypeAlias = np.int8
int16: TypeAlias = np.int16
int32: TypeAlias = np.int32
int64: TypeAlias = np.int64

float16: TypeAlias = np.float16
float32: TypeAlias = np.float32
float64: TypeAlias = np.float64