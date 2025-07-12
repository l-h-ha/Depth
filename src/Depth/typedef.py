from __future__ import annotations
from typing import TypeAlias, TYPE_CHECKING

if TYPE_CHECKING:
    from .activations._base_activation import base_activation
    from .losses._base_loss import base_loss
    from .initializers._base_initializer import base_initializer

ActivationLike: TypeAlias = 'base_activation'
LossLike: TypeAlias = 'base_loss'
InitializerLike: TypeAlias = 'base_initializer'

import numpy as np
DTypeLike: TypeAlias = np.typing.DTypeLike
float32: TypeAlias = np.float32