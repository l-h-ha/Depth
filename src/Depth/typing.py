from typing import TYPE_CHECKING

import numpy as _np
import numpy.typing as _npt

type DTypeLike = _npt.DTypeLike
float32 = _np.float32
float64 = _np.float64

if TYPE_CHECKING:
    from .layers import _base_layer
    type LayerLike = _base_layer.base_layer

from .activations import _base_activation
type ActivationLike = _base_activation.base_activation

from .losses import _base_loss
type LossLike = _base_loss.base_loss