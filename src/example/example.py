from ..dlib.models import Stack
from ..dlib.layers import Input, AffineMap
from ..dlib.activations import LeakyReLU, Softmax
from ..dlib.losses import FocalLoss, CategorialCrossEntropy, MeanSquaredError

from ..dlib import Tensor

model = Stack([
    Input(shape=(3, 3)),
    AffineMap(units=3, activation=LeakyReLU(alpha=0.1)),
    AffineMap(units=3, activation=Softmax()),
])

inp = Tensor.rand(3, 3, requires_grad=True)

import numpy as np
indices = np.random.randint(0, 3, size=3)
y_true = Tensor.ndarray(np.eye(3)[indices])

for i in range(2000):
    y = model.forward(inp)
    L = model.backward(Y_true=y_true, Y_pred=y, loss=FocalLoss(), learning_rate=1)
    print(L.mean().data[0])