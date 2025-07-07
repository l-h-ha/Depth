from ..dlib.models import Stack
from ..dlib.layers import Input, AffineMap
from ..dlib.activations import LeakyReLU, Softmax
from ..dlib.losses import CategorialCrossEntropy

from ..dlib import Tensor

model = Stack([
    Input(shape=(10, 10)),
    AffineMap(units=10, activation=LeakyReLU(alpha=0.1)),
    AffineMap(units=10, activation=Softmax()),
])

inp = Tensor.rand(10, 10, requires_grad=True)

import numpy as np
indices = np.random.randint(0, 10, size=10)
y_true = Tensor.ndarray(np.eye(10)[indices])

for i in range(1000000):
    y = model.forward(inp)
    L = model.backward(Y_true=y_true, Y_pred=y, loss=CategorialCrossEntropy(), learning_rate=.1)
    print(L.data[0])