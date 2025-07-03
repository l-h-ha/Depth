from ..dlib.models import Stack
from ..dlib.layers import Input, AffineMap
from ..dlib.activations import ReLU
from ..dlib.losses import MSE

from ..dlib import Tensor

model = Stack([
    Input(shape=(10, 10)),
    AffineMap(units=10, activation=ReLU()),
])

y_true = Tensor.rand(10, 10, requires_grad=True)
inp = Tensor.rand(10, 10, requires_grad=True)

for i in range(1000000):
    y = model.forward(inp)
    L = model.backward(Y_true=y_true, Y_pred=y, loss=MSE(), learning_rate=0.1)
    print(L.data[0])