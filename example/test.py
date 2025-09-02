import cProfile

code_1 = """
from src.Depth.models import Stack
from src.Depth.components import Affine
from src.Depth.components import LeakyReLU
from src.Depth.components import MeanSquaredError
from src.Depth.initializers import Xavier

import DepthTensor as DTensor
from DepthTensor import Tensor, random

a = Stack([
    Affine(units=5000, initializer=Xavier()),
    LeakyReLU()
])

x = [random.rand(1000, 5000, device="cpu", requires_grad=True)]*1
y_t = [random.rand(1000, 5000, device="cpu", requires_grad=True)]*1
loss = MeanSquaredError()

a.fit(
    x, y_t, loss, epoch=1, learning_rate=1e-6
)
"""

cProfile.run(code_1, filename='benchmark.prof')