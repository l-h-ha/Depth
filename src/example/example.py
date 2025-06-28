from ..lib._tensor import Tensor

a = Tensor([0, 0, 0])
b = Tensor([1, 1, 1])
print(a + b)
print(a * b)
print(a @ b)