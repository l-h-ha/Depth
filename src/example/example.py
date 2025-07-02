from ..lib import Tensor

a = Tensor([1, 0, 0], requires_grad=True)
b = Tensor([1, 2, 3], requires_grad=True)
c = a @ b
d = c + a
e = d - c
f = e * c
g = f * a

g.backward()