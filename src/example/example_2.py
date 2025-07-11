from ..Depth import Tensor
from ..Depth.activations._identity import Identity

iden = Identity()

w = Tensor.rand(2, 3, name='w', requires_grad=True)
a = Tensor.rand(3, 3, name='a', requires_grad=True)
b = Tensor.rand(2, 3, name='b', requires_grad=True)

z = w @ a + b
z.set_name('z')


a = iden(z)
a.set_name('a')
a.backward()

print(a.grad.shape)