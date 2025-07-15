from __future__ import annotations
from typing import Callable, Union, Optional

import numpy as np
import numpy.typing as npt

type tdata = Union[np.ndarray, np.floating, np.integer, list, tuple, int, float]

def convert(data: tdata, dtype: npt.DTypeLike) -> np.ndarray:
    if isinstance(data, np.ndarray):
        if data.dtype == dtype:
            return data
        return data.astype(dtype=dtype)
    elif isinstance(data, (int, float, np.floating, np.integer)):
        return np.asarray([data]).astype(dtype=dtype)
    elif isinstance(data, (tuple, list)):
        return np.asarray(data).astype(dtype=dtype)
    else:
        raise ValueError(f'Unsupported datatype for conversion: {type(data)}')

def _sum_to_shape(grad: np.ndarray, target_shape: tuple) -> np.ndarray:
    if grad.shape == target_shape:
        return grad
    
    ndim_diff = grad.ndim - len(target_shape)
    if ndim_diff < 0:
        raise ValueError("Gradient cannot have fewer dimensions than the target shape.")
    
    padded_target_shape = (1,)*ndim_diff + target_shape
    axes_to_sum = []
    for i, (grad_dim, target_dim) in enumerate(zip(grad.shape, padded_target_shape)):
        if grad_dim != target_dim:
            if target_dim == 1:
                axes_to_sum.append(i)
            else:
                raise ValueError(f'Shapes {grad.shape} and {target_shape} are not broadcast-compatible for summation.')
    
    if axes_to_sum:
        summed_grad = np.sum(grad, axis=tuple(axes_to_sum), keepdims=True)
    else:
        summed_grad = grad
    return summed_grad.reshape(target_shape)

def _backward_sum(a: Tensor, b: Tensor, result: Tensor) -> None:
    def _backward() -> None:
        a.grad += _sum_to_shape(result.grad * 1, a.shape)
        b.grad += _sum_to_shape(result.grad * 1, b.shape)
    result._backward = _backward

def _backward_sub(a: Tensor, b: Tensor, result: Tensor) -> None:
    def _backward() -> None:
        a.grad += _sum_to_shape(result.grad * 1, a.shape)
        b.grad += _sum_to_shape(result.grad * -1, b.shape)
    result._backward = _backward

def _backward_mul(a: Tensor, b: Tensor, result: Tensor) -> None:
    def _backward() -> None:
        a.grad += _sum_to_shape(result.grad * b.data, a.shape)
        b.grad += _sum_to_shape(result.grad * a.data, b.shape)
    result._backward = _backward

def _backward_div(a: Tensor, b: Tensor, result: Tensor) -> None:
    def _backward() -> None:
        a.grad += _sum_to_shape(result.grad * (1 / b.data), a.shape)
        b.grad += _sum_to_shape(result.grad * (a.data * -(1 / (b.data**2))), b.shape)
    result._backward = _backward

def _backward_pow(a: Tensor, b: Tensor, result: Tensor) -> None:
    def _backward() -> None:
        a.grad += _sum_to_shape(result.grad * b.data*a.data**(b.data-1), a.shape)
        b.grad += _sum_to_shape(result.grad * a.data**b.data*np.log(a.data), b.shape)
    result._backward = _backward

def _backward_matmul(a: Tensor, b: Tensor, result: Tensor) -> None:
    def _backward() -> None:
        #* Vec-vec product
        if a.ndim == 1 and b.ndim == 1:
            a.grad += result.grad * b.data
            b.grad += result.grad * a.data
        #* Mat-vec product
        elif a.ndim == 2 and b.ndim == 1:
            a.grad += np.outer(result.grad, b.data)
            b.grad += a.data.T @ result.grad
        #* Vec-mat product
        elif a.ndim == 1 and b.data.ndim == 2:
            a.grad += b.data.T @ result.grad
            b.grad += np.outer(result.grad, a.data)
        #* Mat-mat product or ten-ten product
        else:
            #! This is mathematically incorrect, is standard in ML implementations, since matrices are often batched.   
            grad_a = result.grad @ b.data.swapaxes(-2, -1)
            grad_b = a.data.swapaxes(-2, -1) @ result.grad
            a.grad += _sum_to_shape(grad_a, a.shape)
            b.grad += _sum_to_shape(grad_b, b.shape)
    result._backward = _backward

def _backward_neg(a: Tensor, result: Tensor) -> None:
    def _backward() -> None:
        a.grad += result.grad * -1
    result._backward = _backward

def _backward_transpose(a: Tensor, result: Tensor) -> None:
    def _backward() -> None:
        a.grad += result.grad.T
    result._backward = _backward

def _backward_mean(a: Tensor, result: Tensor, axis: Optional[Union[tuple, int]]=None, keepdims: bool=False) -> None:
    if axis is None:
        N = np.prod(a.shape)
    elif isinstance(axis, int):
        N = a.shape[axis]
    elif isinstance(axis, tuple):
        N = np.prod([a.shape[i] for i in axis]) 

    def _backward() -> None:
        grad = result.grad
        if not keepdims and axis is not None:
            grad = np.expand_dims(grad, axis=axis)
        a.grad += grad / N
    result._backward = _backward

def _perform_op(a: Tensor | tdata, b: Tensor | tdata, func: Callable[[np.ndarray, np.ndarray], np.ndarray], backward_func: Callable[[Tensor, Tensor, Tensor], None], dtype: npt.DTypeLike) -> Tensor:
    a = a if isinstance(a, Tensor) else Tensor(data=convert(a, dtype=dtype), requires_grad=False, dtype=dtype)
    b = b if isinstance(b, Tensor) else Tensor(data=convert(b, dtype=dtype), requires_grad=False, dtype=dtype)
    result = Tensor(data=func(a.data, b.data), prev=(a, b), requires_grad=a.requires_grad or b.requires_grad, dtype=dtype)

    if a.requires_grad or b.requires_grad:
        backward_func(a, b, result)
    return result

def _perform_in_op(a: Tensor | tdata, b: Tensor | tdata, func: Callable[[np.ndarray, np.ndarray], np.ndarray], dtype: npt.DTypeLike) -> Tensor:
    a = a if isinstance(a, Tensor) else Tensor(data=convert(a, dtype=dtype), dtype=dtype)
    b = b if isinstance(b, Tensor) else Tensor(data=convert(b, dtype=dtype), dtype=dtype)

    if a.requires_grad:
        raise RuntimeError("In-place operations must not be performed on differentiatable tensor objects.")

    a.data = func(a.data, b.data)
    return a

class Tensor():
    def __init__(self, data: tdata, prev: tuple=(), requires_grad: bool=False, dtype: npt.DTypeLike=np.float32, name: str=''):
        self.data = convert(data=data, dtype=dtype)
        self.grad = np.zeros_like(self.data, dtype=dtype)
        
        self.dtype = dtype
        self.prev = prev
        self._backward = lambda: None
        self.requires_grad = requires_grad

        self.shape = self.data.shape
        self.ndim = self.data.ndim
        self.size = self.data.size
        self.name = name

    @classmethod
    def rand(cls, *shape, name: str='', requires_grad: bool=False, dtype: npt.DTypeLike=np.float32):
        return cls(data=np.random.rand(*shape), requires_grad=requires_grad, dtype=dtype, name=name)
    
    @classmethod
    def zeros(cls, shape: tuple, name: str='', requires_grad: bool=False,dtype: npt.DTypeLike=np.float32):
        return cls(data=np.zeros(shape), requires_grad=requires_grad, dtype=dtype, name=name)
    
    @classmethod
    def ones(cls, shape: tuple, name: str='', requires_grad: bool=False,dtype: npt.DTypeLike=np.float32):
        return cls(data=np.ones(shape), requires_grad=requires_grad, dtype=dtype, name=name)
    
    @classmethod
    def ndarray(cls, ndarray: np.ndarray, name: str='', requires_grad: bool=False, dtype: npt.DTypeLike=np.float32):
        return cls(data=ndarray, requires_grad=requires_grad, dtype=dtype, name=name)

    ##
    ##
    ##

    def set_name(self, name: str) -> None:
        self.name = name

    def set_dtype(self, dtype: npt.DTypeLike) -> None:
        self.dtype = dtype
        self.data = self.data.astype(dtype=dtype, copy=False)

    def astype(self, dtype: npt.DTypeLike) -> Tensor:
        return Tensor(data=self.data, requires_grad=self.requires_grad, dtype=dtype)

    def get_antecedents(self):
        antecedents: list[Tensor] = []
        visit: set[Tensor] = set()

        def build(t: Tensor) -> None:
            if t in visit:
                return
            visit.add(t)
            for t_prev in t.prev:
                build(t_prev)
            antecedents.append(t)

        build(self)
        antecedents.reverse()
        return antecedents

    def backward(self, gradient: Optional[np.ndarray]=None) -> list[Tensor]:
        antecedents = self.get_antecedents()
        
        if gradient:
            self.grad = gradient
        else:
            self.grad = np.ones_like(self.data, dtype=self.dtype)

        for t in antecedents:
            t._backward()
        return antecedents
    
    def reset_grad(self, grad: Optional[np.ndarray]=None):
        if grad:
            self.grad = grad
        self.grad = np.zeros(shape=self.shape, dtype=self.dtype)

    ##
    ##
    ##

    def transpose(self, requires_grad: Optional[bool]=None) -> Tensor:
        if requires_grad is not None:
            requires_grad = self.requires_grad and requires_grad
        else:
            requires_grad = self.requires_grad

        result = Tensor(self.data.T, prev=(self,), requires_grad=requires_grad, dtype=self.dtype)
        if result.requires_grad:
            _backward_transpose(self, result)
        return result
    
    def mean(self, requires_grad: Optional[bool]=None, axis: Optional[Union[tuple, int]]=None, keepdims: bool=False) -> Tensor:
        if requires_grad is not None:
            requires_grad = self.requires_grad and requires_grad
        else:
            requires_grad = self.requires_grad

        result = Tensor(self.data.mean(axis=axis, dtype=self.dtype, keepdims=keepdims), prev=(self,), requires_grad=requires_grad, dtype=self.dtype)
        if requires_grad:
            _backward_mean(self, result, axis, keepdims)
        return result


    ##
    ##
    ##

    def __getitem__(self, key):
        return self.data[key]
    
    def __setitem__(self, key, val):
        self.data[key] = val

    ##
    ##
    ##

    def __add__(self, other: Tensor | tdata) -> Tensor:
        return _perform_op(self, other, func=lambda a, b: a + b, backward_func=_backward_sum, dtype=self.dtype)
    def __radd__(self, other: Tensor | tdata) -> Tensor:
        return _perform_op(other, self, func=lambda a, b: b + a, backward_func=_backward_sum, dtype=self.dtype)
    def __iadd__(self, other: Tensor | tdata) -> Tensor:
        return _perform_in_op(self, other, func=lambda a, b: a + b, dtype=self.dtype)


    def __sub__(self, other: Tensor | tdata) -> Tensor:
        return _perform_op(self, other, func=lambda a, b: a - b, backward_func=_backward_sub, dtype=self.dtype)
    def __rsub__(self, other: Tensor | tdata) -> Tensor:
        return _perform_op(other, self, func=lambda a, b: b - a, backward_func=_backward_sub, dtype=self.dtype)
    def __isub__(self, other: Tensor | tdata) -> Tensor:
        return _perform_in_op(self, other, func=lambda a, b: a - b, dtype=self.dtype)
    

    def __mul__(self, other: Tensor | tdata) -> Tensor:
        return _perform_op(self, other, func=lambda a, b: a * b, backward_func=_backward_mul, dtype=self.dtype)
    def __rmul__(self, other: Tensor | tdata) -> Tensor:
        return _perform_op(other, self, func=lambda a, b: b * a, backward_func=_backward_mul, dtype=self.dtype)
    def __imul__(self, other: Tensor | tdata) -> Tensor:
        return _perform_in_op(self, other, func=lambda a, b: a * b, dtype=self.dtype)
    

    def __truediv__(self, other: Tensor | tdata) -> Tensor:
        return _perform_op(self, other, func=lambda a, b: a / b, backward_func=_backward_div, dtype=self.dtype)
    def __rtruediv__(self, other: Tensor | tdata) -> Tensor:
        return _perform_op(other, self, func=lambda a, b: b / a, backward_func=_backward_div, dtype=self.dtype)
    def __itruediv__(self, other: Tensor | tdata) -> Tensor:
        return _perform_in_op(self, other, func=lambda a, b: a / b, dtype=self.dtype)
    

    def __pow__(self, other: Tensor | tdata) -> Tensor:
        return _perform_op(self, other, func=lambda a, b: a**b, backward_func=_backward_pow, dtype=self.dtype)
    def __rpow__(self, other: Tensor | tdata) -> Tensor:
        return _perform_op(other, self, func=lambda a, b: b**a, backward_func=_backward_pow, dtype=self.dtype)
    def __ipow__(self, other: Tensor | tdata) -> Tensor:
        return _perform_in_op(self, other, func=lambda a, b: a**b, dtype=self.dtype)


    def __matmul__(self, other: Tensor | tdata) -> Tensor:
        return _perform_op(self, other, func=lambda a, b: a @ b, backward_func=_backward_matmul, dtype=self.dtype)
    def __rmatmul__(self, other: Tensor | tdata) -> Tensor:
        return _perform_op(other, self, func=lambda a, b: b @ a, backward_func=_backward_matmul, dtype=self.dtype)
    def __imatmul__(self, other: Tensor | tdata) -> Tensor:
        return _perform_in_op(self, other, func=lambda a, b: a @ b, dtype=self.dtype)


    def __neg__(self):
        result = Tensor(data=-self.data, prev=(self,), requires_grad=self.requires_grad, dtype=self.dtype)
        if result.requires_grad:
            _backward_neg(self, result)
        return result

    def __repr__(self) -> str:
        return f'{self.name}_Tensor({self.data})'