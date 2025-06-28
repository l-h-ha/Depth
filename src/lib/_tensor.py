from __future__ import annotations
from typing import Callable, Union

import numpy as np
import numpy.typing as npt

type tdata = Union[np.ndarray, np.floating, np.integer, list, tuple, int, float]

def convert(data: tdata, dtype: npt.DTypeLike) -> np.ndarray:
    if isinstance(data, np.ndarray) and data.dtype == dtype:
        return data
    if isinstance(data, (int, float, np.floating, np.integer)):
        return np.asarray([data]).astype(dtype=dtype)
    elif isinstance(data, (tuple, list)):
        return np.asarray(data).astype(dtype=dtype)
    elif isinstance(data, np.ndarray):
        return data.astype(dtype=dtype)
    else:
        raise ValueError(f'Unsupported datatype for conversion: {type(data)}')
    
def _perform_op(a: Tensor | tdata, b: Tensor | tdata, func: Callable[[np.ndarray, np.ndarray], np.ndarray], dtype: npt.DTypeLike) -> Tensor:
    a = a if isinstance(a, Tensor) else Tensor(data=convert(a, dtype=dtype), dtype=dtype)
    b = b if isinstance(b, Tensor) else Tensor(data=convert(b, dtype=dtype), dtype=dtype)
    return Tensor(data=func(a.data, b.data), dtype=dtype)

class Tensor():
    def __init__(self, data: tdata, dtype: npt.DTypeLike = np.float32):
        self.data: np.ndarray = convert(data=data, dtype=dtype)
        self.grad: np.ndarray = np.zeros_like(self.data, dtype=dtype)
        
        self.dtype = dtype

    def __add__(self, other: Tensor | tdata) -> Tensor:
        return _perform_op(self, other, func=lambda a, b: a + b, dtype=self.dtype)
    
    def __radd__(self, other: Tensor | tdata) -> Tensor:
        return _perform_op(other, self, func=lambda a, b: b + a, dtype=self.dtype)
    
    def __iadd__(self, other: Tensor | tdata) -> Tensor:
        self.data += other.data if isinstance(other, Tensor) else convert(other, self.dtype)
        return self
    

    def __sub__(self, other: Tensor | tdata) -> Tensor:
        return _perform_op(self, other, func=lambda a, b: a - b, dtype=self.dtype)
    
    def __rsub__(self, other: Tensor | tdata) -> Tensor:
        return _perform_op(other, self, func=lambda a, b: b - a, dtype=self.dtype)

    def __isub__(self, other: Tensor | tdata) -> Tensor:
        self.data -= other.data if isinstance(other, Tensor) else convert(other, self.dtype)
        return self
    

    def __mul__(self, other: Tensor | tdata) -> Tensor:
        return _perform_op(self, other, func=lambda a, b: a * b, dtype=self.dtype)
    
    def __rmul__(self, other: Tensor | tdata) -> Tensor:
        return _perform_op(other, self, func=lambda a, b: b * a, dtype=self.dtype)

    def __imul__(self, other: Tensor | tdata) -> Tensor:
        self.data *= other.data if isinstance(other, Tensor) else convert(other, self.dtype)
        return self
    

    def __truediv__(self, other: Tensor | tdata) -> Tensor:
        return _perform_op(self, other, func=lambda a, b: a / b, dtype=self.dtype)
    
    def __rtruediv__(self, other: Tensor | tdata) -> Tensor:
        return _perform_op(other, self, func=lambda a, b: b / a, dtype=self.dtype)
    
    def __itruediv__(self, other: Tensor | tdata) -> Tensor:
        self.data /= other.data if isinstance(other, Tensor) else convert(other, self.dtype)
        return self
    

    def __matmul__(self, other: Tensor | tdata) -> Tensor:
        return _perform_op(self, other, func=lambda a, b: a @ b, dtype=self.dtype)
    
    def __rmatmul__(self, other: Tensor | tdata) -> Tensor:
        return _perform_op(other, self, func=lambda a, b: b @ a, dtype=self.dtype)

    def __imatmul__(self, other: Tensor | tdata) -> Tensor:
        self.data @= other.data if isinstance(other, Tensor) else convert(other, self.dtype)
        return self
    

    def __neg__(self):
        return Tensor(data=-self.data, dtype=self.dtype)

    def __repr__(self) -> str:
        return f'Tensor({self.data})'