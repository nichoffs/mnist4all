from __future__ import annotations
import numpy as np
from typing import Callable, List


class Function:
    def __init__(self, *tensors: Tensor):
        self.parents = tensors

    @classmethod
    def apply(fxn: Callable[Function], *tensors: Tensor):  # pyright: ignore
        ctx = fxn(*tensors)
        ret = Tensor(ctx.forward(*[t.buf for t in tensors]))
        ret._ctx = ctx
        return ret


class Tensor:
    def __init__(self, buf: np.ndarray):
        self.buf = buf
        self.grad = None
        self._ctx = None

    def __repr__(self):
        return f"Tensor with shape: {self.buf.shape}"

    # backprop
    def backward(self, implicit: bool = True):
        if self._ctx is None:
            return

        if implicit:
            assert self.buf.size == 1, "Can only backprop scalar"
            self.grad = np.array(1.0)

        assert self.grad is not None

        grads = self._ctx.backward(self.grad)

        for t, g in zip(self._ctx.parents, grads):
            assert (
                g.shape == t.buf.shape
            ), f"grad shape {g.shape} != tensor shape {t.buf.shape}"

            t.grad = g
            t.backward(False)

    # ops

    def __mul__(self, other: Tensor) -> Tensor:
        return Mul.apply(self, other)

    def relu(self):
        return ReLU.apply(self)

    def dot(self, other: Tensor):
        return Dot.apply(self, other)

    def sum(self):
        return Sum.apply(self)

    def logsoftmax(self):
        return LogSoftmax.apply(self)

    def mean(self):
        div = Tensor(np.array([1 / self.buf.size]))
        return self.sum() * div


class Mul(Function):
    def forward(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        self.x, self.y = x, y
        return x * y

    def backward(self, grad_output: np.ndarray):
        return [grad_output * self.y, grad_output * self.x]


class ReLU(Function):
    def forward(self, x):
        self.x = x
        return np.maximum(x, 0)

    def backward(self, grad_output):
        grad_input = grad_output.copy()
        grad_input[self.x < 0] = 0
        return [grad_input]


class Dot(Function):
    def forward(self, x: np.ndarray, y: np.ndarray):
        self.x, self.y = x, y
        return x.dot(y)

    def backward(self, grad_output: np.ndarray):
        grad_x = grad_output.dot(self.y.T)
        grad_y = grad_output.T.dot(self.x).T
        return [grad_x, grad_y]


class Sum(Function):
    def forward(self, x: np.ndarray):
        self.x = x
        return np.array([x.sum()])

    def backward(self, grad_output: np.ndarray):
        return [np.full_like(self.x, grad_output)]


class LogSoftmax(Function):
    def forward(self, x: np.ndarray):
        def logsumexp(x):
            c = x.max(axis=1)
            return c + np.log(np.exp(x - c.reshape((-1, 1))).sum(axis=1))

        self.output = x - logsumexp(x).reshape((-1, 1))
        return self.output

    def backward(self, grad_output):
        return [
            grad_output - np.exp(self.output) * grad_output.sum(axis=1).reshape((-1, 1))
        ]


class SGD:
    def __init__(self, params: List[Tensor], lr: float):
        self.params = params
        self.lr = lr

    def step(self):
        for p in self.params:
            p.buf -= p.grad * self.lr
            p.grad = None
