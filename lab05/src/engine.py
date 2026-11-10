"""Scalar autograd engine. Each Value is a node in the computation graph."""
import math


class Value:
    def __init__(self, data, parents=(), op=""):
        self.data = float(data)
        self.grad = 0.0
        self._parents = parents          # nodes this one was computed from
        self._op = op                    # op name, for debugging/graph printing
        self._backward = lambda: None    # local derivative rule (chain rule step)

    def __repr__(self):
        return f"Value(data={self.data:.6g}, grad={self.grad:.6g})"

    @staticmethod
    def _wrap(x):
        return x if isinstance(x, Value) else Value(x)

    # ---------- core ops ----------
    def __add__(self, other):
        other = Value._wrap(other)
        out = Value(self.data + other.data, (self, other), "+")

        def _backward():
            # d(a+b)/da = 1, d(a+b)/db = 1; += accumulates
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward
        return out

    def __mul__(self, other):
        other = Value._wrap(other)
        out = Value(self.data * other.data, (self, other), "*")

        def _backward():
            # d(a*b)/da = b, d(a*b)/db = a
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out

    def __pow__(self, k):
        assert isinstance(k, (int, float)), "only constant powers"
        out = Value(self.data ** k, (self,), f"**{k}")

        def _backward():
            # d(a^k)/da = k * a^(k-1)
            self.grad += k * self.data ** (k - 1) * out.grad
        out._backward = _backward
        return out

    def relu(self):
        out = Value(self.data if self.data > 0 else 0.0, (self,), "relu")

        def _backward():
            self.grad += (1.0 if self.data > 0 else 0.0) * out.grad
        out._backward = _backward
        return out

    # ---------- extra ops for training ----------
    def sigmoid(self):
        s = 1.0 / (1.0 + math.exp(-self.data))
        out = Value(s, (self,), "sigmoid")

        def _backward():
            self.grad += s * (1 - s) * out.grad
        out._backward = _backward
        return out

    def log(self):
        out = Value(math.log(self.data), (self,), "log")

        def _backward():
            self.grad += (1.0 / self.data) * out.grad
        out._backward = _backward
        return out

    # ---------- derived ops (built from core ones) ----------
    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + (-Value._wrap(other))

    def __truediv__(self, other):
        return self * Value._wrap(other) ** -1

    def __radd__(self, other):
        return self + other

    def __rsub__(self, other):
        return Value._wrap(other) - self

    def __rmul__(self, other):
        return self * other

    # ---------- backward pass ----------
    def topo_order(self):
        """Nodes so that every node comes after all its parents."""
        order, seen = [], set()
        stack = [(self, False)]
        while stack:                      # iterative DFS, no recursion limit
            node, done = stack.pop()
            if done:
                order.append(node)
                continue
            if id(node) in seen:
                continue
            seen.add(id(node))
            stack.append((node, True))
            for p in node._parents:
                if id(p) not in seen:
                    stack.append((p, False))
        return order

    def backward(self):
        order = self.topo_order()
        for node in order:
            node.grad = 0.0               # fresh pass
        self.grad = 1.0                   # dL/dL = 1
        for node in reversed(order):      # output -> inputs
            node._backward()


def vsum(values):
    """Sum of a list of Values as one chain of additions."""
    total = Value(0.0)
    for v in values:
        total = total + v
    return total
