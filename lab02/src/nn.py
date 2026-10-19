"""Two-hidden-layer MLP in pure NumPy: 64 -> W -> ReLU -> W -> ReLU -> 10 -> softmax."""
import numpy as np


def init_params(width, n_in=64, n_out=10, std=0.01, seed=42):
    rng = np.random.default_rng(seed)
    return {
        "W1": rng.normal(0, std, (n_in, width)), "b1": np.zeros(width),
        "W2": rng.normal(0, std, (width, width)), "b2": np.zeros(width),
        "W3": rng.normal(0, std, (width, n_out)), "b3": np.zeros(n_out),
    }


def relu(z):
    return np.maximum(0, z)


def softmax(z):
    # subtract row max for numerical stability
    e = np.exp(z - z.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def forward(p, X):
    """Return probabilities and cache of intermediates for backprop."""
    z1 = X @ p["W1"] + p["b1"]
    a1 = relu(z1)
    z2 = a1 @ p["W2"] + p["b2"]
    a2 = relu(z2)
    z3 = a2 @ p["W3"] + p["b3"]
    probs = softmax(z3)
    cache = (X, z1, a1, z2, a2, probs)
    return probs, cache


def cross_entropy(probs, Y):
    # mean over batch; clip avoids log(0)
    return float(-np.mean(np.sum(Y * np.log(np.clip(probs, 1e-12, 1.0)), axis=1)))


def backward(p, cache, Y):
    """Gradients of mean cross-entropy w.r.t. all params."""
    X, z1, a1, z2, a2, probs = cache
    n = X.shape[0]
    dz3 = (probs - Y) / n            # softmax + CE combined
    dW3 = a2.T @ dz3
    db3 = dz3.sum(axis=0)
    da2 = dz3 @ p["W3"].T
    dz2 = da2 * (z2 > 0)             # ReLU derivative
    dW2 = a1.T @ dz2
    db2 = dz2.sum(axis=0)
    da1 = dz2 @ p["W2"].T
    dz1 = da1 * (z1 > 0)
    dW1 = X.T @ dz1
    db1 = dz1.sum(axis=0)
    return {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2, "W3": dW3, "b3": db3}


def sgd_step(p, grads, lr):
    for k in p:
        p[k] -= lr * grads[k]


def accuracy(p, X, Y):
    probs, _ = forward(p, X)
    return float(np.mean(probs.argmax(1) == Y.argmax(1)))
