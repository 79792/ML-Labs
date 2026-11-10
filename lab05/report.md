# Lab 5: Frameworks and automatic differentiation

## 1. Goal

Build a small scalar autograd engine (like a mini PyTorch), check its gradients against PyTorch and numerical estimates, train a tiny network with it, and compare framework execution modes: eager vs traced/compiled, and contiguous vs non-contiguous tensors.

## 2. Method

**Value node** (`src/engine.py`). Each `Value` stores:
- `data`: the number,
- `grad`: dL/d(this node), starts at 0,
- `_parents`: the nodes it was computed from (graph edges),
- `_backward`: the local derivative rule, a small function that pushes `out.grad` to the parents with the chain rule.

**Ops.** `+`, `*`, `**k`, `relu` (required), plus `sigmoid` and `log` for training. Subtraction, negation and division are built from these. Each op creates a new `Value` with its inputs as parents, so the computation graph is built automatically during the forward pass.

| Op | Local derivative |
|---|---|
| a + b | da += out.grad, db += out.grad |
| a · b | da += b · out.grad, db += a · out.grad |
| a^k | da += k · a^(k−1) · out.grad |
| relu(a) | da += (a > 0) · out.grad |
| sigmoid(a) | da += s(1 − s) · out.grad |
| log(a) | da += (1/a) · out.grad |

**Backward pass.** Topological sort of the graph (each node after its parents, iterative DFS), reset grads, set output grad to 1, then call `_backward` on each node in reverse order. Gradients use `+=`, so a node used in several places gets the sum of all contributions.

**Gradient check.** f = ((a·b + c)²).relu().sum() for vectors of length 5 (seed 42). Compared with PyTorch autograd (float64) and a central difference (ε = 1e−6).

**make_moons.** 200 points, noise 0.1. Network 2 → 8 (ReLU) → 1 (sigmoid), BCE loss, full-batch gradient descent, lr 0.5, 100 epochs. Same initial weights in the engine and in PyTorch. Training time: median of 5 runs.

**PyTorch modes.** MLP 256 → 512 → 512 → 10, batch 64, eval mode, `no_grad`. Eager vs `torch.jit.trace` vs `torch.compile` (if supported), 20 warm-up calls then median of 200. Tensor shape, dtype, stride and contiguity printed. Non-contiguous test: `big.t()` on a 4096 × 4096 tensor vs its `.contiguous()` copy, for `+1` and row sum.

## 3. Results

### Gradient check (`results/grad_check.csv`)

| Comparison | Max absolute difference |
|---|---|
| engine vs PyTorch | 0 (exactly equal) |
| engine vs numerical | 6.7e−10 |

Both are below 1e−6: **PASS**. The forward value is also identical: 6.4580512298 in both.

### make_moons (`results/moons.csv`)

| | Final loss | Train accuracy | Time, 100 epochs (median of 5) |
|---|---|---|---|
| My engine | 0.2481 | 0.870 | 25.47 s |
| PyTorch | 0.2481 | 0.870 | 0.088 s |

Both reach the same loss and accuracy from the same starting weights, so the engine trains correctly. PyTorch is **289× faster**.

### Eager vs compiled (`results/timings.csv`)

| Mode | Median (ms) |
|---|---|
| eager | 0.640 |
| torch.jit.trace | 0.619 |
| torch.compile | 0.656 |

### Contiguity

| Operation | Contiguous (ms) | Non-contiguous view (ms) |
|---|---|---|
| x + 1 | 8.37 | 8.11 |
| row sum | 5.28 | **16.75** |

Cost of `.contiguous()` copy: 33.9 ms.

Tensor info: input (64, 256), float32, contiguous, stride (256, 1). `big.t()` has stride (1, 4096) and is not contiguous; after `.contiguous()` the stride is (4096, 1) again.

Graph printouts are in `results/graph.txt`.

## 4. Conclusions

1. My engine computes correct gradients. It matches PyTorch exactly (difference 0) and the numerical estimate up to 6.7e−10, far below the 1e−6 limit. The key details were sorting the graph topologically, so every node is processed only after everything that depends on it, and using `+=` for gradients, so a value that is used in several places gets the sum of all its contributions.

2. The engine trains the network correctly but is 289 times slower than PyTorch (25.5 s vs 0.09 s). Both ended with the same loss (0.2481) and accuracy (87%), so the math is the same. The difference is how the work is done: my engine makes a separate Python object and a Python function call for every single number, while PyTorch runs one optimized C++ operation on a whole matrix at once. This is the main reason frameworks work with tensors and not scalars.

3. Compilation did not help for this model: eager 0.64 ms, trace 0.62 ms, compile 0.66 ms. The model is only three big matrix multiplications with ReLU between them. Almost all the time goes into the matrix multiplications, which are already fast library calls, so there is little Python overhead for the compiler to remove.

4. Memory layout matters, but not for every operation. For `x + 1` the non-contiguous view was as fast as the normal tensor, because every element is processed on its own and PyTorch can walk the memory in its real order. For a row sum it was 3.2 times slower (16.7 ms vs 5.3 ms), because after `.t()` the numbers of one row are 4096 positions apart in memory, so the CPU cache cannot help. Making a contiguous copy costs 33.9 ms, which is more than the 11.5 ms saved on one sum, so the copy is only worth it if the tensor is used for several such operations.

### When compilation helps
Compilation helps when a model has many small operations in a row (for example elementwise ops, activations and normalizations), because it can fuse them into fewer kernels and remove the Python overhead between them. It also helps when the same model runs many times with the same input shape, so the one-time compile cost pays off. It does not help much when the time is spent in a few large matrix multiplications, like in my MLP, or when the model runs only a few times or with changing shapes, because then compiling costs more than it saves.
