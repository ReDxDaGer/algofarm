# Algofarm

<div align="center">
  <img src="assets/logo.svg" alt="AlgoFarm Logo" width="500" height="400"/>
</div>


Fast algorithms and a from-scratch deep learning toolkit for Python, implemented in Rust.

Two layers:

- **Core algorithms** — sorting, searching, string distance, number theory, stats, a
  hand-rolled hashmap. Zero dependencies beyond `pyo3`, everything hand-rolled from
  scratch, nothing you can't read and understand in five minutes.
- **Tensor / deep learning layer** — a `Tensor` type, activations, losses, a fused
  softmax+cross-entropy kernel, in-place Adam/SGD optimizer steps, and classic ML
  routines (k-means, matrix inverse, pairwise distances). This layer deliberately *does*
  use a couple of purpose-built dependencies — `matrixmultiply` for GEMM and `rayon` for
  parallelism — because reinventing BLAS-quality matrix multiplication from scratch buys
  you nothing; see [Design philosophy](#design-philosophy) below.

## Why

Python's standard library and `numpy` already cover a lot of this, but:

- `sorted(x)[:k]` sorts the whole array to get the top `k` — `top_k` doesn't.
- Pure-Python Levenshtein distance is slow enough to matter on real string data.
- `list(dict.fromkeys(x))` for dedup pays per-element interpreter overhead at scale.
- Everyone has a copy-pasted GCD/sieve/Miller-Rabin snippet from an old competitive
  programming folder.
- Hand-writing an Adam update loop over millions of parameters in pure Python is one of
  the slowest parts of a from-scratch training loop — and softmax+cross-entropy computed
  as two separate NumPy passes is both slower and less numerically stable than doing it
  fused, in one pass, in Rust.

algofarm exists so you don't have to write (or copy-paste) any of this again.

## Install

```bash
pip install maturin numpy
maturin develop --release
```

This builds the Rust extension and installs it into your active Python environment.
Use `--release` — debug builds are dramatically slower, especially for the sort/search
and Tensor GEMM paths.

---

## Core algorithms

### Sorting

```python
import algofarm

algofarm.quicksort([5, 3, 8, 1])          # [1, 3, 5, 8]   — randomized pivot, 3-way partition
algofarm.quicksort_f64([3.1, 1.5, 2.2])   # float variant, NaN-safe ordering
algofarm.mergesort([5, 3, 8, 1])          # stable, O(n log n) guaranteed
algofarm.parallel_sort(big_list)          # multi-threaded via std::thread; only worth it above ~50k elements
```

`quicksort` uses a randomized pivot and Dutch-flag 3-way partitioning, so duplicate-heavy
data and adversarial inputs don't degrade to O(n²). Recursion always descends into the
smaller partition first, which bounds stack depth to O(log n) regardless of input shape.
Arrays of 16 or fewer elements fall back to insertion sort.

### Searching

```python
algofarm.binary_search([1, 3, 5, 7, 9], 5)   # 2 (index), or None if not found
algofarm.kth_smallest([9, 2, 7, 4, 1], 2)    # 2nd smallest, O(n) average — no full sort
algofarm.top_k([9, 2, 7, 4, 1], 3)           # [9, 7, 4], descending
```

`kth_smallest` and `top_k` use quickselect, so pulling the top 100 rows out of 50 million
doesn't pay for sorting the other 49,999,900.

### Deduplication

```python
algofarm.dedup([1, 2, 2, 3, 1, 4])   # [1, 2, 3, 4] — first-occurrence order preserved
```

### String distance

```python
algofarm.levenshtein("kitten", "sitting")   # 3
```

Rolling two-row DP — O(min(n, m)) space instead of the full O(n·m) matrix.

### Number theory

```python
algofarm.gcd(48, 18)              # 6
algofarm.lcm(4, 6)                # 12
algofarm.mod_pow(7, 128, 13)      # binary exponentiation, u128 intermediate — no overflow
algofarm.is_prime(982451653)      # deterministic Miller-Rabin, correct for the full u64 range
algofarm.sieve(100)               # all primes up to 100, as a list
algofarm.fibonacci(50)            # fast doubling, O(log n) — not the naive O(n) loop
```

### Statistics

```python
algofarm.mean([1.0, 2.0, 3.0])       # 2.0
algofarm.median([1.0, 2.0, 3.0])     # 2.0
algofarm.variance([1.0, 2.0, 3.0])   # population variance (divides by n, not n-1)
algofarm.std_dev([1.0, 2.0, 3.0])
```

### Linear algebra (scalar lists)

```python
algofarm.dot_product([1.0, 2.0], [3.0, 4.0])   # 11.0
algofarm.matmul(a, b)                          # a: m×k, b: k×n -> m×n; raises ValueError on mismatch
```

`matmul` here uses i-k-j loop ordering for cache locality — meaningfully faster than a
naive i-j-k triple loop on `Vec<Vec<f64>>`, but this is the plain-list version. For
anything performance-critical, use `Tensor.matmul` below instead, which is backed by a
real GEMM implementation.

### FarmHashMap

A hand-rolled open-addressing hashmap (linear probing, tombstone deletion, grows at 70%
load factor), usable as a Python object:

```python
from algofarm import FarmHashMap

m = FarmHashMap()
m.insert(1, 100)
m.get(1)          # 100
m.contains(1)     # True
m.remove(1)       # 100 (the removed value)
len(m)
m.keys()
m.values()
m.items()
```

Currently keyed on `int -> int`. Hashing is a splitmix64-style bit mixer, not
`std::collections::HashMap`'s default hasher.

---

## Tensor

```python
from algofarm import Tensor

a = Tensor([1.0, 2.0, 3.0, 4.0], [2, 2])
b = Tensor([5.0, 6.0, 7.0, 8.0], [2, 2])

a.add(b)             # element-wise, GIL released, size-gated parallelism
a.sub(b)
a.mul_elementwise(b)
a.mul_scalar(0.1)     # e.g. applying a learning rate
a.matmul(b)           # backed by matrixmultiply's dgemm, not a naive triple loop
a.reshape([4, 1])     # O(1) — shares the underlying buffer via Arc, doesn't copy
a.to_list()           # nested Python list, matching `shape`

Tensor.rand([3, 3], seed=42)    # uniform [0, 1), parallel generation when unseeded
Tensor.randn([3, 3], seed=42)   # standard normal
```

**Known limitations** (roadmap, not yet implemented):
- No broadcasting — `add`/`sub`/`mul_elementwise` require identical shapes. Bias-vector
  addition currently has to happen NumPy-side after converting out of `Tensor`.
- No `.transpose()` — backward-pass matmuls that need `X.T` currently fall back to NumPy.
- `Tensor.new()` takes a plain Python list, not a NumPy buffer, so constructing a Tensor
  from a NumPy array pays a real per-element conversion cost. Worth fixing by accepting
  `PyReadonlyArray1` the way the `dl_formulas` functions already do.
- `matmul` is rank-2 only; no batched matmul yet (needed for attention/transformer work).

## Deep learning primitives (`dl_formulas`)

All of these take/return NumPy arrays directly (`PyReadonlyArray`/`PyReadwriteArray`) —
zero-copy on the way in, no per-element unboxing the way a `Vec<f64>` parameter would
cost.

**Activations** (each with a `_derivative` counterpart where relevant):
```python
algofarm.relu(x)
algofarm.leaky_relu(x, alpha=0.01)
algofarm.elu(x, alpha=1.0)
algofarm.gelu(x)            # tanh approximation, as used in BERT/GPT-style transformers
algofarm.silu(x)            # Swish: x * sigmoid(x)
algofarm.sigmoid(x)
algofarm.tanh(x)
```

**Softmax + fused cross-entropy** — the standout piece. Computes the numerically stable
softmax, the loss, *and* the gradient w.r.t. the logits in a single pass:
```python
probs = algofarm.softmax(logits)                       # (batch, classes), row-wise, numerically stable
loss, grad = algofarm.softmax_cross_entropy(logits, targets)  # targets = class indices, like PyTorch
```
This mirrors what `torch.nn.functional.cross_entropy` does internally. Computing softmax
and cross-entropy as two separate NumPy steps is both slower (an extra full pass over the
data) and less numerically stable — you'd be taking `log()` of an already-exponentiated,
already-normalized value instead of using the algebraic simplification that avoids it.

```python
algofarm.one_hot(indices, num_classes)
```

**Losses**:
```python
algofarm.mse_loss(predictions.data, targets.data)
algofarm.mae_loss(predictions.data, targets.data)
algofarm.bce_loss(predictions.data, targets.data)
algofarm.categorical_cross_entropy(predictions, targets)
```

**Optimizers** — in-place parameter updates, GIL released across the whole step. This is
where Rust wins by the widest margin: a hand-written Adam loop over millions of
parameters in pure Python is glacial by comparison.
```python
algofarm.sgd_momentum_step(params, grads, velocity, lr, momentum)
algofarm.adam_step(params, grads, m, v, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8, t=1)
```
`params`, `velocity`/`m`/`v` are mutated in place — allocate them once (e.g.
`np.zeros_like(params)`) and keep passing the same arrays back in every step.

**Gradient utilities**:
```python
algofarm.clip_grad_norm(grads, max_norm)   # in-place, matches torch's clip_grad_norm_ semantics; returns the pre-clip norm
algofarm.dropout(x, p, seed=None)          # inverted dropout — no inference-time rescaling needed
```

## Classic ML (`ml`)

```python
algofarm.pairwise_manhattan(matrix_a, matrix_b)
algofarm.pairwise_cosine(matrix_a, matrix_b)
algofarm.determinant(matrix)
algofarm.matrix_inverse(matrix)             # Gauss-Jordan with partial pivoting
algofarm.kmeans(data, k, max_iters, tolerance)   # returns (centroids, assignments)
```

---

## Performance notes

- All functions release the GIL during computation (`py.allow_threads`), so other Python
  threads keep running while algofarm works — this matters for multi-threaded
  applications generally, not just `parallel_sort`.
- Rayon parallelism (used across `dl_formulas`, `ml`, and `Tensor`) is size-gated at
  10,000 elements — below that threshold, dispatch overhead to the thread pool costs more
  than the sequential path, so everything falls back automatically.
- `Tensor.matmul` is backed by `matrixmultiply::dgemm`, a solid pure-Rust GEMM — but it is
  **not** BLAS-tuned the way NumPy's matmul (OpenBLAS/MKL) is. For very large matmuls,
  NumPy may still win on that specific op; algofarm's advantage shows up more in fused
  kernels (`softmax_cross_entropy`) and in-place optimizer steps, where NumPy pays
  multiple full-array passes and temporary allocations that algofarm avoids.
- `is_prime` is deterministic (not probabilistic) for the entire `u64` range using a fixed
  12-witness Miller-Rabin set — no false positives, unlike a naive random-witness version.

### Benchmarking

`benchmark_train.py` trains an identical 2-layer MLP two ways — pure NumPy vs.
algofarm (`Tensor.matmul` for the GEMMs, `dl_formulas` for activations/loss,
`adam_step` for the optimizer) — from identical initial weights, and times both, with a
final-loss sanity check to confirm both actually converged to the same place rather than
just comparing speed of two different results. Run it yourself to reproduce numbers on
your own hardware:
```bash
python benchmark_train.py
```

## Testing

Correctness is checked against Python's standard library and reference implementations
across randomized inputs (fixed seed for reproducibility) — sorting against `sorted()`,
number theory against `math`/trial division, stats against the `statistics` module,
`FarmHashMap` against `dict` under randomized insert/get/remove/contains sequences, and
the training-loop primitives against a from-scratch NumPy MLP via `benchmark_train.py`.

## Design philosophy

The core algorithms module is zero-dependency beyond `pyo3` — no `rand`, no `rayon`, no
`std::collections::HashMap`. Randomization there uses a hand-rolled xorshift64* PRNG;
parallelism uses raw `std::thread`; the hashmap is open-addressing from scratch. The point
isn't that these beat a well-optimized crate on every benchmark — it's that nothing in
that layer is a black box you have to trust without being able to read it in five minutes.

The Tensor/deep-learning layer relaxes that constraint deliberately: `matrixmultiply` for
GEMM, `rayon` for parallelism, `rand`/`rand_distr` for tensor initialization, and `numpy`
for zero-copy interop are all genuinely load-bearing here, because hand-rolling a
competitive matrix multiply or a Mersenne-quality RNG from scratch would cost real
correctness and performance risk for no benefit — unlike a hashmap or a quicksort, GEMM
is not something worth reinventing badly just to avoid a dependency.

## License

MIT
