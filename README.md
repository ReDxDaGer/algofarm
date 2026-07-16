# algofarm

Fast, dependency-free algorithms for Python, implemented from scratch in Rust.

No `rand`, no `rayon`, no `std::collections::HashMap` under the hood — just `pyo3` and
hand-rolled implementations of the things people rewrite in every project: sorting,
searching, string distance, number theory, basic stats, and a custom hashmap you can
use directly from Python.

## Why

Python's standard library and `numpy` already cover a lot of this, but:

- `sorted(x)[:k]` sorts the whole array to get the top `k` — `top_k` doesn't.
- Pure-Python Levenshtein distance is slow enough to matter on real string data.
- `list(dict.fromkeys(x))` for dedup pays per-element interpreter overhead at scale.
- Everyone has a copy-pasted GCD/sieve/Miller-Rabin snippet from an old competitive
  programming folder.

algofarm exists so you don't have to write (or copy-paste) these again.

## Install

```bash
pip install maturin
maturin develop --release
```

This builds the Rust extension and installs it into your active Python environment.
Use `--release` — debug builds of the sort/search functions are dramatically slower.

## Sorting

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

## Searching

```python
algofarm.binary_search([1, 3, 5, 7, 9], 5)   # 2 (index), or None if not found
algofarm.kth_smallest([9, 2, 7, 4, 1], 2)    # 2nd smallest, O(n) average — no full sort
algofarm.top_k([9, 2, 7, 4, 1], 3)           # [9, 7, 4], descending
```

`kth_smallest` and `top_k` use quickselect, so pulling the top 100 rows out of 50 million
doesn't pay for sorting the other 49,999,900.

## Deduplication

```python
algofarm.dedup([1, 2, 2, 3, 1, 4])   # [1, 2, 3, 4] — first-occurrence order preserved
```

## String distance

```python
algofarm.levenshtein("kitten", "sitting")   # 3
```

Rolling two-row DP — O(min(n, m)) space instead of the full O(n·m) matrix.

## Number theory

```python
algofarm.gcd(48, 18)              # 6
algofarm.lcm(4, 6)                # 12
algofarm.mod_pow(7, 128, 13)      # binary exponentiation, u128 intermediate — no overflow
algofarm.is_prime(982451653)      # deterministic Miller-Rabin, correct for the full u64 range
algofarm.sieve(100)               # all primes up to 100, as a list
algofarm.fibonacci(50)            # fast doubling, O(log n) — not the naive O(n) loop
```

## Statistics

```python
algofarm.mean([1.0, 2.0, 3.0])       # 2.0
algofarm.median([1.0, 2.0, 3.0])     # 2.0
algofarm.variance([1.0, 2.0, 3.0])   # population variance (divides by n, not n-1)
algofarm.std_dev([1.0, 2.0, 3.0])
```

## Linear algebra

```python
algofarm.dot_product([1.0, 2.0], [3.0, 4.0])   # 11.0
algofarm.matmul(a, b)                          # a: m×k, b: k×n -> m×n; raises ValueError on mismatch
```

`matmul` uses i-k-j loop ordering for cache locality and skips multiplies against zero
entries — meaningfully faster than the naive i-j-k triple loop on anything but tiny matrices.

## FarmHashMap

A hand-rolled open-addressing hashmap (linear probing, tombstone deletion, grows at 70%
load factor), usable as a Python object:

```python
from algofarm import FarmHashMap

m = FarmHashMap()
m.insert(1, 100)
m.get(1)          # 100
m.contains(1)      # True
m.remove(1)        # 100 (the removed value)
len(m)
m.keys()
m.values()
m.items()
```

Currently keyed on `int -> int`. Hashing is a splitmix64-style bit mixer, not
`std::collections::HashMap`'s default hasher.

## Performance notes

- `parallel_sort` spawns real OS threads via `std::thread::scope` and only pays that
  overhead above ~50,000 elements — below that it silently falls back to the sequential
  `quicksort`.
- All functions release the GIL during the actual computation (`py.allow_threads`), so
  other Python threads keep running while algofarm works — this matters if you're calling
  these from a multi-threaded application, not just for the parallel_sort case.
- `is_prime` is deterministic (not probabilistic) for the entire `u64` range using a fixed
  12-witness Miller-Rabin set — no false positives, unlike a naive random-witness version.

## Testing

Correctness is checked against Python's standard library and reference implementations
across randomized inputs (fixed seed for reproducibility) — sorting against `sorted()`,
number theory against `math`/trial division, stats against the `statistics` module,
`FarmHashMap` against `dict` under randomized insert/get/remove/contains sequences.

## Design philosophy

Zero dependencies beyond `pyo3`. No `rand`, no `rayon`, no `std::collections::HashMap`.
Randomization uses a hand-rolled xorshift64* PRNG; parallelism uses raw `std::thread`;
the hashmap is open-addressing from scratch. The point isn't that these beat a
well-optimized crate on every benchmark — it's that nothing in this library is a black
box you have to trust without being able to read it in five minutes.

## License

MIT
