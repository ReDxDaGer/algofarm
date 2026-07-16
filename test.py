import random
import math
import statistics
import algofarm

random.seed(42)
PASS, FAIL = 0, 0

def check(name, actual, expected):
    global PASS, FAIL
    ok = actual == expected
    if isinstance(actual, float) or isinstance(expected, float):
        ok = math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-9)
    status = "PASS" if ok else "FAIL"
    if ok: PASS += 1
    else: FAIL += 1
    print(f"[{status}] {name}")
    if not ok:
        print(f"    got:      {actual}")
        print(f"    expected: {expected}")

def rand_arr(n=200, lo=-10_000, hi=10_000):
    return [random.randint(lo, hi) for _ in range(n)]

def rand_arr_dupes(n=200, lo=-20, hi=20):
    return [random.randint(lo, hi) for _ in range(n)]

def rand_floats(n=200):
    return [random.uniform(-1000, 1000) for _ in range(n)]

def ref_levenshtein(a, b):
    dp = list(range(len(b) + 1))
    for i in range(1, len(a) + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, len(b) + 1):
            cur = dp[j]
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + cost)
            prev = cur
    return dp[len(b)]

def ref_is_prime(n):
    if n < 2: return False
    if n < 4: return True
    if n % 2 == 0: return False
    i = 3
    while i * i <= n:
        if n % i == 0: return False
        i += 2
    return True

def ref_sieve(limit):
    return [n for n in range(2, limit + 1) if ref_is_prime(n)]

def ref_fib(n):
    a, b = 0, 1
    for _ in range(n):
        a, b = b, a + b
    return a

def ref_matmul(a, b):
    rows, inner, cols = len(a), len(b), len(b[0])
    out = [[0.0] * cols for _ in range(rows)]
    for i in range(rows):
        for k in range(inner):
            for j in range(cols):
                out[i][j] += a[i][k] * b[k][j]
    return out

# ---- sorting ----
for _ in range(20):
    arr = rand_arr()
    check("quicksort", algofarm.quicksort(arr), sorted(arr))
    check("mergesort", algofarm.mergesort(arr), sorted(arr))
    check("parallel_sort", algofarm.parallel_sort(arr), sorted(arr))

# force the actual parallel path (threshold is 50_000)
big = rand_arr(60_000)
check("parallel_sort (large, threaded path)", algofarm.parallel_sort(big), sorted(big))

for _ in range(10):
    arr = rand_floats()
    check("quicksort_f64", algofarm.quicksort_f64(arr), sorted(arr))

# ---- searching ----
for _ in range(20):
    arr = sorted(rand_arr(200))
    target = random.choice(arr) if random.random() < 0.5 else random.randint(-20000, 20000)
    result = algofarm.binary_search(arr, target)
    if result is None:
        check("binary_search (absence)", target in arr, False)
    else:
        check("binary_search (value at index)", arr[result], target)

for _ in range(20):
    arr = rand_arr()
    k = random.randint(1, len(arr))
    check("kth_smallest", algofarm.kth_smallest(arr, k), sorted(arr)[k - 1])

for _ in range(20):
    arr = rand_arr()
    k = random.randint(1, len(arr))
    check("top_k", algofarm.top_k(arr, k), sorted(arr, reverse=True)[:k])

# ---- dedup ----
for _ in range(20):
    arr = rand_arr_dupes()
    check("dedup", algofarm.dedup(arr), list(dict.fromkeys(arr)))

# ---- levenshtein ----
words = ["kitten", "sitting", "flaw", "lawn", "intention", "execution", "", "a", "abc", "xyz"]
for _ in range(30):
    a, b = random.choice(words), random.choice(words)
    check(f"levenshtein({a!r}, {b!r})", algofarm.levenshtein(a, b), ref_levenshtein(a, b))

# ---- number theory ----
for _ in range(30):
    a, b = random.randint(-1000, 1000), random.randint(-1000, 1000)
    if a == 0 and b == 0: continue
    check("gcd", algofarm.gcd(a, b), math.gcd(a, b))

for _ in range(30):
    a, b = random.randint(1, 1000), random.randint(1, 1000)
    check("lcm", algofarm.lcm(a, b), math.lcm(a, b))

for _ in range(30):
    base, exp, mod = random.randint(0, 10_000), random.randint(0, 1000), random.randint(1, 10_000)
    check("mod_pow", algofarm.mod_pow(base, exp, mod), pow(base, exp, mod))

for _ in range(30):
    n = random.randint(0, 100_000)
    check(f"is_prime({n})", algofarm.is_prime(n), ref_is_prime(n))

for limit in [10, 100, 1000, random.randint(500, 5000)]:
    check(f"sieve({limit})", algofarm.sieve(limit), ref_sieve(limit))

for n in range(0, 50):
    check(f"fibonacci({n})", algofarm.fibonacci(n), ref_fib(n))

# ---- stats ----
for _ in range(20):
    arr = rand_floats()
    check("mean", algofarm.mean(arr), statistics.mean(arr))
    check("median", algofarm.median(arr), statistics.median(arr))
    check("variance", algofarm.variance(arr), statistics.pvariance(arr))
    check("std_dev", algofarm.std_dev(arr), statistics.pstdev(arr))

for _ in range(20):
    n = random.randint(1, 50)
    a = [random.uniform(-10, 10) for _ in range(n)]
    b = [random.uniform(-10, 10) for _ in range(n)]
    check("dot_product", algofarm.dot_product(a, b), sum(x * y for x, y in zip(a, b)))

for _ in range(10):
    r, k, c = random.randint(1, 5), random.randint(1, 5), random.randint(1, 5)
    a = [[random.uniform(-5, 5) for _ in range(k)] for _ in range(r)]
    b = [[random.uniform(-5, 5) for _ in range(c)] for _ in range(k)]
    got = algofarm.matmul(a, b)
    expected = ref_matmul(a, b)
    all_close = all(
        math.isclose(got[i][j], expected[i][j], abs_tol=1e-9)
        for i in range(r) for j in range(c)
    )
    check("matmul", all_close, True)

# ---- FarmHashMap vs dict ----
for trial in range(2):
    m = algofarm.FarmHashMap()
    ref = {}
    for _ in range(500):
        op = random.choice(["insert", "get", "remove", "contains"])
        key = random.randint(-50, 50)
        if op == "insert":
            val = random.randint(0, 1000)
            check(f"FarmHashMap.insert (trial {trial})", m.insert(key, val), ref.get(key))
            ref[key] = val
        elif op == "get":
            check(f"FarmHashMap.get (trial {trial})", m.get(key), ref.get(key))
        elif op == "remove":
            check(f"FarmHashMap.remove (trial {trial})", m.remove(key), ref.pop(key, None))
        elif op == "contains":
            check(f"FarmHashMap.contains (trial {trial})", m.contains(key), key in ref)
    check(f"FarmHashMap.__len__ (trial {trial})", len(m), len(ref))

print(f"\n{PASS} passed, {FAIL} failed")
