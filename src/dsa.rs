use pyo3::prelude::*;
use std::cell::RefCell;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

// ---------- PRNG (xorshift64*, no external crate) ----------
static SEED_COUNTER: AtomicU64 = AtomicU64::new(1);

fn next_seed() -> u64 {
    let t = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .subsec_nanos() as u64;
    let c = SEED_COUNTER.fetch_add(1, Ordering::Relaxed);
    let s = t ^ c.wrapping_mul(0x9E3779B97F4A7C15);
    if s == 0 { 0xDEADBEEF } else { s }
}

struct Rng {
    state: u64,
}
impl Rng {
    fn new() -> Self {
        Rng { state: next_seed() }
    }
    fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x.wrapping_mul(0x2545F4914F6CDD1D)
    }
    fn gen_range(&mut self, bound: usize) -> usize {
        (self.next_u64() % bound as u64) as usize
    }
}

thread_local! { static RNG: RefCell<Rng> = RefCell::new(Rng::new()); }
fn rand_index(bound: usize) -> usize {
    RNG.with(|r| r.borrow_mut().gen_range(bound))
}

// ---------- Sorting ----------
fn insertion_sort(arr: &mut [i64]) {
    for i in 1..arr.len() {
        let mut j = i;
        while j > 0 && arr[j - 1] > arr[j] {
            arr.swap(j - 1, j);
            j -= 1;
        }
    }
}

fn three_way_partition(arr: &mut [i64]) -> (usize, usize) {
    let pivot = arr[rand_index(arr.len())];
    let (mut lt, mut i, mut gt) = (0, 0, arr.len());
    while i < gt {
        match arr[i].cmp(&pivot) {
            std::cmp::Ordering::Less => {
                arr.swap(lt, i);
                lt += 1;
                i += 1;
            }
            std::cmp::Ordering::Greater => {
                gt -= 1;
                arr.swap(i, gt);
            }
            std::cmp::Ordering::Equal => {
                i += 1;
            }
        }
    }
    (lt, gt)
}

fn quicksort_worker(mut arr: &mut [i64]) {
    loop {
        if arr.len() <= 16 {
            insertion_sort(arr);
            return;
        }
        let (lt, gt) = three_way_partition(arr);
        let (left, right) = arr.split_at_mut(gt);
        let (left, _eq) = left.split_at_mut(lt);
        if left.len() < right.len() {
            quicksort_worker(left);
            arr = right;
        } else {
            quicksort_worker(right);
            arr = left;
        }
    }
}

#[pyfunction]
pub fn quicksort(py: Python<'_>, mut arr: Vec<i64>) -> Vec<i64> {
    py.allow_threads(|| {
        quicksort_worker(&mut arr);
        arr
    })
}

#[pyfunction]
pub fn quicksort_f64(py: Python<'_>, mut arr: Vec<f64>) -> Vec<f64> {
    py.allow_threads(|| {
        arr.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        arr
    })
}

fn mergesort_worker(arr: &mut [i64], scratch: &mut [i64]) {
    let size = arr.len();
    if size <= 1 {
        return;
    }
    let mid = size / 2;
    let (left, right) = arr.split_at_mut(mid);
    let (s_left, s_right) = scratch.split_at_mut(mid);
    mergesort_worker(left, s_left);
    mergesort_worker(right, s_right);
    let (mut i, mut j, mut k) = (0, mid, 0);
    while i < mid && j < size {
        if arr[i] <= arr[j] {
            scratch[k] = arr[i];
            i += 1;
        } else {
            scratch[k] = arr[j];
            j += 1;
        }
        k += 1;
    }
    if i < mid {
        scratch[k..size].copy_from_slice(&arr[i..mid]);
    } else if j < size {
        scratch[k..size].copy_from_slice(&arr[j..size]);
    }
    arr.copy_from_slice(scratch);
}

#[pyfunction]
pub fn mergesort(py: Python<'_>, mut arr: Vec<i64>) -> Vec<i64> {
    py.allow_threads(|| {
        if arr.len() <= 1 {
            return arr;
        }
        let mut scratch = vec![0; arr.len()];
        mergesort_worker(&mut arr, &mut scratch);
        arr
    })
}

fn merge_two(a: &[i64], b: &[i64], out: &mut Vec<i64>) {
    let (mut i, mut j) = (0, 0);
    while i < a.len() && j < b.len() {
        if a[i] <= b[j] {
            out.push(a[i]);
            i += 1;
        } else {
            out.push(b[j]);
            j += 1;
        }
    }
    out.extend_from_slice(&a[i..]);
    out.extend_from_slice(&b[j..]);
}

#[pyfunction]
pub fn parallel_sort(py: Python<'_>, mut arr: Vec<i64>) -> Vec<i64> {
    py.allow_threads(|| {
        let n = arr.len();
        if n < 50_000 {
            quicksort_worker(&mut arr);
            return arr;
        }
        let threads = std::thread::available_parallelism()
            .map(|t| t.get())
            .unwrap_or(4)
            .max(1);
        let chunk_size = (n + threads - 1) / threads;
        std::thread::scope(|scope| {
            for chunk in arr.chunks_mut(chunk_size) {
                scope.spawn(move || quicksort_worker(chunk));
            }
        });
        let mut chunks: Vec<Vec<i64>> = arr.chunks(chunk_size).map(|c| c.to_vec()).collect();
        while chunks.len() > 1 {
            let mut next = Vec::with_capacity((chunks.len() + 1) / 2);
            let mut it = chunks.into_iter();
            loop {
                match (it.next(), it.next()) {
                    (Some(a), Some(b)) => {
                        let mut out = Vec::with_capacity(a.len() + b.len());
                        merge_two(&a, &b, &mut out);
                        next.push(out);
                    }
                    (Some(a), None) => next.push(a),
                    _ => break,
                }
            }
            chunks = next;
        }
        chunks.into_iter().next().unwrap_or_default()
    })
}

// ---------- Searching ----------
#[pyfunction]
pub fn binary_search(arr: Vec<i64>, target: i64) -> Option<usize> {
    if arr.is_empty() {
        return None;
    }
    let (mut low, mut high) = (0, arr.len() - 1);
    while low <= high {
        let mid = low + (high - low) / 2;
        match arr[mid].cmp(&target) {
            std::cmp::Ordering::Equal => return Some(mid),
            std::cmp::Ordering::Less => low = mid + 1,
            std::cmp::Ordering::Greater => {
                if mid == 0 {
                    break;
                }
                high = mid - 1;
            }
        }
    }
    None
}

fn quickselect(mut arr: &mut [i64], mut k: usize) -> i64 {
    loop {
        if arr.len() == 1 {
            return arr[0];
        }
        let (lt, gt) = three_way_partition(arr);
        if k < lt {
            arr = &mut arr[..lt];
        } else if k < gt {
            return arr[k];
        } else {
            k -= gt;
            arr = &mut arr[gt..];
        }
    }
}

fn quickselect_partition_to(mut arr: &mut [i64], mut cut: usize) {
    if cut == 0 || cut >= arr.len() {
        return;
    }
    loop {
        if arr.len() <= 1 {
            return;
        }
        let (lt, gt) = three_way_partition(arr);
        if cut <= lt {
            arr = &mut arr[..lt];
        } else if cut <= gt {
            return;
        } else {
            cut -= gt;
            arr = &mut arr[gt..];
        }
    }
}

#[pyfunction]
pub fn kth_smallest(py: Python<'_>, mut arr: Vec<i64>, k: usize) -> PyResult<i64> {
    if k == 0 || k > arr.len() {
        return Err(pyo3::exceptions::PyIndexError::new_err("k out of range"));
    }
    Ok(py.allow_threads(|| quickselect(&mut arr, k - 1)))
}

#[pyfunction]
pub fn top_k(py: Python<'_>, mut arr: Vec<i64>, k: usize) -> Vec<i64> {
    py.allow_threads(|| {
        let k = k.min(arr.len());
        if k == 0 {
            return Vec::new();
        }
        let cut = arr.len() - k;
        quickselect_partition_to(&mut arr, cut);
        let mut result = arr[cut..].to_vec();
        result.sort_unstable_by(|a, b| b.cmp(a));
        result
    })
}

// ---------- Hashing primitive (shared by dedup + FarmHashMap) ----------
fn hash_i64(k: i64) -> u64 {
    let mut x = k as u64;
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D049BB133111EB);
    x ^ (x >> 31)
}

#[derive(Clone, Copy)]
enum SetSlot {
    Empty,
    Occupied(i64),
}

#[pyfunction]
pub fn dedup(py: Python<'_>, arr: Vec<i64>) -> Vec<i64> {
    py.allow_threads(|| {
        let mut cap = 16usize;
        while cap < arr.len() * 2 {
            cap *= 2;
        }
        let mut slots: Vec<SetSlot> = vec![SetSlot::Empty; cap];
        let mut out = Vec::with_capacity(arr.len());
        for x in arr {
            let mut idx = (hash_i64(x) as usize) & (cap - 1);
            loop {
                match slots[idx] {
                    SetSlot::Empty => {
                        slots[idx] = SetSlot::Occupied(x);
                        out.push(x);
                        break;
                    }
                    SetSlot::Occupied(v) if v == x => break,
                    _ => idx = (idx + 1) & (cap - 1),
                }
            }
        }
        out
    })
}

// ---------- Levenshtein ----------
#[pyfunction]
pub fn levenshtein(py: Python<'_>, a: &str, b: &str) -> usize {
    py.allow_threads(|| {
        let (a, b) = (a.as_bytes(), b.as_bytes());
        let (a, b) = if a.len() < b.len() { (b, a) } else { (a, b) };
        let mut prev: Vec<usize> = (0..=b.len()).collect();
        let mut curr = vec![0usize; b.len() + 1];
        for i in 1..=a.len() {
            curr[0] = i;
            for j in 1..=b.len() {
                let cost = if a[i - 1] == b[j - 1] { 0 } else { 1 };
                curr[j] = (prev[j] + 1).min(curr[j - 1] + 1).min(prev[j - 1] + cost);
            }
            std::mem::swap(&mut prev, &mut curr);
        }
        prev[b.len()]
    })
}

// ---------- Calculations ----------
#[pyfunction]
pub fn gcd(a: i64, b: i64) -> i64 {
    let (mut a, mut b) = (a.abs(), b.abs());
    while b != 0 {
        let t = b;
        b = a % b;
        a = t;
    }
    a
}

#[pyfunction]
pub fn lcm(a: i64, b: i64) -> i64 {
    if a == 0 || b == 0 {
        return 0;
    }
    (a / gcd(a, b) * b).abs()
}

#[pyfunction]
pub fn mod_pow(mut base: u64, mut exp: u64, modulus: u64) -> u64 {
    if modulus == 1 {
        return 0;
    }
    let mut result = 1u64;
    base %= modulus;
    while exp > 0 {
        if exp & 1 == 1 {
            result = ((result as u128 * base as u128) % modulus as u128) as u64;
        }
        exp >>= 1;
        base = ((base as u128 * base as u128) % modulus as u128) as u64;
    }
    result
}

#[pyfunction]
pub fn is_prime(n: u64) -> bool {
    if n < 2 {
        return false;
    }
    for p in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37] {
        if n == p {
            return true;
        }
        if n % p == 0 {
            return false;
        }
    }
    let (mut d, mut r) = (n - 1, 0);
    while d % 2 == 0 {
        d /= 2;
        r += 1;
    }
    'witness: for a in [2u64, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37] {
        if a >= n {
            continue;
        }
        let mut x = mod_pow(a, d, n);
        if x == 1 || x == n - 1 {
            continue;
        }
        for _ in 0..r - 1 {
            x = ((x as u128 * x as u128) % n as u128) as u64;
            if x == n - 1 {
                continue 'witness;
            }
        }
        return false;
    }
    true
}

#[pyfunction]
pub fn sieve(py: Python<'_>, limit: u64) -> Vec<u64> {
    py.allow_threads(|| {
        if limit < 2 {
            return Vec::new();
        }
        let n = limit as usize;
        let mut is_composite = vec![false; n + 1];
        let mut primes = Vec::new();
        for i in 2..=n {
            if !is_composite[i] {
                primes.push(i as u64);
                let mut j = i * i;
                while j <= n {
                    is_composite[j] = true;
                    j += i;
                }
            }
        }
        primes
    })
}

#[pyfunction]
pub fn fibonacci(n: u64) -> u64 {
    fn fib_pair(n: u64) -> (u64, u64) {
        if n == 0 {
            return (0, 1);
        }
        let (a, b) = fib_pair(n / 2);
        let c = a.wrapping_mul(b.wrapping_mul(2).wrapping_sub(a));
        let d = a.wrapping_mul(a).wrapping_add(b.wrapping_mul(b));
        if n % 2 == 0 {
            (c, d)
        } else {
            (d, c.wrapping_add(d))
        }
    }
    fib_pair(n).0
}

#[pyfunction]
pub fn mean(arr: Vec<f64>) -> f64 {
    if arr.is_empty() {
        return 0.0;
    }
    arr.iter().sum::<f64>() / arr.len() as f64
}

#[pyfunction]
pub fn median(mut arr: Vec<f64>) -> f64 {
    if arr.is_empty() {
        return 0.0;
    }
    arr.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
    let n = arr.len();
    if n % 2 == 1 {
        arr[n / 2]
    } else {
        (arr[n / 2 - 1] + arr[n / 2]) / 2.0
    }
}

#[pyfunction]
pub fn variance(arr: Vec<f64>) -> f64 {
    if arr.is_empty() {
        return 0.0;
    }
    let m = mean(arr.clone());
    arr.iter().map(|x| (x - m).powi(2)).sum::<f64>() / arr.len() as f64
}

#[pyfunction]
pub fn std_dev(arr: Vec<f64>) -> f64 {
    variance(arr).sqrt()
}

#[pyfunction]
pub fn dot_product(a: Vec<f64>, b: Vec<f64>) -> PyResult<f64> {
    if a.len() != b.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "vectors must be same length",
        ));
    }
    Ok(a.iter().zip(b.iter()).map(|(x, y)| x * y).sum())
}

#[pyfunction]
pub fn matmul(a: Vec<Vec<f64>>, b: Vec<Vec<f64>>) -> PyResult<Vec<Vec<f64>>> {
    if a.is_empty() || b.is_empty() || a[0].len() != b.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "incompatible matrix dimensions",
        ));
    }
    let (rows, inner, cols) = (a.len(), b.len(), b[0].len());
    let mut out = vec![vec![0.0; cols]; rows];
    for i in 0..rows {
        for k in 0..inner {
            let aik = a[i][k];
            if aik == 0.0 {
                continue;
            }
            for j in 0..cols {
                out[i][j] += aik * b[k][j];
            }
        }
    }
    Ok(out)
}

// ---------- Hand-rolled hashmap (open addressing, linear probing) ----------
#[derive(Clone, Copy)]
enum MapSlot {
    Empty,
    Deleted,
    Occupied(i64, i64),
}

#[pyclass]
pub struct FarmHashMap {
    slots: Vec<MapSlot>,
    len: usize,
}

impl FarmHashMap {
    fn grow(&mut self) {
        let new_cap = self.slots.len() * 2;
        let old = std::mem::replace(&mut self.slots, vec![MapSlot::Empty; new_cap]);
        self.len = 0;
        for slot in old {
            if let MapSlot::Occupied(k, v) = slot {
                self.raw_insert(k, v);
            }
        }
    }

    fn raw_insert(&mut self, key: i64, value: i64) -> Option<i64> {
        let cap = self.slots.len();
        let mut idx = (hash_i64(key) as usize) & (cap - 1);
        let mut first_deleted: Option<usize> = None;
        loop {
            match self.slots[idx] {
                MapSlot::Empty => {
                    let put_at = first_deleted.unwrap_or(idx);
                    self.slots[put_at] = MapSlot::Occupied(key, value);
                    self.len += 1;
                    return None;
                }
                MapSlot::Deleted => {
                    if first_deleted.is_none() {
                        first_deleted = Some(idx);
                    }
                    idx = (idx + 1) & (cap - 1);
                }
                MapSlot::Occupied(k, old_v) if k == key => {
                    self.slots[idx] = MapSlot::Occupied(key, value);
                    return Some(old_v);
                }
                _ => idx = (idx + 1) & (cap - 1),
            }
        }
    }
}

#[pymethods]
impl FarmHashMap {
    #[new]
    pub fn new() -> Self {
        FarmHashMap {
            slots: vec![MapSlot::Empty; 16],
            len: 0,
        }
    }

    pub fn insert(&mut self, key: i64, value: i64) -> Option<i64> {
        if (self.len + 1) * 10 >= self.slots.len() * 7 {
            self.grow();
        }
        self.raw_insert(key, value)
    }

    pub fn get(&self, key: i64) -> Option<i64> {
        let cap = self.slots.len();
        let mut idx = (hash_i64(key) as usize) & (cap - 1);
        let mut count = 0;
        loop {
            if count >= cap {
                return None;
            }
            match self.slots[idx] {
                MapSlot::Empty => return None,
                MapSlot::Occupied(k, v) if k == key => return Some(v),
                _ => {
                    idx = (idx + 1) & (cap - 1);
                    count += 1;
                }
            }
        }
    }

    pub fn remove(&mut self, key: i64) -> Option<i64> {
        let cap = self.slots.len();
        let mut idx = (hash_i64(key) as usize) & (cap - 1);
        let mut count = 0;
        loop {
            if count >= cap {
                return None;
            }
            match self.slots[idx] {
                MapSlot::Empty => return None,
                MapSlot::Occupied(k, v) if k == key => {
                    self.slots[idx] = MapSlot::Deleted;
                    self.len -= 1;
                    return Some(v);
                }
                _ => {
                    idx = (idx + 1) & (cap - 1);
                    count += 1;
                }
            }
        }
    }

    pub fn contains(&self, key: i64) -> bool {
        self.get(key).is_some()
    }

    pub fn __len__(&self) -> usize {
        self.len
    }

    pub fn keys(&self) -> Vec<i64> {
        self.slots
            .iter()
            .filter_map(|s| {
                if let MapSlot::Occupied(k, _) = s {
                    Some(*k)
                } else {
                    None
                }
            })
            .collect()
    }

    pub fn values(&self) -> Vec<i64> {
        self.slots
            .iter()
            .filter_map(|s| {
                if let MapSlot::Occupied(_, v) = s {
                    Some(*v)
                } else {
                    None
                }
            })
            .collect()
    }

    pub fn items(&self) -> Vec<(i64, i64)> {
        self.slots
            .iter()
            .filter_map(|s| {
                if let MapSlot::Occupied(k, v) = s {
                    Some((*k, *v))
                } else {
                    None
                }
            })
            .collect()
    }
}
