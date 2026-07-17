use numpy::PyUntypedArrayMethods;
use numpy::ndarray::Array2;
use numpy::{
    IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2, PyReadwriteArray1,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rand::prelude::*;
use rayon::prelude::*;

const PAR_THRESHOLD: usize = 10_000;

// =====================================================================
// Shared helpers
// =====================================================================

fn map_f64_array<'py>(
    py: Python<'py>,
    x: &PyReadonlyArray1<'py, f64>,
    f: impl Fn(f64) -> f64 + Sync,
) -> Bound<'py, PyArray1<f64>> {
    let view = x.as_array();
    let len = view.len();
    let result: Vec<f64> = py.allow_threads(|| {
        if len < PAR_THRESHOLD {
            view.iter().map(|&v| f(v)).collect()
        } else {
            match view.as_slice() {
                Some(slice) => slice.par_iter().map(|&v| f(v)).collect(),
                None => view.iter().map(|&v| f(v)).collect(),
            }
        }
    });
    result.into_pyarray(py)
}

fn zip_reduce_f64<'py>(
    py: Python<'py>,
    predictions: &PyReadonlyArray1<'py, f64>,
    targets: &PyReadonlyArray1<'py, f64>,
    f: impl Fn(f64, f64) -> f64 + Sync,
) -> PyResult<f64> {
    let pv = predictions.as_array();
    let tv = targets.as_array();
    if pv.len() != tv.len() || pv.is_empty() {
        return Err(PyValueError::new_err(
            "Predictions and targets must have matching non-zero lengths.",
        ));
    }
    let len = pv.len();
    let sum = py.allow_threads(|| {
        if len < PAR_THRESHOLD {
            pv.iter().zip(tv.iter()).map(|(&p, &t)| f(p, t)).sum()
        } else {
            match (pv.as_slice(), tv.as_slice()) {
                (Some(sp), Some(st)) => sp.par_iter().zip(st).map(|(&p, &t)| f(p, t)).sum(),
                _ => pv.iter().zip(tv.iter()).map(|(&p, &t)| f(p, t)).sum(),
            }
        }
    });
    Ok(sum)
}

// =====================================================================
// Activations
// =====================================================================

#[pyfunction]
pub fn relu<'py>(py: Python<'py>, x: PyReadonlyArray1<'py, f64>) -> Bound<'py, PyArray1<f64>> {
    map_f64_array(py, &x, |v| if v > 0.0 { v } else { 0.0 })
}

#[pyfunction]
pub fn relu_derivative<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<'py, f64>,
) -> Bound<'py, PyArray1<f64>> {
    map_f64_array(py, &x, |v| if v > 0.0 { 1.0 } else { 0.0 })
}

/// Leaky ReLU: avoids the "dead neuron" problem where a standard ReLU
/// permanently zeroes out a unit whose input never goes positive again.
#[pyfunction]
#[pyo3(signature = (x, alpha=0.01))]
pub fn leaky_relu<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<'py, f64>,
    alpha: f64,
) -> Bound<'py, PyArray1<f64>> {
    map_f64_array(py, &x, move |v| if v > 0.0 { v } else { alpha * v })
}

#[pyfunction]
#[pyo3(signature = (x, alpha=0.01))]
pub fn leaky_relu_derivative<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<'py, f64>,
    alpha: f64,
) -> Bound<'py, PyArray1<f64>> {
    map_f64_array(py, &x, move |v| if v > 0.0 { 1.0 } else { alpha })
}

/// ELU: smooth for x < 0 (unlike Leaky ReLU's hard kink), which can speed
/// up convergence by keeping mean activations closer to zero.
#[pyfunction]
#[pyo3(signature = (x, alpha=1.0))]
pub fn elu<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<'py, f64>,
    alpha: f64,
) -> Bound<'py, PyArray1<f64>> {
    map_f64_array(
        py,
        &x,
        move |v| if v > 0.0 { v } else { alpha * (v.exp() - 1.0) },
    )
}

#[pyfunction]
#[pyo3(signature = (x, alpha=1.0))]
pub fn elu_derivative<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<'py, f64>,
    alpha: f64,
) -> Bound<'py, PyArray1<f64>> {
    map_f64_array(py, &x, move |v| if v > 0.0 { 1.0 } else { alpha * v.exp() })
}

/// GELU (tanh approximation) — the activation used in BERT/GPT-style
/// transformers. Smoother than ReLU everywhere, which matters for
/// transformer training stability.
#[pyfunction]
pub fn gelu<'py>(py: Python<'py>, x: PyReadonlyArray1<'py, f64>) -> Bound<'py, PyArray1<f64>> {
    const SQRT_2_OVER_PI: f64 = 0.7978845608028654;
    map_f64_array(py, &x, move |v| {
        0.5 * v * (1.0 + (SQRT_2_OVER_PI * (v + 0.044715 * v.powi(3))).tanh())
    })
}

/// Swish / SiLU: x * sigmoid(x). Used in EfficientNet and several modern
/// architectures; unlike ReLU, it's non-monotonic and smooth everywhere.
#[pyfunction]
pub fn silu<'py>(py: Python<'py>, x: PyReadonlyArray1<'py, f64>) -> Bound<'py, PyArray1<f64>> {
    map_f64_array(py, &x, |v| v / (1.0 + (-v).exp()))
}

#[pyfunction]
pub fn silu_derivative<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<'py, f64>,
) -> Bound<'py, PyArray1<f64>> {
    map_f64_array(py, &x, |v| {
        let s = 1.0 / (1.0 + (-v).exp());
        s + v * s * (1.0 - s)
    })
}

#[pyfunction]
pub fn sigmoid<'py>(py: Python<'py>, x: PyReadonlyArray1<'py, f64>) -> Bound<'py, PyArray1<f64>> {
    map_f64_array(py, &x, |v| 1.0 / (1.0 + (-v).exp()))
}

#[pyfunction]
pub fn sigmoid_derivative<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<'py, f64>,
) -> Bound<'py, PyArray1<f64>> {
    map_f64_array(py, &x, |v| {
        let s = 1.0 / (1.0 + (-v).exp());
        s * (1.0 - s)
    })
}

#[pyfunction]
pub fn tanh<'py>(py: Python<'py>, x: PyReadonlyArray1<'py, f64>) -> Bound<'py, PyArray1<f64>> {
    map_f64_array(py, &x, |v| v.tanh())
}

#[pyfunction]
pub fn tanh_derivative<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<'py, f64>,
) -> Bound<'py, PyArray1<f64>> {
    map_f64_array(py, &x, |v| {
        let t = v.tanh();
        1.0 - t * t
    })
}

// =====================================================================
// Softmax + fused softmax/cross-entropy
// =====================================================================

fn softmax_row_inplace(row_in: &[f64], row_out: &mut [f64]) {
    let max = row_in.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let mut sum = 0.0;
    for (o, &v) in row_out.iter_mut().zip(row_in) {
        let e = (v - max).exp(); // subtracting max keeps exp() from overflowing
        *o = e;
        sum += e;
    }
    for o in row_out.iter_mut() {
        *o /= sum;
    }
}

/// Row-wise softmax over a (batch, classes) matrix. Numerically stable —
/// subtracts the row max before exponentiating, which a naive
/// `exp(x) / sum(exp(x))` implementation typically skips and then overflows
/// on any reasonably large logit.
#[pyfunction]
pub fn softmax<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<'py, f64>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let shape = x.shape();
    let (rows, cols) = (shape[0], shape[1]);
    let input = x
        .as_slice()
        .map_err(|_| PyValueError::new_err("input must be C-contiguous"))?;

    let mut out = vec![0.0f64; rows * cols];
    py.allow_threads(|| {
        if rows < PAR_THRESHOLD {
            for r in 0..rows {
                softmax_row_inplace(
                    &input[r * cols..(r + 1) * cols],
                    &mut out[r * cols..(r + 1) * cols],
                );
            }
        } else {
            out.par_chunks_mut(cols).enumerate().for_each(|(r, chunk)| {
                softmax_row_inplace(&input[r * cols..(r + 1) * cols], chunk)
            });
        }
    });

    let arr = Array2::from_shape_vec((rows, cols), out)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(arr.into_pyarray(py))
}

/// Fused softmax + cross-entropy: computes the mean loss AND the gradient
/// w.r.t. the logits in a single pass, ready to backprop directly. This is
/// exactly what PyTorch's `F.cross_entropy` does internally — computing
/// softmax and cross-entropy as separate steps in Python is both slower
/// (two full passes over the data) and less numerically stable (you'd be
/// taking log() of an already-exponentiated, already-normalized value).
/// `targets` is class indices (like PyTorch), not one-hot vectors.
#[pyfunction]
pub fn softmax_cross_entropy<'py>(
    py: Python<'py>,
    logits: PyReadonlyArray2<'py, f64>,
    targets: PyReadonlyArray1<'py, i64>,
) -> PyResult<(f64, Bound<'py, PyArray2<f64>>)> {
    let shape = logits.shape();
    let (rows, cols) = (shape[0], shape[1]);
    let input = logits
        .as_slice()
        .map_err(|_| PyValueError::new_err("logits must be C-contiguous"))?;
    let tgt = targets
        .as_slice()
        .map_err(|_| PyValueError::new_err("targets must be contiguous"))?;

    if tgt.len() != rows {
        return Err(PyValueError::new_err(
            "targets length must match logits batch size",
        ));
    }
    if tgt.iter().any(|&c| c < 0 || c as usize >= cols) {
        return Err(PyValueError::new_err(
            "target class index out of range for logits.shape[1]",
        ));
    }

    let mut grad = vec![0.0f64; rows * cols];

    let compute_row = |r: usize, out_row: &mut [f64]| -> f64 {
        softmax_row_inplace(&input[r * cols..(r + 1) * cols], out_row);
        let t = tgt[r] as usize;
        let p_correct = out_row[t].max(1e-15);
        out_row[t] -= 1.0; // softmax probs become (softmax - one_hot), i.e. dL/dlogits, pre-batch-normalization
        -p_correct.ln()
    };

    let losses: Vec<f64> = py.allow_threads(|| {
        if rows < PAR_THRESHOLD {
            (0..rows)
                .map(|r| compute_row(r, &mut grad[r * cols..(r + 1) * cols]))
                .collect()
        } else {
            grad.par_chunks_mut(cols)
                .enumerate()
                .map(|(r, chunk)| compute_row(r, chunk))
                .collect()
        }
    });

    let n = rows as f64;
    let mean_loss = losses.iter().sum::<f64>() / n;
    grad.iter_mut().for_each(|g| *g /= n);

    let grad_arr = Array2::from_shape_vec((rows, cols), grad)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok((mean_loss, grad_arr.into_pyarray(py)))
}

/// One-hot encode class indices into a (n, num_classes) float matrix.
#[pyfunction]
pub fn one_hot<'py>(
    py: Python<'py>,
    indices: PyReadonlyArray1<'py, i64>,
    num_classes: usize,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let idx = indices
        .as_slice()
        .map_err(|_| PyValueError::new_err("indices must be contiguous"))?;
    if idx.iter().any(|&c| c < 0 || c as usize >= num_classes) {
        return Err(PyValueError::new_err(
            "class index out of range for num_classes",
        ));
    }
    let rows = idx.len();
    let mut data = vec![0.0f64; rows * num_classes];
    py.allow_threads(|| {
        for (r, &class) in idx.iter().enumerate() {
            data[r * num_classes + class as usize] = 1.0;
        }
    });
    let arr = Array2::from_shape_vec((rows, num_classes), data)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(arr.into_pyarray(py))
}

// =====================================================================
// Losses
// =====================================================================

#[pyfunction]
pub fn mse_loss(
    py: Python<'_>,
    predictions: PyReadonlyArray1<'_, f64>,
    targets: PyReadonlyArray1<'_, f64>,
) -> PyResult<f64> {
    let n = predictions.as_array().len() as f64;
    let sum = zip_reduce_f64(py, &predictions, &targets, |p, t| {
        let diff = p - t;
        diff * diff
    })?;
    Ok(sum / n)
}

#[pyfunction]
pub fn categorical_cross_entropy(
    py: Python<'_>,
    predictions: PyReadonlyArray1<'_, f64>,
    targets: PyReadonlyArray1<'_, f64>,
) -> PyResult<f64> {
    let n = predictions.as_array().len() as f64;
    let sum = zip_reduce_f64(py, &predictions, &targets, |p, t| {
        let clipped_p = p.clamp(1e-15, 1.0 - 1e-15);
        -t * clipped_p.ln()
    })?;
    Ok(sum / n)
}

/// Mean Absolute Error (MAE / L1) Loss: L = (1/N) * sum(|y_pred - y_true|)
#[pyfunction]
pub fn mae_loss(
    py: Python<'_>,
    predictions: PyReadonlyArray1<'_, f64>,
    targets: PyReadonlyArray1<'_, f64>,
) -> PyResult<f64> {
    let n = predictions.as_array().len() as f64;
    let sum = zip_reduce_f64(py, &predictions, &targets, |p, t| (p - t).abs())?;
    Ok(sum / n)
}

/// Binary Cross Entropy (BCE) Loss: L = -(1/N) * sum(y_true*log(y_pred) + (1-y_true)*log(1-y_pred))
#[pyfunction]
pub fn bce_loss(
    py: Python<'_>,
    predictions: PyReadonlyArray1<'_, f64>,
    targets: PyReadonlyArray1<'_, f64>,
) -> PyResult<f64> {
    let n = predictions.as_array().len() as f64;
    let sum = zip_reduce_f64(py, &predictions, &targets, |p, t| {
        let clipped_p = p.clamp(1e-15, 1.0 - 1e-15);
        t * clipped_p.ln() + (1.0 - t) * (1.0 - clipped_p).ln()
    })?;
    Ok(-sum / n)
}

// =====================================================================
// Optimizers — in-place parameter updates. This is where a Rust backend
// matters most: a hand-written Adam/SGD loop over millions of parameters
// in pure Python is one of the slowest parts of a from-scratch training
// loop, and these run with the GIL released across the whole update.
// =====================================================================

#[pyfunction]
pub fn sgd_momentum_step(
    py: Python<'_>,
    mut params: PyReadwriteArray1<'_, f64>,
    grads: PyReadonlyArray1<'_, f64>,
    mut velocity: PyReadwriteArray1<'_, f64>,
    lr: f64,
    momentum: f64,
) -> PyResult<()> {
    let g = grads
        .as_slice()
        .map_err(|_| PyValueError::new_err("grads must be contiguous"))?;
    let p = params
        .as_slice_mut()
        .map_err(|_| PyValueError::new_err("params must be contiguous"))?;
    let v = velocity
        .as_slice_mut()
        .map_err(|_| PyValueError::new_err("velocity must be contiguous"))?;
    if p.len() != g.len() || p.len() != v.len() {
        return Err(PyValueError::new_err(
            "params, grads, and velocity must be the same length",
        ));
    }

    py.allow_threads(|| {
        if p.len() < PAR_THRESHOLD {
            for i in 0..p.len() {
                v[i] = momentum * v[i] + g[i];
                p[i] -= lr * v[i];
            }
        } else {
            p.par_iter_mut()
                .zip(v.par_iter_mut())
                .zip(g.par_iter())
                .for_each(|((pi, vi), &gi)| {
                    *vi = momentum * *vi + gi;
                    *pi -= lr * *vi;
                });
        }
    });
    Ok(())
}

/// Adam optimizer step. `m` and `v` are the first/second moment buffers
/// the caller owns and passes back in each step; `t` is the 1-indexed
/// timestep, used for bias correction.
#[pyfunction]
#[pyo3(signature = (params, grads, m, v, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8, t=1))]
pub fn adam_step(
    py: Python<'_>,
    mut params: PyReadwriteArray1<'_, f64>,
    grads: PyReadonlyArray1<'_, f64>,
    mut m: PyReadwriteArray1<'_, f64>,
    mut v: PyReadwriteArray1<'_, f64>,
    lr: f64,
    beta1: f64,
    beta2: f64,
    eps: f64,
    t: i32,
) -> PyResult<()> {
    let g = grads
        .as_slice()
        .map_err(|_| PyValueError::new_err("grads must be contiguous"))?;
    let p = params
        .as_slice_mut()
        .map_err(|_| PyValueError::new_err("params must be contiguous"))?;
    let m_s = m
        .as_slice_mut()
        .map_err(|_| PyValueError::new_err("m must be contiguous"))?;
    let v_s = v
        .as_slice_mut()
        .map_err(|_| PyValueError::new_err("v must be contiguous"))?;
    if p.len() != g.len() || p.len() != m_s.len() || p.len() != v_s.len() {
        return Err(PyValueError::new_err(
            "params, grads, m, and v must be the same length",
        ));
    }

    let bias_correction1 = 1.0 - beta1.powi(t);
    let bias_correction2 = 1.0 - beta2.powi(t);

    py.allow_threads(|| {
        let update = |pi: &mut f64, mi: &mut f64, vi: &mut f64, gi: f64| {
            *mi = beta1 * *mi + (1.0 - beta1) * gi;
            *vi = beta2 * *vi + (1.0 - beta2) * gi * gi;
            let m_hat = *mi / bias_correction1;
            let v_hat = *vi / bias_correction2;
            *pi -= lr * m_hat / (v_hat.sqrt() + eps);
        };
        if p.len() < PAR_THRESHOLD {
            for i in 0..p.len() {
                update(&mut p[i], &mut m_s[i], &mut v_s[i], g[i]);
            }
        } else {
            p.par_iter_mut()
                .zip(m_s.par_iter_mut())
                .zip(v_s.par_iter_mut())
                .zip(g.par_iter())
                .for_each(|(((pi, mi), vi), &gi)| update(pi, mi, vi, gi));
        }
    });
    Ok(())
}

/// Clips gradients in place by global L2 norm (matches
/// torch.nn.utils.clip_grad_norm_ semantics) and returns the pre-clip norm,
/// useful for logging/debugging exploding gradients.
#[pyfunction]
pub fn clip_grad_norm(
    py: Python<'_>,
    mut grads: PyReadwriteArray1<'_, f64>,
    max_norm: f64,
) -> PyResult<f64> {
    let g = grads
        .as_slice_mut()
        .map_err(|_| PyValueError::new_err("grads must be contiguous"))?;
    let norm = py.allow_threads(|| {
        let sum_sq: f64 = if g.len() < PAR_THRESHOLD {
            g.iter().map(|&x| x * x).sum()
        } else {
            g.par_iter().map(|&x| x * x).sum()
        };
        let norm = sum_sq.sqrt();
        if norm > max_norm && norm > 0.0 {
            let scale = max_norm / norm;
            if g.len() < PAR_THRESHOLD {
                g.iter_mut().for_each(|x| *x *= scale);
            } else {
                g.par_iter_mut().for_each(|x| *x *= scale);
            }
        }
        norm
    });
    Ok(norm)
}

/// Inverted dropout: zeroes each element with probability `p` and scales
/// survivors by 1/(1-p), so no rescaling is needed at inference time.
/// Pass `seed` for reproducible masks (e.g. in tests), omit it for a fresh
/// random mask drawn from a thread-local RNG each call.
#[pyfunction]
#[pyo3(signature = (x, p, seed=None))]
pub fn dropout<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<'py, f64>,
    p: f64,
    seed: Option<u64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    if !(0.0..1.0).contains(&p) {
        return Err(PyValueError::new_err(
            "dropout probability p must be in [0, 1)",
        ));
    }
    let input = x
        .as_slice()
        .map_err(|_| PyValueError::new_err("input must be contiguous"))?;
    let keep_prob = 1.0 - p;
    let scale = 1.0 / keep_prob;

    let result: Vec<f64> = py.allow_threads(|| {
        if let Some(s) = seed {
            let mut rng = StdRng::seed_from_u64(s);
            input
                .iter()
                .map(|&v| {
                    if rng.r#gen::<f64>() < keep_prob {
                        v * scale
                    } else {
                        0.0
                    }
                })
                .collect()
        } else {
            input
                .par_iter()
                .map_init(rand::thread_rng, |rng, &v| {
                    if rng.r#gen::<f64>() < keep_prob {
                        v * scale
                    } else {
                        0.0
                    }
                })
                .collect()
        }
    });
    Ok(result.into_pyarray(py))
}
