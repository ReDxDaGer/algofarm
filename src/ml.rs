use pyo3::prelude::*;
use rayon::prelude::*;

const PAR_THRESHOLD: usize = 10_000;

#[inline]
fn squared_euclidean(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(x, y)| {
            let diff = x - y;
            diff * diff
        })
        .sum()
}

#[inline]
fn single_manhattan(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).sum()
}

#[inline]
fn single_cosine(a: &[f64], b: &[f64]) -> f64 {
    let mut dot_product = 0.0;
    let mut norm_a = 0.0;
    let mut norm_b = 0.0;

    for (x, y) in a.iter().zip(b) {
        dot_product += x * y;
        norm_a += x * x;
        norm_b += y * y;
    }

    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }
    dot_product / (norm_a.sqrt() * norm_b.sqrt())
}

#[pyfunction]
pub fn pairwise_manhattan(
    py: Python<'_>,
    matrix_a: Vec<Vec<f64>>,
    matrix_b: Vec<Vec<f64>>,
) -> PyResult<Vec<Vec<f64>>> {
    if matrix_a.is_empty() || matrix_b.is_empty() {
        return Ok(Vec::new());
    }

    let result = py.allow_threads(|| {
        if matrix_a.len() < PAR_THRESHOLD {
            matrix_a
                .iter()
                .map(|row_a| {
                    matrix_b
                        .iter()
                        .map(|row_b| single_manhattan(row_a, row_b))
                        .collect()
                })
                .collect()
        } else {
            matrix_a
                .par_iter()
                .map(|row_a| {
                    matrix_b
                        .iter()
                        .map(|row_b| single_manhattan(row_a, row_b))
                        .collect()
                })
                .collect()
        }
    });

    Ok(result)
}

#[pyfunction]
pub fn pairwise_cosine(
    py: Python<'_>,
    matrix_a: Vec<Vec<f64>>,
    matrix_b: Vec<Vec<f64>>,
) -> PyResult<Vec<Vec<f64>>> {
    if matrix_a.is_empty() || matrix_b.is_empty() {
        return Ok(Vec::new());
    }

    let result = py.allow_threads(|| {
        if matrix_a.len() < PAR_THRESHOLD {
            matrix_a
                .iter()
                .map(|row_a| {
                    matrix_b
                        .iter()
                        .map(|row_b| single_cosine(row_a, row_b))
                        .collect()
                })
                .collect()
        } else {
            matrix_a
                .par_iter()
                .map(|row_a| {
                    matrix_b
                        .iter()
                        .map(|row_b| single_cosine(row_a, row_b))
                        .collect()
                })
                .collect()
        }
    });

    Ok(result)
}

/// Helper to copy a flat vector representation into a square matrix
fn to_square_matrix(data: &[Vec<f64>]) -> Option<Vec<Vec<f64>>> {
    let n = data.len();
    if n == 0 {
        return None;
    }
    for row in data {
        if row.len() != n {
            return None;
        }
    }
    Some(data.to_vec())
}

#[pyfunction]
pub fn determinant(py: Python<'_>, matrix: Vec<Vec<f64>>) -> PyResult<f64> {
    let mut a = match to_square_matrix(&matrix) {
        Some(m) => m,
        None => {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Matrix must be square.",
            ));
        }
    };

    let n = a.len();
    if n == 0 {
        return Ok(0.0);
    }
    if n == 1 {
        return Ok(a[0][0]);
    }

    let det = py.allow_threads(|| {
        let mut d = 1.0;
        for i in 0..n {
            let mut pivot_row = i;
            for r in (i + 1)..n {
                if a[r][i].abs() > a[pivot_row][i].abs() {
                    pivot_row = r;
                }
            }

            if i != pivot_row {
                a.swap(i, pivot_row);
                d *= -1.0;
            }

            if a[i][i].abs() < 1e-12 {
                return 0.0;
            }

            d *= a[i][i];

            for r in (i + 1)..n {
                let factor = a[r][i] / a[i][i];
                for c in i..n {
                    a[r][c] -= factor * a[i][c];
                }
            }
        }
        d
    });

    Ok(det)
}

#[pyfunction]
pub fn matrix_inverse(py: Python<'_>, matrix: Vec<Vec<f64>>) -> PyResult<Vec<Vec<f64>>> {
    let mut a = match to_square_matrix(&matrix) {
        Some(m) => m,
        None => {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Matrix must be square.",
            ));
        }
    };

    let n = a.len();
    if n == 0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Matrix cannot be empty.",
        ));
    }

    let inv = py.allow_threads(|| {
        let mut i1 = vec![vec![0.0; n]; n];
        for i in 0..n {
            i1[i][i] = 1.0;
        }

        for i in 0..n {
            let mut pivot_row = i;
            for r in (i + 1)..n {
                if a[r][i].abs() > a[pivot_row][i].abs() {
                    pivot_row = r;
                }
            }

            if a[pivot_row][i].abs() < 1e-12 {
                return None; // Singular matrix
            }

            if i != pivot_row {
                a.swap(i, pivot_row);
                i1.swap(i, pivot_row);
            }

            let pivot_val = a[i][i];
            for c in 0..n {
                a[i][c] /= pivot_val;
                i1[i][c] /= pivot_val;
            }

            for r in 0..n {
                if r != i {
                    let factor = a[r][i];
                    for c in 0..n {
                        a[r][c] -= factor * a[i][c];
                        // BUG FIX: was `i1[r][c] -= factor * i1[r][c]` — that
                        // subtracts a multiple of the row from itself instead
                        // of subtracting a multiple of the PIVOT row (row i).
                        // Gauss-Jordan requires referencing i1[i][c] here, the
                        // same as the `a` half does with a[i][c] above.
                        i1[r][c] -= factor * i1[i][c];
                    }
                }
            }
        }
        Some(i1)
    });

    match inv {
        Some(result) => Ok(result),
        None => Err(pyo3::exceptions::PyValueError::new_err(
            "Matrix is singular and cannot be inverted.",
        )),
    }
}

#[pyfunction]
pub fn kmeans(
    py: Python<'_>,
    data: Vec<Vec<f64>>,
    k: usize,
    max_iters: usize,
    tolerance: f64,
) -> PyResult<(Vec<Vec<f64>>, Vec<usize>)> {
    if data.is_empty() || k == 0 || k > data.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Invalid input data size or cluster count 'k'.",
        ));
    }

    let num_samples = data.len();
    let num_features = data[0].len();

    let result = py.allow_threads(move || {
        let step = num_samples / k;
        let mut centroids: Vec<Vec<f64>> = (0..k).map(|i| data[i * step].clone()).collect();
        let mut assignments = vec![0; num_samples];
        let mut iteration = 0;
        let mut converged = false;

        while iteration < max_iters && !converged {
            let assign_point = |point: &Vec<f64>| -> usize {
                let mut min_dist = f64::MAX;
                let mut closest_centroid = 0;
                for (idx, centroid) in centroids.iter().enumerate() {
                    let dist = squared_euclidean(point, centroid);
                    if dist < min_dist {
                        min_dist = dist;
                        closest_centroid = idx;
                    }
                }
                closest_centroid
            };

            let new_assignments: Vec<usize> = if num_samples < PAR_THRESHOLD {
                data.iter().map(assign_point).collect()
            } else {
                data.par_iter().map(assign_point).collect()
            };

            let mut new_centroids = vec![vec![0.0; num_features]; k];
            let mut centroid_counts = vec![0; k];

            for (i, &cluster_idx) in new_assignments.iter().enumerate() {
                centroid_counts[cluster_idx] += 1;
                for j in 0..num_features {
                    new_centroids[cluster_idx][j] += data[i][j];
                }
            }

            for idx in 0..k {
                if centroid_counts[idx] > 0 {
                    for j in 0..num_features {
                        new_centroids[idx][j] /= centroid_counts[idx] as f64;
                    }
                } else {
                    let fallback_idx = (idx * step) % num_samples;
                    new_centroids[idx] = data[fallback_idx].clone();
                }
            }

            let mut shift = 0.0;
            for i in 0..k {
                shift += squared_euclidean(&centroids[i], &new_centroids[i]).sqrt();
            }

            assignments = new_assignments;
            centroids = new_centroids;

            if shift < tolerance {
                converged = true;
            }

            iteration += 1;
        }

        (centroids, assignments)
    });

    Ok(result)
}
