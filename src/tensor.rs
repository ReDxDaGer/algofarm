use matrixmultiply::dgemm;
use numpy::PyArray1;
use numpy::PyReadonlyArrayDyn;
use numpy::PyUntypedArrayMethods;
use pyo3::Bound;
use pyo3::prelude::*;
use pyo3::types::PyType;
use rand::prelude::*;
use rand_distr::{Distribution, StandardNormal};
use rayon::prelude::*;
use std::sync::Arc;
const PAR_THRESHOLD: usize = 10_000;

#[pyclass]
#[derive(Clone)]
pub struct Tensor {
    // Arc, not a bare Vec<f64>: cloning a Tensor (which #[derive(Clone)]
    // does implicitly, and which reshape/view-style ops need) becomes an
    // O(1) refcount bump instead of an O(n) buffer copy.
    pub data: Arc<Vec<f64>>,
    #[pyo3(get)]
    pub shape: Vec<usize>,
    #[pyo3(get)]
    pub strides: Vec<usize>,
}

fn calculate_strides(shape: &[usize]) -> Vec<usize> {
    if shape.is_empty() {
        return Vec::new();
    }
    let mut strides = vec![1; shape.len()];
    for i in (0..shape.len() - 1).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    strides
}

fn elementwise_binary(
    py: Python<'_>,
    a: &[f64],
    b: &[f64],
    f: impl Fn(f64, f64) -> f64 + Sync,
) -> Vec<f64> {
    py.allow_threads(|| {
        if a.len() < PAR_THRESHOLD {
            a.iter().zip(b).map(|(&x, &y)| f(x, y)).collect()
        } else {
            a.par_iter().zip(b).map(|(&x, &y)| f(x, y)).collect()
        }
    })
}

fn elementwise_scalar(
    py: Python<'_>,
    a: &[f64],
    scalar: f64,
    f: impl Fn(f64, f64) -> f64 + Sync,
) -> Vec<f64> {
    py.allow_threads(|| {
        if a.len() < PAR_THRESHOLD {
            a.iter().map(|&x| f(x, scalar)).collect()
        } else {
            a.par_iter().map(|&x| f(x, scalar)).collect()
        }
    })
}

#[pymethods]
impl Tensor {
    #[new]
    pub fn new(array: PyReadonlyArrayDyn<'_, f64>) -> Self {
        let shape = array.shape().to_vec();

        // Fast path: if the NumPy array is contiguous, perform a rapid bulk memory copy.
        // Slow path: if it's sliced or has non-standard strides, fall back to iterating.
        let data = if let Ok(slice) = array.as_slice() {
            slice.to_vec()
        } else {
            array.as_array().iter().copied().collect()
        };

        let strides = calculate_strides(&shape);

        Tensor {
            data: Arc::new(data),
            shape,
            strides,
        }
    }

    #[getter]
    pub fn data<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        // Safe, highly optimized zero-copy layout pointing to our Arc slice data
        PyArray1::from_slice(py, &self.data)
    }

    pub fn sub(&self, py: Python<'_>, other: &Tensor) -> PyResult<Self> {
        if self.shape != other.shape {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Shapes must match for subtraction.",
            ));
        }
        let result_data = elementwise_binary(py, &self.data, &other.data, |x, y| x - y);
        Ok(Tensor {
            data: Arc::new(result_data),
            shape: self.shape.clone(),
            strides: self.strides.clone(),
        })
    }

    pub fn mul_elementwise(&self, py: Python<'_>, other: &Tensor) -> PyResult<Self> {
        if self.shape != other.shape {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Shapes must match for element-wise multiplication.",
            ));
        }
        let result_data = elementwise_binary(py, &self.data, &other.data, |x, y| x * y);
        Ok(Tensor {
            data: Arc::new(result_data),
            shape: self.shape.clone(),
            strides: self.strides.clone(),
        })
    }

    pub fn mul_scalar(&self, py: Python<'_>, scalar: f64) -> PyResult<Self> {
        let result_data = elementwise_scalar(py, &self.data, scalar, |x, s| x * s);
        Ok(Tensor {
            data: Arc::new(result_data),
            shape: self.shape.clone(),
            strides: self.strides.clone(),
        })
    }

    pub fn reshape(&self, new_shape: Vec<usize>) -> PyResult<Self> {
        let expected_size: usize = new_shape.iter().product();
        let current_size: usize = self.shape.iter().product();
        if expected_size != current_size {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Total number of elements cannot change during reshape.",
            ));
        }
        let strides = calculate_strides(&new_shape);
        Ok(Tensor {
            data: Arc::clone(&self.data),
            shape: new_shape,
            strides,
        })
    }

    pub fn add(&self, py: Python<'_>, other: &Tensor) -> PyResult<Self> {
        if self.shape != other.shape {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Shapes must match for element-wise addition.",
            ));
        }
        let result_data = elementwise_binary(py, &self.data, &other.data, |x, y| x + y);
        Ok(Tensor {
            data: Arc::new(result_data),
            shape: self.shape.clone(),
            strides: self.strides.clone(),
        })
    }

    #[pyo3(signature = (other, transpose_a=false, transpose_b=false))]
    pub fn matmul(
        &self,
        py: Python<'_>,
        other: &Tensor,
        transpose_a: bool,
        transpose_b: bool,
    ) -> PyResult<Self> {
        if self.shape.len() != 2 || other.shape.len() != 2 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "matmul requires rank-2 tensors.",
            ));
        }

        let m = if transpose_a {
            self.shape[1]
        } else {
            self.shape[0]
        };
        let k_a = if transpose_a {
            self.shape[0]
        } else {
            self.shape[1]
        };

        let k_b = if transpose_b {
            other.shape[1]
        } else {
            other.shape[0]
        };
        let n = if transpose_b {
            other.shape[0]
        } else {
            other.shape[1]
        };

        if k_a != k_b {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Dimension mismatch for multiplication: {} vs {}.",
                k_a, k_b
            )));
        }

        let rsa = if transpose_a {
            1
        } else {
            self.strides[0] as isize
        };
        let csa = if transpose_a {
            self.strides[0] as isize
        } else {
            1
        };

        let rsb = if transpose_b {
            1
        } else {
            other.strides[0] as isize
        };
        let csb = if transpose_b {
            other.strides[0] as isize
        } else {
            1
        };

        let mut result_data = vec![0.0; m * n];

        py.allow_threads(|| unsafe {
            dgemm(
                m,
                k_a,
                n,
                1.0,
                self.data.as_ptr(),
                rsa,
                csa,
                other.data.as_ptr(),
                rsb,
                csb,
                0.0,
                result_data.as_mut_ptr(),
                n as isize,
                1,
            );
        });

        Ok(Tensor {
            data: Arc::new(result_data),
            shape: vec![m, n],
            strides: vec![n, 1],
        })
    }

    pub fn to_list(&self, py: Python<'_>) -> PyResult<PyObject> {
        fn recurse(
            py: Python<'_>,
            data: &[f64],
            shape: &[usize],
            index: &mut usize,
        ) -> PyResult<PyObject> {
            if shape.is_empty() {
                let val = data[*index];
                *index += 1;
                // Use into_pyobject, convert to generic Bound<PyAny>, then unbind to raw PyObject
                return Ok(val.into_pyobject(py)?.into_any().unbind());
            }
            let mut list = Vec::new();
            for _ in 0..shape[0] {
                list.push(recurse(py, data, &shape[1..], index)?);
            }
            Ok(list.into_pyobject(py)?.into_any().unbind())
        }

        let mut index = 0;
        recurse(py, &self.data, &self.shape, &mut index)
    }

    #[classmethod]
    #[pyo3(signature = (shape, seed=None))]
    pub fn rand(
        _cls: &Bound<'_, PyType>,
        py: Python<'_>,
        shape: Vec<usize>,
        seed: Option<u64>,
    ) -> PyResult<Self> {
        let size: usize = shape.iter().product();

        let data = py.allow_threads(|| {
            if let Some(s) = seed {
                let mut rng = StdRng::seed_from_u64(s);
                (0..size).map(|_| rng.r#gen::<f64>()).collect()
            } else {
                (0..size)
                    .into_par_iter()
                    .map_init(thread_rng, |rng, _| rng.r#gen::<f64>())
                    .collect()
            }
        });

        let strides = calculate_strides(&shape);
        Ok(Tensor {
            data: Arc::new(data),
            shape,
            strides,
        })
    }

    #[classmethod]
    #[pyo3(signature = (shape, seed=None))]
    pub fn randn(
        _cls: &Bound<'_, PyType>,
        py: Python<'_>,
        shape: Vec<usize>,
        seed: Option<u64>,
    ) -> PyResult<Self> {
        let size: usize = shape.iter().product();

        let data = py.allow_threads(|| {
            if let Some(s) = seed {
                let mut rng = StdRng::seed_from_u64(s);
                (0..size).map(|_| StandardNormal.sample(&mut rng)).collect()
            } else {
                (0..size)
                    .into_par_iter()
                    .map_init(thread_rng, |rng, _| StandardNormal.sample(rng))
                    .collect()
            }
        });

        let strides = calculate_strides(&shape);
        Ok(Tensor {
            data: Arc::new(data),
            shape,
            strides,
        })
    }

    pub fn add_bias_inplace(
        &mut self,
        py: Python<'_>,
        bias: PyReadonlyArrayDyn<'_, f64>,
    ) -> PyResult<()> {
        let n = self.shape[0];
        let d = self.shape[1];

        let bias_slice = bias.as_slice().map_err(|_| {
            pyo3::exceptions::PyValueError::new_err("Bias array must be contiguous.")
        })?;

        if bias_slice.len() != d {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Bias length must match the second dimension of the tensor: {} vs {}.",
                bias_slice.len(),
                d
            )));
        }

        let data = Arc::make_mut(&mut self.data);

        py.allow_threads(|| {
            if data.len() < PAR_THRESHOLD {
                for i in 0..n {
                    let offset = i * d;
                    for j in 0..d {
                        data[offset + j] += bias_slice[j];
                    }
                }
            } else {
                data.par_chunks_mut(d).for_each(|row| {
                    for j in 0..d {
                        row[j] += bias_slice[j];
                    }
                });
            }
        });

        Ok(())
    }

    pub fn relu_inplace(&mut self, py: Python<'_>) {
        let data = Arc::make_mut(&mut self.data);
        py.allow_threads(|| {
            if data.len() < PAR_THRESHOLD {
                data.iter_mut().for_each(|x| {
                    if *x < 0.0 {
                        *x = 0.0;
                    }
                });
            } else {
                data.par_iter_mut().for_each(|x| {
                    if *x < 0.0 {
                        *x = 0.0;
                    }
                });
            }
        });
    }

    pub fn relu_derivative(&self, py: Python<'_>) -> Self {
        let result_data: Vec<f64> = py.allow_threads(|| {
            if self.data.len() < PAR_THRESHOLD {
                self.data
                    .iter()
                    .map(|&x| if x > 0.0 { 1.0 } else { 0.0 })
                    .collect()
            } else {
                self.data
                    .par_iter()
                    .map(|&x| if x > 0.0 { 1.0 } else { 0.0 })
                    .collect()
            }
        });

        Tensor {
            data: Arc::new(result_data),
            shape: self.shape.clone(),
            strides: self.strides.clone(),
        }
    }
}
