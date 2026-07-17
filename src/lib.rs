use pyo3::prelude::*;

mod dl_formulas;
mod dsa;
mod ml;
mod tensor; // Cleanly linked our algorithms module

#[pymodule]
fn algofarm(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // ---------- DSA Module Functions ----------
    m.add_function(wrap_pyfunction!(dsa::quicksort, m)?)?;
    m.add_function(wrap_pyfunction!(dsa::quicksort_f64, m)?)?;
    m.add_function(wrap_pyfunction!(dsa::mergesort, m)?)?;
    m.add_function(wrap_pyfunction!(dsa::parallel_sort, m)?)?;
    m.add_function(wrap_pyfunction!(dsa::binary_search, m)?)?;
    m.add_function(wrap_pyfunction!(dsa::kth_smallest, m)?)?;
    m.add_function(wrap_pyfunction!(dsa::top_k, m)?)?;
    m.add_function(wrap_pyfunction!(dsa::dedup, m)?)?;
    m.add_function(wrap_pyfunction!(dsa::levenshtein, m)?)?;
    m.add_function(wrap_pyfunction!(dsa::gcd, m)?)?;
    m.add_function(wrap_pyfunction!(dsa::lcm, m)?)?;
    m.add_function(wrap_pyfunction!(dsa::mod_pow, m)?)?;
    m.add_function(wrap_pyfunction!(dsa::is_prime, m)?)?;
    m.add_function(wrap_pyfunction!(dsa::sieve, m)?)?;
    m.add_function(wrap_pyfunction!(dsa::fibonacci, m)?)?;
    m.add_function(wrap_pyfunction!(dsa::mean, m)?)?;
    m.add_function(wrap_pyfunction!(dsa::median, m)?)?;
    m.add_function(wrap_pyfunction!(dsa::variance, m)?)?;
    m.add_function(wrap_pyfunction!(dsa::std_dev, m)?)?;
    m.add_function(wrap_pyfunction!(dsa::dot_product, m)?)?;
    m.add_function(wrap_pyfunction!(dsa::matmul, m)?)?;

    // ---------- ML / Core Math Functions ----------
    m.add_function(wrap_pyfunction!(ml::kmeans, m)?)?;
    m.add_function(wrap_pyfunction!(ml::pairwise_manhattan, m)?)?;
    m.add_function(wrap_pyfunction!(ml::pairwise_cosine, m)?)?;
    m.add_function(wrap_pyfunction!(ml::determinant, m)?)?;
    m.add_function(wrap_pyfunction!(ml::matrix_inverse, m)?)?;

    // ---------- Deep Learning Formulas ----------
    m.add_function(wrap_pyfunction!(dl_formulas::relu, m)?)?;
    m.add_function(wrap_pyfunction!(dl_formulas::relu_derivative, m)?)?;
    m.add_function(wrap_pyfunction!(dl_formulas::leaky_relu, m)?)?;
    m.add_function(wrap_pyfunction!(dl_formulas::leaky_relu_derivative, m)?)?;
    m.add_function(wrap_pyfunction!(dl_formulas::elu, m)?)?;
    m.add_function(wrap_pyfunction!(dl_formulas::elu_derivative, m)?)?;
    m.add_function(wrap_pyfunction!(dl_formulas::gelu, m)?)?;
    m.add_function(wrap_pyfunction!(dl_formulas::silu, m)?)?;
    m.add_function(wrap_pyfunction!(dl_formulas::silu_derivative, m)?)?;
    m.add_function(wrap_pyfunction!(dl_formulas::sigmoid, m)?)?;
    m.add_function(wrap_pyfunction!(dl_formulas::sigmoid_derivative, m)?)?;
    m.add_function(wrap_pyfunction!(dl_formulas::tanh, m)?)?;
    m.add_function(wrap_pyfunction!(dl_formulas::tanh_derivative, m)?)?;
    m.add_function(wrap_pyfunction!(dl_formulas::softmax, m)?)?;
    m.add_function(wrap_pyfunction!(dl_formulas::softmax_cross_entropy, m)?)?;
    m.add_function(wrap_pyfunction!(dl_formulas::one_hot, m)?)?;
    m.add_function(wrap_pyfunction!(dl_formulas::mse_loss, m)?)?;
    m.add_function(wrap_pyfunction!(dl_formulas::categorical_cross_entropy, m)?)?;
    m.add_function(wrap_pyfunction!(dl_formulas::mae_loss, m)?)?;
    m.add_function(wrap_pyfunction!(dl_formulas::bce_loss, m)?)?;
    m.add_function(wrap_pyfunction!(dl_formulas::sgd_momentum_step, m)?)?;
    m.add_function(wrap_pyfunction!(dl_formulas::adam_step, m)?)?;
    m.add_function(wrap_pyfunction!(dl_formulas::clip_grad_norm, m)?)?;
    m.add_function(wrap_pyfunction!(dl_formulas::dropout, m)?)?;

    // ---------- Native Structures ----------
    m.add_class::<tensor::Tensor>()?;
    m.add_class::<dsa::FarmHashMap>()?;

    Ok(())
}
