use pyo3::prelude::*;

#[pyfunction]
fn add(left: u64, right: u64) -> PyResult<u64> {
    Ok(dynamic_learned_index::add(left, right))
}

#[pymodule]
fn py_dynamic_learned_index(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add_function(wrap_pyfunction!(add, m)?)?;
    Ok(())
}
