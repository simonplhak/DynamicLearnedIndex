use pyo3::prelude::*;

#[pyclass(unsendable)]
struct DynamicLearnedIndex {
    index: dynamic_learned_index::Index,
}

#[pymethods]
impl DynamicLearnedIndex {
    #[new]
    fn new() -> Self {
        let index = dynamic_learned_index::IndexConfig::default()
            .build()
            .unwrap();
        DynamicLearnedIndex { index }
    }

    fn search(&self, query: Vec<f32>, k: usize) -> PyResult<Vec<u32>> {
        Ok(self.index.search(&query, k))
    }

    fn __repr__(&self) -> String {
        "DynamicLearnedIndex()".to_string()
    }

    #[staticmethod]
    fn default() -> Self {
        DynamicLearnedIndex {
            index: dynamic_learned_index::IndexConfig::default()
                .build()
                .unwrap(),
        }
    }
}

#[pymodule(crate = "pyo3")]
fn py_dynamic_learned_index(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add_class::<DynamicLearnedIndex>()?;
    Ok(())
}
