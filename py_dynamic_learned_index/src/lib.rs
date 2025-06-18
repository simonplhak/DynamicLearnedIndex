use dynamic_learned_index::{IndexConfig, ModelDevice};
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::{prelude::*, PyErr};

#[pyclass]
#[derive(Clone)]
struct DynamicLearnedIndexBuilder {
    builder: dynamic_learned_index::IndexConfig,
}

#[pymethods]
impl DynamicLearnedIndexBuilder {
    #[new]
    fn new() -> PyResult<Self> {
        let builder = dynamic_learned_index::IndexConfig::default();
        Ok(DynamicLearnedIndexBuilder { builder })
    }

    #[staticmethod]
    fn from_yaml(file: &str) -> PyResult<Self> {
        Ok(DynamicLearnedIndexBuilder {
            builder: IndexConfig::from_yaml(file)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?,
        })
    }

    fn buffer_size(&self, size: usize) -> PyResult<Self> {
        let mut builder = self.clone();
        builder.builder.buffer_size = size;
        Ok(builder)
    }

    fn input_shape(&self, shape: usize) -> PyResult<Self> {
        let mut builder = self.clone();
        builder.builder.input_shape = shape;
        Ok(builder)
    }

    fn device(&self, device: &str) -> PyResult<Self> {
        let mut builder = self.clone();
        let device = if device == "cpu" {
            ModelDevice::Cpu
        } else if device.starts_with("gpu:") {
            let parts: Vec<&str> = device.split(':').collect();
            if parts.len() == 2 {
                let index_str = parts[1];
                match index_str.parse::<usize>() {
                    Ok(index) => ModelDevice::Gpu(index),
                    Err(_) => {
                        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                            "Invalid device type",
                        ))
                    }
                }
            } else {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    "Invalid device type",
                ));
            }
        } else {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Invalid device type",
            ));
        };
        builder.builder.device = device;
        Ok(builder)
    }

    fn build(&self) -> PyResult<DynamicLearnedIndex> {
        let index = self
            .builder
            .clone()
            .build()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        Ok(DynamicLearnedIndex { index })
    }
}

#[pyclass]
struct DynamicLearnedIndex {
    index: dynamic_learned_index::Index,
}

#[pymethods]
impl DynamicLearnedIndex {
    #[new]
    fn new() -> PyResult<Self> {
        let index = dynamic_learned_index::IndexConfig::default()
            .build()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        Ok(DynamicLearnedIndex { index })
    }

    fn search<'py>(
        &self,
        py: Python<'py>,
        query: PyReadonlyArray1<'py, f32>,
        k: usize,
    ) -> Bound<'py, PyArray1<u32>> {
        let query = array2vec(query);
        let r = self.index.search(&query, k);
        let x = r.into_pyarray_bound(py);
        x
    }

    fn insert<'py>(&mut self, record: PyReadonlyArray1<'py, f32>, id: u32) {
        let record = array2vec(record);
        self.index.insert(record, id);
    }

    fn __repr__(&self) -> String {
        format!("DynamicLearnedIndex(index={:?})", self.index)
    }
}

fn array2vec<'py>(x: PyReadonlyArray1<'py, f32>) -> Vec<f32> {
    x.as_array().iter().copied().collect()
}

#[pymodule(crate = "pyo3")]
fn py_dynamic_learned_index(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add_class::<DynamicLearnedIndex>()?;
    m.add_class::<DynamicLearnedIndexBuilder>()?;
    Ok(())
}
