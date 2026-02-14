use std::io::Write;
use std::path::Path;

use dynamic_learned_index::{
    model::RetrainStrategy, IndexBuilder, ModelDevice, ModelLayer, SearchParams,
};
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::{
    prelude::*,
    types::{PyBytes, PyDict},
    PyErr,
};
use structured_logger::json::new_writer;

struct PyFileLike {
    inner: Py<PyAny>,
}

impl Write for PyFileLike {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        Python::attach(|py| {
            let obj = self.inner.bind(py);
            if let Ok(s) = std::str::from_utf8(buf) {
                // Try writing as string first
                if obj.call_method1("write", (s,)).is_ok() {
                    return Ok(buf.len());
                }
            }
            // Fallback to bytes
            let bytes = PyBytes::new(py, buf);
            obj.call_method1("write", (bytes,))
                .map_err(|e| std::io::Error::other(e.to_string()))?;
            Ok(buf.len())
        })
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Python::attach(|py| {
            let obj = self.inner.bind(py);
            let _ = obj.call_method0("flush");
            Ok(())
        })
    }
}

// Py<PyAny> is Send. Access is guarded by GIL so it is Sync.
unsafe impl Sync for PyFileLike {}

#[pyclass]
#[derive(Clone)]
struct DynamicLearnedIndexBuilder {
    builder: IndexBuilder,
}

#[pymethods]
impl DynamicLearnedIndexBuilder {
    #[new]
    fn new() -> PyResult<Self> {
        let builder = IndexBuilder::default();
        Ok(DynamicLearnedIndexBuilder { builder })
    }

    #[staticmethod]
    fn from_yaml(file: &str) -> PyResult<Self> {
        Ok(DynamicLearnedIndexBuilder {
            builder: IndexBuilder::from_yaml(Path::new(file))?,
        })
    }

    #[staticmethod]
    fn from_disk(working_dir: &str) -> PyResult<Self> {
        Ok(DynamicLearnedIndexBuilder {
            builder: IndexBuilder::from_disk(Path::new(working_dir))?,
        })
    }

    fn buffer_size(&self, size: usize) -> PyResult<Self> {
        let mut builder = self.clone();
        builder.builder = builder.builder.buffer_size(size);
        Ok(builder)
    }

    fn bucket_size(&self, size: usize) -> PyResult<Self> {
        let mut builder = self.clone();
        builder.builder = builder.builder.bucket_size(size);
        Ok(builder)
    }

    fn arity(&self, arity: usize) -> PyResult<Self> {
        let mut builder = self.clone();
        builder.builder = builder.builder.arity(arity);
        Ok(builder)
    }

    fn compaction_strategy(&self, compaction: &str) -> PyResult<Self> {
        let mut builder = self.clone();
        builder.builder = builder.builder.compaction_strategy(compaction.into());
        Ok(builder)
    }

    fn distance_fn(&self, distance_fn: &str) -> PyResult<Self> {
        let mut builder = self.clone();
        builder.builder = builder.builder.distance_fn(distance_fn.into());
        Ok(builder)
    }

    fn train_threshold_samples(&self, samples: usize) -> PyResult<Self> {
        let mut builder = self.clone();
        builder.builder = builder.builder.train_threshold_samples(samples);
        Ok(builder)
    }

    fn linear_model_layer(&self, hidden_neurons: usize) -> PyResult<Self> {
        let mut builder = self.clone();
        builder.builder = builder
            .builder
            .add_layer(ModelLayer::Linear(hidden_neurons));
        Ok(builder)
    }

    fn relu_layer(&self) -> PyResult<Self> {
        let mut builder = self.clone();
        builder.builder = builder.builder.add_layer(ModelLayer::ReLU);
        Ok(builder)
    }

    fn train_batch_size(&self, size: usize) -> PyResult<Self> {
        let mut builder = self.clone();
        builder.builder = builder.builder.train_batch_size(size);
        Ok(builder)
    }

    fn train_epochs(&self, epochs: usize) -> PyResult<Self> {
        let mut builder = self.clone();
        builder.builder = builder.builder.train_epochs(epochs);
        Ok(builder)
    }

    fn retrain_strategy(&self, strategy: &str) -> PyResult<Self> {
        let mut builder = self.clone();
        let retrain_strategy = match strategy {
            "no_retrain" => RetrainStrategy::NoRetrain,
            "from_scratch" => RetrainStrategy::FromScratch,
            _ => {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    "Invalid retrain strategy",
                ))
            }
        };
        builder.builder = builder.builder.retrain_strategy(retrain_strategy);
        Ok(builder)
    }

    fn input_shape(&self, shape: usize) -> PyResult<Self> {
        let mut builder = self.clone();
        builder.builder = builder.builder.input_shape(shape);
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
        builder.builder = builder.builder.device(device);
        Ok(builder)
    }

    fn delete_method(&self, delete_method: &str) -> PyResult<Self> {
        let mut builder = self.clone();
        builder.builder = builder.builder.delete_method(delete_method.into());
        Ok(builder)
    }

    fn build(&self) -> PyResult<DynamicLearnedIndex> {
        let index = self.builder.clone().build()?;
        Ok(DynamicLearnedIndex { index })
    }
}

#[pyclass]
struct DynamicLearnedIndex {
    index: dynamic_learned_index::Index,
}

fn parse_search_kwargs(py_kwargs: Option<&Bound<'_, PyDict>>, k: usize) -> PyResult<SearchParams> {
    match py_kwargs {
        Some(kwargs) => {
            let n_candidates = kwargs
                .iter()
                .find(|(key, _)| key.extract::<String>().unwrap_or_default() == "n_candidates") // todo remove unwrap
                .map(|(_, value)| {
                    value.extract::<usize>().map_err(|_| {
                        PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                            "n_candidates must be an integer",
                        )
                    })
                })
                .unwrap_or(Ok(1))?;
            let search_strategy = kwargs
                .iter()
                .find(|(key, _)| key.extract::<String>().unwrap_or_default() == "search_strategy")
                .map(|(_, value)| match value.extract::<String>() {
                    Ok(strategy) => match strategy.as_str() {
                        "knn" => Ok(dynamic_learned_index::SearchStrategy::Base(n_candidates)),
                        "model" => Ok(dynamic_learned_index::SearchStrategy::ModelDriven(
                            n_candidates,
                        )),
                        _ => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                            "Invalid search strategy",
                        )),
                    },
                    Err(_) => Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                        "search_strategy must be a string",
                    )),
                })
                .unwrap_or(Ok(Default::default()))?;
            Ok(SearchParams { search_strategy, k })
        }

        None => Ok(SearchParams {
            search_strategy: Default::default(),
            k,
        }),
    }
}

#[pymethods]
impl DynamicLearnedIndex {
    #[new]
    fn new() -> PyResult<Self> {
        let index = IndexBuilder::default().build()?;
        Ok(DynamicLearnedIndex { index })
    }

    #[pyo3(signature = (query, k, **py_kwargs))]
    fn search<'py>(
        &self,
        py: Python<'py>,
        query: PyReadonlyArray1<'py, f32>,
        k: usize,
        py_kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Bound<'py, PyArray1<u32>>> {
        let query = array2vec(query);
        let search_params = parse_search_kwargs(py_kwargs, k)?;
        let r = self.index.search(&query, search_params)?;
        let x = r.into_pyarray(py);
        Ok(x)
    }

    fn insert<'py>(&mut self, record: PyReadonlyArray1<'py, f32>, id: u32) -> PyResult<()> {
        let record = array2vec(record);
        self.index.insert(record, id)?;
        Ok(())
    }

    fn delete(&mut self, id: u32) -> PyResult<Option<(Vec<f32>, u32)>> {
        Ok(self.index.delete(id)?)
    }

    fn n_buckets(&self) -> usize {
        self.index.n_buckets()
    }

    fn n_levels(&self) -> usize {
        self.index.n_levels()
    }

    fn occupied(&self) -> usize {
        self.index.occupied()
    }

    fn n_empty_buckets(&self) -> usize {
        self.index.n_empty_buckets()
    }

    fn dump(&self, working_dir: &str) -> PyResult<()> {
        self.index.dump(Path::new(working_dir))?;
        Ok(())
    }

    fn buffer_occupied(&self) -> usize {
        self.index.buffer_occupied()
    }

    fn level_occupied(&self, level_idx: usize) -> usize {
        self.index.level_occupied(level_idx)
    }

    fn level_n_buckets(&self, level_idx: usize) -> usize {
        self.index.level_n_buckets(level_idx)
    }

    fn level_total_size(&self, level_idx: usize) -> usize {
        self.index.level_total_size(level_idx)
    }

    fn level_n_empty_buckets(&self, level_idx: usize) -> usize {
        self.index.level_n_empty_buckets(level_idx)
    }

    fn bucket_occupied(&self, level_idx: usize, bucket_idx: usize) -> usize {
        self.index.bucket_occupied(level_idx, bucket_idx)
    }

    fn memory_usage(&self) -> usize {
        self.index.memory_usage()
    }
}

fn array2vec<'py>(x: PyReadonlyArray1<'py, f32>) -> Vec<f32> {
    x.as_array().iter().copied().collect()
}

#[pyfunction]
fn log_init(target: &Bound<'_, PyAny>, level: &str) -> PyResult<()> {
    if let Ok(file_path) = target.extract::<String>() {
        let file = std::fs::File::create(file_path)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(e.to_string()))?;
        if let Err(e) = structured_logger::Builder::with_level(level)
            .with_target_writer("*", new_writer(file))
            .try_init()
        {
            eprintln!("Warning: Logger already initialized: {}", e);
        }
    } else {
        // Assume it's a file-like object
        let writer = PyFileLike {
            inner: target.clone().unbind(),
        };
        if let Err(e) = structured_logger::Builder::with_level(level)
            .with_target_writer("*", new_writer(writer))
            .try_init()
        {
            eprintln!("Warning: Logger already initialized: {}", e);
        }
    }
    Ok(())
}

#[pymodule(crate = "pyo3")]
fn py_dynamic_learned_index(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add_class::<DynamicLearnedIndex>()?;
    m.add_class::<DynamicLearnedIndexBuilder>()?;
    m.add_function(wrap_pyfunction!(log_init, m)?)?;
    Ok(())
}
