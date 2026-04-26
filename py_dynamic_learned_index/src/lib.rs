use std::io::Write;
use std::path::Path;

use dynamic_learned_index::{
    model::RetrainStrategy, IndexBuilder, ModelDevice, ModelLayer, SearchParams,
};
use half::f16;
use numpy::{IntoPyArray, PyArray1, PyArrayDyn, PyArrayMethods, PyReadonlyArray1};
use pyo3::{
    prelude::*,
    types::{PyBytes, PyDict},
    PyErr,
};
use structured_logger::json::new_writer;

/// Helper function to apply configuration settings to an IndexBuilder
fn apply_config<T>(
    mut builder: IndexBuilder<T>,
    config: &Bound<'_, PyDict>,
) -> PyResult<IndexBuilder<T>>
where
    T: dynamic_learned_index::structs::FloatElement,
{
    if let Some(v) = config.get_item("buffer_size").ok().flatten() {
        if let Ok(size) = v.extract::<usize>() {
            builder = builder.buffer_size(size);
        }
    }
    if let Some(v) = config.get_item("bucket_size").ok().flatten() {
        if let Ok(size) = v.extract::<usize>() {
            builder = builder.bucket_size(size);
        }
    }
    if let Some(v) = config.get_item("arity").ok().flatten() {
        if let Ok(arity) = v.extract::<usize>() {
            builder = builder.arity(arity);
        }
    }
    if let Some(v) = config.get_item("compaction_strategy").ok().flatten() {
        if let Ok(s) = v.extract::<String>() {
            builder = builder.compaction_strategy(s.as_str().into());
        }
    }
    if let Some(v) = config.get_item("distance_fn").ok().flatten() {
        if let Ok(s) = v.extract::<String>() {
            builder = builder.distance_fn(s.as_str().into());
        }
    }
    if let Some(v) = config.get_item("train_threshold_samples").ok().flatten() {
        if let Ok(samples) = v.extract::<usize>() {
            builder = builder.train_threshold_samples(samples);
        }
    }
    if let Some(v) = config.get_item("train_batch_size").ok().flatten() {
        if let Ok(size) = v.extract::<usize>() {
            builder = builder.train_batch_size(size);
        }
    }
    if let Some(v) = config.get_item("train_epochs").ok().flatten() {
        if let Ok(epochs) = v.extract::<usize>() {
            builder = builder.train_epochs(epochs);
        }
    }
    if let Some(v) = config.get_item("retrain_strategy").ok().flatten() {
        if let Ok(s) = v.extract::<String>() {
            let retrain_strategy = match s.as_str() {
                "no_retrain" => RetrainStrategy::NoRetrain,
                "from_scratch" => RetrainStrategy::FromScratch,
                _ => {
                    return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                        "Invalid retrain strategy",
                    ))
                }
            };
            builder = builder.retrain_strategy(retrain_strategy);
        }
    }
    if let Some(v) = config.get_item("input_shape").ok().flatten() {
        if let Ok(shape) = v.extract::<usize>() {
            builder = builder.input_shape(shape);
        }
    }
    if let Some(v) = config.get_item("device").ok().flatten() {
        if let Ok(s) = v.extract::<String>() {
            let device = if s == "cpu" {
                ModelDevice::Cpu
            } else if s.starts_with("gpu:") {
                let parts: Vec<&str> = s.split(':').collect();
                if parts.len() == 2 {
                    if let Ok(index) = parts[1].parse::<usize>() {
                        ModelDevice::Gpu(index)
                    } else {
                        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                            "Invalid device type",
                        ));
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
            builder = builder.device(device);
        }
    }
    if let Some(v) = config.get_item("delete_method").ok().flatten() {
        if let Ok(s) = v.extract::<String>() {
            builder = builder.delete_method(s.as_str().into());
        }
    }
    if let Some(v) = config.get_item("layers").ok().flatten() {
        if let Ok(layers) = v.extract::<Vec<Bound<'_, PyDict>>>() {
            for layer_dict in layers {
                if let Some(layer_type) = layer_dict.get_item("type").ok().flatten() {
                    if let Ok(layer_type_str) = layer_type.extract::<String>() {
                        match layer_type_str.as_str() {
                            "linear" => {
                                if let Some(hidden) =
                                    layer_dict.get_item("hidden_neurons").ok().flatten()
                                {
                                    if let Ok(hidden_neurons) = hidden.extract::<usize>() {
                                        builder =
                                            builder.add_layer(ModelLayer::Linear(hidden_neurons));
                                    }
                                }
                            }
                            "relu" => {
                                builder = builder.add_layer(ModelLayer::ReLU);
                            }
                            _ => {} // ignore unknown
                        }
                    }
                }
            }
        }
    }
    if let Some(quantize) = config.get_item("quantize").ok().flatten() {
        if let Ok(quantize) = quantize.extract::<bool>() {
            builder = builder.quantize(quantize);
        }
    }
    if let Some(v) = config.get_item("seed").ok().flatten() {
        if let Ok(seed) = v.extract::<u64>() {
            builder = builder.seed(seed);
        }
    }
    Ok(builder)
}

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

#[pyclass]
struct _DynamicIndexBuilderF16 {
    builder: IndexBuilder<f16>,
}

#[pymethods]
impl _DynamicIndexBuilderF16 {
    #[new]
    fn new() -> PyResult<Self> {
        let builder = IndexBuilder::<f16>::default();
        Ok(_DynamicIndexBuilderF16 { builder })
    }

    #[staticmethod]
    fn from_yaml(file: &str) -> PyResult<Self> {
        Ok(_DynamicIndexBuilderF16 {
            builder: IndexBuilder::from_yaml(Path::new(file))?,
        })
    }

    #[staticmethod]
    fn from_disk(working_dir: &str) -> PyResult<Self> {
        Ok(_DynamicIndexBuilderF16 {
            builder: IndexBuilder::from_disk(Path::new(working_dir))?,
        })
    }

    #[staticmethod]
    fn from_config(config: &Bound<'_, PyDict>) -> PyResult<Self> {
        let builder = IndexBuilder::<f16>::default();
        let builder = apply_config(builder, config)?;
        Ok(_DynamicIndexBuilderF16 { builder })
    }

    fn build(&self) -> PyResult<_DynamicLearnedIndexF16> {
        let index = self.builder.clone().build()?;
        Ok(_DynamicLearnedIndexF16 { index })
    }
}

#[pyclass]
struct _DynamicLearnedIndexF16 {
    index: dynamic_learned_index::Index<f16>,
}

fn parse_search_kwargs(py_kwargs: Option<&Bound<'_, PyDict>>, k: usize) -> PyResult<SearchParams> {
    match py_kwargs {
        Some(kwargs) => {
            let n_candidates = kwargs
                .iter()
                .find(|(key, _)| {
                    if let Ok(k) = key.extract::<String>() {
                        k == "n_candidates"
                    } else {
                        false
                    }
                })
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
                .find(|(key, _)| {
                    if let Ok(k) = key.extract::<String>() {
                        k == "search_strategy"
                    } else {
                        false
                    }
                })
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
impl _DynamicLearnedIndexF16 {
    #[new]
    fn new() -> PyResult<Self> {
        let index = IndexBuilder::default().build()?;
        Ok(_DynamicLearnedIndexF16 { index })
    }

    #[pyo3(signature = (query, k, **py_kwargs))]
    fn search<'py>(
        &self,
        py: Python<'py>,
        query: Bound<'py, PyArrayDyn<f16>>,
        k: usize,
        py_kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Bound<'py, PyArray1<u32>>> {
        let query = array2vec(query);
        let search_params = parse_search_kwargs(py_kwargs, k)?;
        let result = py.detach(|| self.index.search(&query, search_params));
        Ok(result?.into_pyarray(py))
    }

    fn insert<'py>(
        &mut self,
        py: Python<'_>,
        record: Bound<'py, PyArrayDyn<f16>>,
        id: u32,
    ) -> PyResult<()> {
        let record = array2vec(record);
        py.detach(|| {
            self.index
                .insert(record, id)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))
        })
    }

    fn delete(&mut self, id: u32) -> PyResult<Option<(Vec<u16>, u32)>> {
        let deleted = self
            .index
            .delete(id)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        if let Some((vec, id)) = deleted {
            let vec_u16: Vec<u16> = f16tou16_vec(vec);
            Ok(Some((vec_u16, id)))
        } else {
            Ok(None)
        }
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

fn array2vec<'py>(x: Bound<'py, PyArrayDyn<f16>>) -> Vec<half::f16> {
    let view = x.readonly();
    if let Ok(slice) = view.as_slice() {
        return slice.to_vec();
    }
    // Fallback for non-contiguous arrays
    let array = view.as_array();
    array.iter().cloned().collect()
}

fn array2vec_f32<'py>(x: PyReadonlyArray1<'py, f32>) -> Vec<f32> {
    x.as_array().iter().copied().collect()
}

fn f16tou16_vec(x: Vec<half::f16>) -> Vec<u16> {
    let mut x = std::mem::ManuallyDrop::new(x);
    unsafe { Vec::from_raw_parts(x.as_mut_ptr() as *mut u16, x.len(), x.capacity()) }
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

#[pyclass]
struct _DynamicLearnedIndexBuilderF32 {
    builder: IndexBuilder<f32>,
}

#[pymethods]
impl _DynamicLearnedIndexBuilderF32 {
    #[new]
    fn new() -> PyResult<Self> {
        let builder = IndexBuilder::<f32>::default();
        Ok(_DynamicLearnedIndexBuilderF32 { builder })
    }

    #[staticmethod]
    fn from_yaml(file: &str) -> PyResult<Self> {
        Ok(_DynamicLearnedIndexBuilderF32 {
            builder: IndexBuilder::from_yaml(Path::new(file))?,
        })
    }

    #[staticmethod]
    fn from_disk(working_dir: &str) -> PyResult<Self> {
        Ok(_DynamicLearnedIndexBuilderF32 {
            builder: IndexBuilder::from_disk(Path::new(working_dir))?,
        })
    }

    #[staticmethod]
    fn from_config(config: &Bound<'_, PyDict>) -> PyResult<Self> {
        let builder = IndexBuilder::<f32>::default();
        let builder = apply_config(builder, config)?;
        Ok(_DynamicLearnedIndexBuilderF32 { builder })
    }

    fn build(&self) -> PyResult<_DynamicLearnedIndexF32> {
        let index = self.builder.clone().build()?;
        Ok(_DynamicLearnedIndexF32 { index })
    }
}

#[pyclass]
struct _DynamicLearnedIndexF32 {
    index: dynamic_learned_index::Index<f32>,
}

#[pymethods]
impl _DynamicLearnedIndexF32 {
    #[new]
    fn new() -> PyResult<Self> {
        let index = IndexBuilder::default().build()?;
        Ok(_DynamicLearnedIndexF32 { index })
    }

    #[pyo3(signature = (query, k, **py_kwargs))]
    fn search<'py>(
        &self,
        py: Python<'py>,
        query: PyReadonlyArray1<'py, f32>,
        k: usize,
        py_kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Bound<'py, PyArray1<u32>>> {
        let query = array2vec_f32(query);
        let search_params = parse_search_kwargs(py_kwargs, k)?;
        let result = py.detach(|| self.index.search(&query, search_params));
        Ok(result?.into_pyarray(py))
    }

    fn insert<'py>(
        &mut self,
        py: Python<'_>,
        record: PyReadonlyArray1<'py, f32>,
        id: u32,
    ) -> PyResult<()> {
        let record = array2vec_f32(record);
        py.detach(|| {
            self.index
                .insert(record, id)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))
        })
    }

    fn delete(&mut self, id: u32) -> PyResult<Option<(Vec<f32>, u32)>> {
        let deleted = self
            .index
            .delete(id)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        if let Some((vec, id)) = deleted {
            Ok(Some((vec, id)))
        } else {
            Ok(None)
        }
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

#[pymodule(crate = "pyo3")]
fn _pydli(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add_class::<_DynamicLearnedIndexF16>()?;
    m.add_class::<_DynamicIndexBuilderF16>()?;
    m.add_class::<_DynamicLearnedIndexF32>()?;
    m.add_class::<_DynamicLearnedIndexBuilderF32>()?;
    m.add_function(wrap_pyfunction!(log_init, m)?)?;
    Ok(())
}
