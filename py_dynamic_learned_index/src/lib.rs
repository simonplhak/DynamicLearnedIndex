use dynamic_learned_index::{
    index::{LevelIndexConfig, SearchParams, SearchStatistics},
    model::{ModelConfig, ModelLayer, TrainParams},
    IndexConfig, ModelDevice,
};
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::{prelude::*, types::PyDict, PyErr};

#[pyclass]
#[derive(Clone)]
struct DynamicLearnedIndexBuilder {
    builder: dynamic_learned_index::IndexConfig,
    threshold_samples: Option<usize>,
    bucket_size: Option<usize>,
    layers: Vec<ModelLayer>,
    batch_size: Option<usize>,
    epochs: Option<usize>,
}

#[pymethods]
impl DynamicLearnedIndexBuilder {
    #[new]
    fn new() -> PyResult<Self> {
        let builder = dynamic_learned_index::IndexConfig::default();
        Ok(DynamicLearnedIndexBuilder {
            builder,
            threshold_samples: None,
            bucket_size: None,
            layers: Vec::new(),
            batch_size: None,
            epochs: None,
        })
    }

    #[staticmethod]
    fn from_yaml(file: &str) -> PyResult<Self> {
        Ok(DynamicLearnedIndexBuilder {
            builder: IndexConfig::from_yaml(file)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?,
            threshold_samples: None,
            bucket_size: None,
            layers: Vec::new(),
            batch_size: None,
            epochs: None,
        })
    }

    fn buffer_size(&self, size: usize) -> PyResult<Self> {
        let mut builder = self.clone();
        builder.builder.buffer_size = size;
        Ok(builder)
    }

    fn bucket_size(&self, size: usize) -> PyResult<Self> {
        let mut builder = self.clone();
        builder.bucket_size = Some(size);
        Ok(builder)
    }

    fn arity(&self, arity: usize) -> PyResult<Self> {
        let mut builder = self.clone();
        builder.builder.arity = arity;
        Ok(builder)
    }

    fn compaction_strategy(&self, compaction: &str) -> PyResult<Self> {
        let mut builder = self.clone();
        match compaction {
            "bentley_saxe" => {
                builder.builder.compaction_strategy =
                    dynamic_learned_index::CompactionStrategy::BentleySaxe;
            }
            _ => {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    "Invalid compaction strategy",
                ))
            }
        }
        Ok(builder)
    }

    fn distance_fn(&self, distance_fn: &str) -> PyResult<Self> {
        let mut builder = self.clone();
        match distance_fn {
            "dot" => {
                builder.builder.distance_fn = dynamic_learned_index::DistanceFn::Dot;
            }
            "l2" => {
                builder.builder.distance_fn = dynamic_learned_index::DistanceFn::L2;
            }
            _ => {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    "Invalid distance function",
                ))
            }
        }
        Ok(builder)
    }

    fn threshold_samples(&self, samples: usize) -> PyResult<Self> {
        let mut builder = self.clone();
        builder.threshold_samples = Some(samples);
        Ok(builder)
    }

    fn linear_model_layer(&self, hidden_neurons: usize) -> PyResult<Self> {
        let mut builder = self.clone();
        builder.layers.push(ModelLayer::Linear(hidden_neurons));
        Ok(builder)
    }

    fn relu_layer(&self) -> PyResult<Self> {
        let mut builder = self.clone();
        builder.layers.push(ModelLayer::ReLU);
        Ok(builder)
    }

    fn batch_size(&self, size: usize) -> PyResult<Self> {
        let mut builder = self.clone();
        builder.batch_size = Some(size);
        Ok(builder)
    }

    fn epochs(&self, epochs: usize) -> PyResult<Self> {
        let mut builder = self.clone();
        builder.epochs = Some(epochs);
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

    fn _build_levels_config(&self) -> PyResult<Self> {
        let mut builder = self.clone();
        let mut train_params = TrainParams::default();
        if let Some(threshold) = self.threshold_samples {
            train_params.threshold_samples = threshold;
        }
        if let Some(batch_size) = self.batch_size {
            train_params.batch_size = batch_size;
        }
        if let Some(epochs) = self.epochs {
            train_params.epochs = epochs;
        }
        // todo retrain params
        let model_config = ModelConfig {
            layers: builder.layers.clone(),
            train_params,
            retrain_params: Default::default(),
        };
        let level_config = LevelIndexConfig {
            model: model_config,
            bucket_size: self.bucket_size.ok_or_else(|| {
                PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("bucket_size not set")
            })?,
        };
        builder.builder.levels.insert(0, level_config);
        Ok(builder)
    }

    fn build(&self) -> PyResult<DynamicLearnedIndex> {
        let builder = self._build_levels_config()?;
        let index = builder
            .builder
            .build()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
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

#[pyclass]
struct PySearchStatistics {
    #[pyo3(get)]
    total_visited_buckets: usize,
    #[pyo3(get)]
    total_visited_records: usize,
}

impl From<SearchStatistics> for PySearchStatistics {
    fn from(stats: SearchStatistics) -> Self {
        PySearchStatistics {
            total_visited_buckets: stats.total_visited_buckets,
            total_visited_records: stats.total_visited_records,
        }
    }
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

    #[pyo3(signature = (query, k, **py_kwargs))]
    fn verbose_search<'py>(
        &self,
        py: Python<'py>,
        query: PyReadonlyArray1<'py, f32>,
        k: usize,
        py_kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<(Bound<'py, PyArray1<u32>>, PySearchStatistics)> {
        let query = array2vec(query);
        let search_params = parse_search_kwargs(py_kwargs, k)?;
        let (r, stats) = self.index.verbose_search(&query, search_params);
        let x = r.into_pyarray_bound(py);
        Ok((x, stats.into()))
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
        let r = self.index.search(&query, search_params);
        let x = r.into_pyarray_bound(py);
        Ok(x)
    }

    fn insert<'py>(&mut self, record: PyReadonlyArray1<'py, f32>, id: u32) {
        let record = array2vec(record);
        self.index.insert(record, id);
    }

    fn n_buckets(&self) -> usize {
        self.index.n_buckets()
    }

    fn n_levels(&self) -> usize {
        self.index.n_levels()
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
