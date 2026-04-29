from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Literal

import py_dynamic_learned_index._pydli as _pydli

if TYPE_CHECKING:
    import numpy as np

log_init = _pydli.log_init

class DynamicLearnedIndexBuilder:
    """Builder for creating Dynamic Learned Index instances with fluent configuration.

    Example:
        index = (
            DynamicLearnedIndexBuilder()
            .bucket_size(100)
            .arity(2)
            .add_linear_layer(64)
            .add_relu_layer()
            .build()
        )
    """

    def __init__(self, dtype: Literal["f16", "f32"] = "f16"):
        """Initialize the builder.

        Args:
            dtype: Precision type, 'f16' or 'f32'.
        """
        self.dtype = dtype
        self.config: Dict[str, Any] = {}
        self.layers: List[Dict[str, Any]] = []

    def buffer_size(self, size: int) -> 'DynamicLearnedIndexBuilder':
        """Set the buffer size."""
        self.config["buffer_size"] = size
        return self

    def bucket_size(self, size: int) -> 'DynamicLearnedIndexBuilder':
        """Set the bucket size."""
        self.config["bucket_size"] = size
        return self

    def arity(self, arity: int) -> 'DynamicLearnedIndexBuilder':
        """Set the arity."""
        self.config["arity"] = arity
        return self

    def compaction_strategy(self, strategy: str) -> 'DynamicLearnedIndexBuilder':
        """Set the compaction strategy."""
        self.config["compaction_strategy"] = strategy
        return self

    def distance_fn(self, fn: str) -> 'DynamicLearnedIndexBuilder':
        """Set the distance function."""
        self.config["distance_fn"] = fn
        return self

    def train_threshold_samples(self, samples: int) -> 'DynamicLearnedIndexBuilder':
        """Set the train threshold samples."""
        self.config["train_threshold_samples"] = samples
        return self

    def train_batch_size(self, size: int) -> 'DynamicLearnedIndexBuilder':
        """Set the train batch size."""
        self.config["train_batch_size"] = size
        return self

    def train_epochs(self, epochs: int) -> 'DynamicLearnedIndexBuilder':
        """Set the train epochs."""
        self.config["train_epochs"] = epochs
        return self

    def retrain_strategy(self, strategy: str) -> 'DynamicLearnedIndexBuilder':
        """Set the retrain strategy."""
        self.config["retrain_strategy"] = strategy
        return self

    def input_shape(self, shape: int) -> 'DynamicLearnedIndexBuilder':
        """Set the input shape."""
        self.config["input_shape"] = shape
        return self

    def device(self, device: str) -> 'DynamicLearnedIndexBuilder':
        """Set the device (cpu or gpu:X)."""
        self.config["device"] = device
        return self

    def delete_method(self, method: str) -> 'DynamicLearnedIndexBuilder':
        """Set the delete method."""
        self.config["delete_method"] = method
        return self

    def add_linear_layer(self, hidden_neurons: int) -> 'DynamicLearnedIndexBuilder':
        """Add a linear layer."""
        self.layers.append({"type": "linear", "hidden_neurons": hidden_neurons})
        return self

    def add_relu_layer(self) -> 'DynamicLearnedIndexBuilder':
        """Add a ReLU layer."""
        self.layers.append({"type": "relu"})
        return self

    def quantize(self, quantize: bool) -> 'DynamicLearnedIndexBuilder':
        """Set whether to quantize the model."""
        self.config["quantize"] = quantize
        return self

    def seed(self, seed: int) -> 'DynamicLearnedIndexBuilder':
        self.config["seed"] = seed
        return self

    def build(self) -> 'DynamicLearnedIndex':
        """Build the index."""
        self.config["layers"] = self.layers
        if self.dtype == "f16":
            builder = _pydli._DynamicIndexBuilderF16.from_config(self.config)
        elif self.dtype == "f32":
            builder = _pydli._DynamicLearnedIndexBuilderF32.from_config(self.config)
        else:
            raise ValueError(f"Invalid dtype: {self.dtype}")
        index = builder.build()
        return DynamicLearnedIndex(index, self.dtype)

    @classmethod
    def from_yaml(
        cls, path: str, dtype: Literal["f16", "f32"] = "f16"
    ) -> 'DynamicLearnedIndexBuilder':
        """Load configuration from YAML file."""
        if dtype == "f16":
            return _pydli._DynamicIndexBuilderF16.from_config(path)
        elif dtype == "f32":
            return _pydli._DynamicLearnedIndexBuilderF32.from_config(path)
        raise ValueError(f"Invalid dtype: {dtype}")

    @classmethod
    def from_disk(
        cls, working_dir: str, dtype: Literal["f16", "f32"] = "f16"
    ) -> 'DynamicLearnedIndex':
        """Load a saved index from disk."""
        if dtype == "f16":
            builder = _pydli._DynamicIndexBuilderF16.from_disk(working_dir)
        elif dtype == "f32":
            builder = _pydli._DynamicLearnedIndexBuilderF32.from_disk(working_dir)
        else:
            raise ValueError(f"Invalid dtype: {dtype}")
        index = builder.build()
        return DynamicLearnedIndex(index, dtype)


class DynamicLearnedIndex:
    """Dynamic Learned Index for efficient similarity search.

    Supports insertion, deletion, and search operations.
    """

    def __init__(self, index, dtype: str):
        self._index = index
        self.dtype = dtype

    def search(
        self,
        query: np.ndarray,
        k: int,
        n_candidates: int = 100,
        search_strategy: str = "knn",
    ) -> List[int]:
        """Search for k nearest neighbors."""
        return self._index.search(
            query, k, n_candidates=n_candidates, search_strategy=search_strategy
        )

    def insert(self, query: np.ndarray, id: int) -> None:
        """Insert a vector with given id."""
        self._index.insert(query, id)

    def delete(self, id: int):
        """Delete the vector with given id."""
        return self._index.delete(id)

    def n_buckets(self) -> int:
        """Get total number of buckets."""
        return self._index.n_buckets()

    def n_levels(self) -> int:
        """Get number of levels."""
        return self._index.n_levels()

    def occupied(self) -> int:
        """Get total occupied slots."""
        return self._index.occupied()

    def n_empty_buckets(self) -> int:
        """Get number of empty buckets."""
        return self._index.n_empty_buckets()

    def dump(self, path: str) -> None:
        """Save the index to disk."""
        self._index.dump(path)

    def buffer_occupied(self) -> int:
        """Get buffer occupied slots."""
        return self._index.buffer_occupied()

    def level_occupied(self, level_idx: int) -> int:
        """Get occupied slots in level."""
        return self._index.level_occupied(level_idx)

    def level_n_buckets(self, level_idx: int) -> int:
        """Get number of buckets in level."""
        return self._index.level_n_buckets(level_idx)

    def level_total_size(self, level_idx: int) -> int:
        """Get total size of level."""
        return self._index.level_total_size(level_idx)

    def level_n_empty_buckets(self, level_idx: int) -> int:
        """Get number of empty buckets in level."""
        return self._index.level_n_empty_buckets(level_idx)

    def bucket_occupied(self, level_idx: int, bucket_idx: int) -> int:
        """Get occupied slots in bucket."""
        return self._index.bucket_occupied(level_idx, bucket_idx)

    def memory_usage(self) -> int:
        """Get memory usage in bytes."""
        return self._index.memory_usage()
