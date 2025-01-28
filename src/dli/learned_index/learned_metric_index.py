from __future__ import annotations

import time
from typing import TYPE_CHECKING, override

from loguru import logger
from torch import Tensor, empty, float32, int64, no_grad
from torch.nn import CrossEntropyLoss, Linear, ReLU, Sequential
from torch.nn.functional import softmax
from torch.optim.adam import Adam
from torch.utils.data import DataLoader

from dli.bucket import DynamicBucket
from dli.faiss_facade import merge_knn_results, obtain_labels_via_kmeans
from dli.labeled_dataset import LabeledDataset
from dli.learned_index.learned_index import LearnedIndex
from dli.sampling import take_sample
from dli.utils import get_model_size, measure_runtime

if TYPE_CHECKING:
    from dli.bucket import Bucket
    from dli.config import IndexConfig


class LearnedMetricIndex(LearnedIndex):
    """Learned Metric Index (LMI) implementation with dynamic buckets."""

    def __init__(self, config: IndexConfig) -> None:
        super().__init__(config)

        self.buckets = {
            i: DynamicBucket(
                config.bucket_shape,
                config.distance_function,
                config.shrink_buckets_during_compaction,
            )
            for i in range(config.n_buckets)
        }

        # Determine the number of neurons in all layers
        n_input_neurons = self.config.bucket_shape[1]
        n_output_neurons = config.n_buckets

        # Inspired by https://stats.stackexchange.com/a/136542
        alpha = 2
        n_hidden_neurons = int(self.config.n_training_samples / (alpha * (n_input_neurons + n_output_neurons)))
        n_hidden_neurons = (
            int((n_input_neurons + n_output_neurons) / 2)
            if n_hidden_neurons < min(n_input_neurons, n_output_neurons)
            else n_hidden_neurons
        )
        logger.debug(f"LMI's model architecture: {n_input_neurons}-{n_hidden_neurons}-{n_output_neurons}")

        # Create a model
        self.model = Sequential(
            Linear(n_input_neurons, n_hidden_neurons),
            ReLU(),
            Linear(n_hidden_neurons, n_output_neurons),
        )

        # Model's hyperparameters
        self.epochs = 10
        self.lr = 0.001
        self.loss_fn = CrossEntropyLoss()
        self.optimizer = Adam(params=self.model.parameters(), lr=self.lr)

    @measure_runtime
    @override
    def train(self, buckets: list[Bucket]) -> float:
        s = time.time()
        X_sample, _ = take_sample(buckets, self.config.sample_threshold, self.config.bucket_shape[1])

        # Run k-means to obtain training labels
        y = obtain_labels_via_kmeans(X_sample, self.config.n_buckets, self.config.distance_function.normalize_centroids)

        # Prepare the data loader for training
        train_loader = DataLoader(dataset=LabeledDataset(X_sample, y), batch_size=256, shuffle=True)

        # Train the model
        self.model.train()

        for _ in range(self.epochs):
            for X_batch, y_batch in train_loader:
                loss = self.loss_fn(self.model(X_batch), y_batch)

                # Do the backpropagation
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

        self.insert(buckets)

        if self.is_degenerated():
            logger.warning('Trained degenerated index!')

        return time.time() - s

    @override
    def search(self, query: Tensor, k: int, nprobe: int) -> tuple[Tensor, Tensor, int]:
        nprobe = min(nprobe, self.config.n_buckets)

        bucket_ids = self._predict(query, nprobe)[1][0]

        D_all, I_all = (
            empty((nprobe, 1, k), dtype=float32),
            empty((nprobe, 1, k), dtype=int64),
        )
        n_candidates = 0

        for i in range(len(bucket_ids)):
            bucket_id = int(bucket_ids[i].item())
            D_all[i, :, :], I_all[i, :, :], n_level_candidates = self.buckets[bucket_id].search(query, k, nprobe)
            n_candidates += n_level_candidates

        return *merge_knn_results(D_all, I_all, keep_max=self.config.distance_function.keep_max_values), n_candidates

    @override
    def insert(self, buckets: list[Bucket]) -> bool:
        for bucket in buckets:  # ! can be parallelized
            X, I = bucket.get_data(), bucket.get_ids()

            bucket_ids = self._predict(X, 1)[1].reshape(-1)

            for i, child_bucket in self.buckets.items():
                child_bucket.insert_bulk(X[bucket_ids == i], I[bucket_ids == i])

        return True

    @override
    def predict_bucket_scores(self, X: Tensor) -> list[tuple[int, float]]:
        assert self.model is not None, 'Model is not trained yet.'

        # Evaluate the model
        self.model.eval()

        with no_grad():
            logits = self.model(X)
            # Compute probabilities from logits
            probabilities = softmax(logits, dim=1)

        return [(i, probabilities[0][i].item()) for i in range(self.config.n_buckets)]

    @override
    def measure_total_allocated_memory(self) -> int:
        return self.measure_allocated_model_memory() + self.measure_allocated_bucket_memory()

    @override
    def measure_allocated_model_memory(self) -> int:
        return get_model_size(self.model)

    @override
    def measure_allocated_bucket_memory(self) -> int:
        return sum([bucket.get_allocated_memory() for bucket in self.buckets.values()])

    def _predict(self, X: Tensor, top_k: int) -> tuple[Tensor, Tensor]:
        assert self.model is not None, 'Model is not trained yet.'

        # Evaluate the model
        self.model.eval()

        with no_grad():
            logits = self.model(X)
            # Compute probabilities from logits
            probs = softmax(logits, dim=1)
            # Select the top k most probable classes and their probabilities
            probabilities, classes = probs.topk(top_k)

        return probabilities, classes
