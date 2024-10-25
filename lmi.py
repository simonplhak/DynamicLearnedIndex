from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch
from faiss import Kmeans, merge_knn_results
from torch import Tensor
from torch.nn import CrossEntropyLoss, Linear, ReLU, Sequential
from torch.nn.functional import softmax
from torch.optim.adam import Adam
from torch.utils.data import DataLoader

from dynamic_bucket import DynamicBucket
from index import Index
from labeled_dataset import LabeledDataset
from utils import measure_runtime, take_sample

if TYPE_CHECKING:
    from bucket import Bucket
    from configuration import IndexConfig


class LMIIndex(Index):
    """Learned Metric Index (LMI) implementation with dynamic buckets."""

    def __init__(self, config: IndexConfig) -> None:
        super().__init__(config)

        self.buckets = {i: DynamicBucket(config.bucket_shape, config.distance.metric) for i in range(config.n_buckets)}

        # Create a model
        self.model = Sequential(
            Linear(self.config.bucket_shape[1], 512),
            ReLU(),
            Linear(512, 384),
            ReLU(),
            Linear(384, config.n_buckets),
        )

        # Model's hyperparameters
        self.epochs = 10
        self.lr = 0.001
        self.loss_fn = CrossEntropyLoss()
        self.optimizer = Adam(params=self.model.parameters(), lr=self.lr)

    @measure_runtime
    def train(self, buckets: list[Bucket]) -> None:
        sample_size = sum(b.get_n_objects() for b in buckets)  # TODO: change
        X_sample, _ = take_sample(buckets, sample_size, self.config.bucket_shape[1])

        # Run k-means to obtain training labels
        kmeans = Kmeans(
            d=self.config.bucket_shape[1],
            k=self.config.n_buckets,
            verbose=False,
            seed=42,
        )
        kmeans.train(X_sample)
        y = torch.from_numpy(kmeans.index.search(X_sample, 1)[1].T[0])  # type: ignore

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

        self._assign_objects_to_new_buckets(buckets)

    def search(self, query: Tensor, k: int, nprobe: int) -> tuple[np.ndarray, np.ndarray, int]:
        nprobe = min(nprobe, self.config.n_buckets)

        bucket_ids = self._predict(query, nprobe)[1][0]

        D_all, I_all = (
            np.zeros((nprobe, 1, k), dtype=np.float32),
            np.zeros((nprobe, 1, k), dtype=np.int64),
        )
        n_candidates = 0

        for i in range(len(bucket_ids)):
            bucket_id = int(bucket_ids[i].item())
            D_all[i, :, :], I_all[i, :, :], n_level_candidates = self.buckets[bucket_id].search(query, k, nprobe)
            n_candidates += n_level_candidates

        return *merge_knn_results(D_all, I_all, keep_max=self.config.distance.keep_max), n_candidates

    def insert(self, buckets: list[Bucket]) -> bool:
        # TODO: get rid of torch.concatenate
        X, I = torch.concatenate([b.get_data() for b in buckets]), np.concatenate([b.get_ids() for b in buckets])

        # Predict to which bucket each vector belongs
        bucket_ids = self._predict(X, 1)[1].reshape(-1)

        # Because we use dynamic bucket size, we do not check for overflowing buckets.

        # Add the vectors to the buckets
        for i, child_bucket in self.buckets.items():
            child_bucket.insert_bulk(X[bucket_ids == i], I[bucket_ids == i])

        return True

    def _assign_objects_to_new_buckets(self, buckets: list[Bucket]) -> None:
        # Add the vectors to the new buckets
        for existing_bucket in buckets:
            bucket_data = existing_bucket.get_data()
            bucket_indexes = existing_bucket.get_ids()

            # Each vector belongs to a new bucket
            classes = self._predict(bucket_data, 1)[1].reshape(-1)

            for i, new_child_bucket in self.buckets.items():
                # ! This insert overflows as k-means produces unbalanced clusters
                new_child_bucket.insert_bulk(bucket_data[classes == i], bucket_indexes[np.where(classes == i)])

    def _predict(self, X: Tensor, top_k: int) -> tuple[Tensor, Tensor]:
        assert self.model is not None, 'Model is not trained yet.'

        # Evaluate the model
        self.model.eval()

        with torch.no_grad():
            logits = self.model(X)
            # Compute probabilities from logits
            probs = softmax(logits, dim=1)
            # Select the top k most probable classes and their probabilities
            probabilities, classes = probs.topk(top_k)

        return probabilities, classes
