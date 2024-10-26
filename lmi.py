from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch
from faiss import Kmeans, merge_knn_results
from loguru import logger
from torch import Tensor
from torch.nn import CrossEntropyLoss, Linear, ReLU, Sequential
from torch.nn.functional import softmax
from torch.optim.adam import Adam
from torch.utils.data import DataLoader

from dynamic_bucket import DynamicBucket
from index import Index
from labeled_dataset import LabeledDataset
from sampling import take_sample
from utils import measure_runtime

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
        X_sample, _ = take_sample(buckets, self.config.sampling, self.config.bucket_shape[1])

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

        self.insert(buckets)

        if self.is_degenerated():
            logger.warning('Trained degenerated index!')

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
        for bucket in buckets:  # ! can be parallelized
            X, I = bucket.get_data(), bucket.get_ids()

            bucket_ids = self._predict(X, 1)[1].reshape(-1)

            for i, child_bucket in self.buckets.items():
                child_bucket.insert_bulk(X[bucket_ids == i], I[np.where(bucket_ids == i)])

        return True

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
