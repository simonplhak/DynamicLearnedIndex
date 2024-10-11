from __future__ import annotations

import numpy as np
import torch
from faiss import Kmeans
from torch import Tensor
from torch.nn import CrossEntropyLoss, Linear, ReLU, Sequential
from torch.nn.functional import softmax
from torch.optim.adam import Adam
from torch.utils.data import DataLoader

from bucket import Bucket
from index import Index
from labeled_dataset import LabeledDataset
from utils import take_sample


class LMIIndex(Index):
    def __init__(self, n_buckets: int, metric: int, bucket_shape: tuple[int, int]) -> None:
        self.dimensionality: int = bucket_shape[1]
        """Dimensionality of the data."""
        self.n_buckets: int = n_buckets
        """Number of buckets."""
        self.buckets: dict[int, Bucket] = {i: Bucket(bucket_shape, metric) for i in range(n_buckets)}

        self.bucket_size = bucket_shape[0]

        # Create a model
        self.model = Sequential(
            Linear(self.dimensionality, 512),
            ReLU(),
            Linear(512, 384),
            ReLU(),
            Linear(384, n_buckets),
        )

        # Model's hyperparameters
        self.epochs = 10
        self.lr = 0.001
        self.loss_fn = CrossEntropyLoss()
        self.optimizer = Adam(params=self.model.parameters(), lr=self.lr)

    def train(self, buckets: list[Bucket]) -> None:
        sample_size = sum(b.get_n_objects() for b in buckets)  # TODO: change
        X_sample = take_sample(buckets, sample_size, self.dimensionality)

        # Run k-means to obtain training labels
        kmeans = Kmeans(
            d=self.dimensionality,
            k=self.n_buckets,
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

        # Add the vectors to the new buckets
        for existing_bucket in buckets:
            bucket_data = existing_bucket.get_data()
            bucket_indexes = existing_bucket.get_ids()

            # Each vector belongs to a new bucket
            classes = self._predict(bucket_data, 1)[1].reshape(-1)

            for i, new_child_bucket in self.buckets.items():
                # ! This insert overflows as k-means produces unbalanced clusters
                new_child_bucket.insert(bucket_data[classes == i], bucket_indexes[classes == i])

        self.is_trained = True

    def search(self, query: Tensor, k: int) -> tuple[np.ndarray, np.ndarray]:
        bucket_id = int(self._predict(query, 1)[1].item())
        return self.buckets[bucket_id].search(query, k)

    def insert(self, buckets: list[Bucket]) -> bool:
        # TODO: get rid of torch.concatenate
        X, I = torch.concatenate([b.get_data() for b in buckets]), np.concatenate([b.get_ids() for b in buckets])

        # Predict to which bucket each vector belongs
        bucket_ids = self._predict(X, 1)[1].reshape(-1)

        # Check that buckets do not overflow
        for i, n_objects_in_bucket in enumerate(torch.bincount(bucket_ids, minlength=self.n_buckets)):
            if n_objects_in_bucket + self.buckets[i].get_n_objects() > self.bucket_size:
                return False  # Overflow detected

        # Add the vectors to the buckets
        for i, child_bucket in self.buckets.items():
            child_bucket.insert(X[bucket_ids == i], I[bucket_ids == i])

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
