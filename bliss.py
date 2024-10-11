from __future__ import annotations

from typing import TYPE_CHECKING

import faiss
import numpy as np
import torch
from sklearn.preprocessing import MultiLabelBinarizer
from torch import Tensor
from torch.nn import BCEWithLogitsLoss, Linear, ReLU, Sequential
from torch.nn.functional import softmax
from torch.optim.adam import Adam
from torch.utils.data import DataLoader

from index import Index
from labeled_dataset import LabeledDataset
from utils import measure_runtime, np_rng, torch_rng

if TYPE_CHECKING:
    from bucket import Bucket


class BLISSIndex(Index):
    def __init__(
        self,
        n_buckets: int,
        metric: int,
        bucket_shape: tuple[int, int],
    ) -> None:
        super().__init__(
            n_buckets,
            metric,
            bucket_shape,
        )
        self.bucket_size = bucket_shape[0]
        self.sample_size: int = self.bucket_size  # TODO: think about sample_size
        self.k_training: int = 100
        self.top_k_buckets_to_load_balance_between: int = (
            n_buckets  # ? Tied to the number of buckets? In the paper it is rather a small number...
        )
        self.n_redistributions: int = 2

        self.model = Sequential(
            Linear(bucket_shape[1], 512),
            ReLU(),
            Linear(512, n_buckets),
        )
        # Model's hyperparameters
        self.epochs = 5
        self.lr = 0.1
        self.loss_fn = BCEWithLogitsLoss(reduction='sum')  # CrossEntropyLoss()
        self.optimizer = Adam(params=self.model.parameters(), lr=self.lr)

        # Feature flags
        self.do_not_shift_when_object_is_within_top_k_buckets = False
        """Do not shift an object when it is within the top K predicted buckets."""

    @measure_runtime
    def train(self, buckets: list[Bucket]) -> None:
        # TODO: get rid of torch.concatenate
        X, I = torch.concatenate([b.get_data() for b in buckets]), np.concatenate([b.get_ids() for b in buckets])

        # Take a sample from the dataset
        sample_indices = torch.randperm(len(X), generator=torch_rng)[: self.sample_size]
        sample = X[sample_indices]

        # Randomly distribute the dataset into buckets
        bucket_assignment = np_rng.choice(self.n_buckets, len(X))

        # Calculate kNN ground truth for the sample on the sample
        ground_truth = self._prepare_ground_truth(sample, self.k_training)

        for _ in range(self.n_redistributions):
            # Select the bucket assignment of the sample for training
            bucket_assignment_for_sample = bucket_assignment[sample_indices]

            # Calculate the labels for the training
            labels = self._calculate_training_labels(bucket_assignment_for_sample, ground_truth)

            # Train the model
            self._train_model(sample, labels)

            # Predict to which bucket each vector belongs
            bucket_predictions = self._predict(X, self.top_k_buckets_to_load_balance_between)[1]

            # Redistribute the objects
            n_shifts = self._redistribute(bucket_assignment, bucket_predictions, len(X))
            # print(f'{n_shifts=}')

        # TODO: What if the number of objects in a bucket is larger than the bucket size?
        # bucket_assignment = self._fix_bucket_assignment(bucket_assignment)

        # Add the vectors to the buckets
        for i, child_bucket in self.buckets.items():
            child_bucket.insert(X[bucket_assignment == i], I[bucket_assignment == i])

        self.is_trained = True

        # TODO: what if all objects were inserted into the same bucket? = unbalanced partitioning

    # def _fix_bucket_assignment(self, bucket_assignment: Tensor) -> Tensor:
    #     # TODO: move objects that are above the capacity limit into
    #     # ??? either (BLISS-like style) the least populated bucket or to the second most probable bucket (???-style)
    #     # BLISS-style
    #     # ???-style
    #     pass

    @measure_runtime
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

        return True  # Insertion successful

    def search(self, query: Tensor, k: int) -> tuple[np.ndarray, np.ndarray]:
        bucket_id = int(self._predict(query, 1)[1].item())
        return self.buckets[bucket_id].search(query, k)

    @measure_runtime
    def _prepare_ground_truth(self, sample: Tensor, k: int) -> np.ndarray:
        return faiss.knn(sample, sample, k, metric=faiss.METRIC_L2)[1]

    # For each sample object calculate whether one of the kNNs is contained in a particular bucket
    def _calculate_training_labels(self, bucket_assignment: np.ndarray, ground_truth: np.ndarray) -> np.ndarray:
        # TODO: remove the dependency on MultiLabelBinarizer -> remove scikit-learn dependency
        return MultiLabelBinarizer(classes=np.arange(self.n_buckets)).fit_transform(bucket_assignment[ground_truth])  # type: ignore

    def _get_least_populated_bucket(self, labels: np.ndarray, top_k_bucket_ids: Tensor) -> int:
        # Histogram of the number of objects in each bucket
        bucket_population = np.bincount(labels, minlength=self.n_buckets)
        # The least populated bucket among the top K
        return int(top_k_bucket_ids[np.argmin(bucket_population[top_k_bucket_ids])])

    @measure_runtime
    def _redistribute(self, bucket_assignment: np.ndarray, top_k_predicted_buckets: Tensor, dataset_size: int) -> int:
        """Redistribute objects across buckets (put each object the least populated bucket among the top K in an incremental manner).

        TODO: Different from BLISS, as this function does respect the bucket capacity limit.
        """
        n_shifts = 0

        # Reassign each object to the least populated bucket among the top K predicted buckets
        # However, (different from BLISS) respect the bucket capacity limit
        for i in range(dataset_size):
            # Select the least populated bucket among the top K
            # TODO: compute this outside the loop for all objects
            least_populated_bucket = self._get_least_populated_bucket(bucket_assignment, top_k_predicted_buckets[i])

            if (
                self.do_not_shift_when_object_is_within_top_k_buckets
                and bucket_assignment[i] in top_k_predicted_buckets[i]
            ):
                continue

            if bucket_assignment[i] != least_populated_bucket:
                # TODO: I want to move object to the least populated bucket that is not yet full
                # TODO: iterate over from most probable to least put into the first non-full bucket
                # Put the object into the least populated bucket
                n_shifts += 1
                bucket_assignment[i] = least_populated_bucket

        return n_shifts

    @measure_runtime
    def _train_model(self, X: Tensor, y: np.ndarray) -> None:
        train_loader = DataLoader(
            dataset=LabeledDataset(X, torch.from_numpy(y.astype(np.float32))),
            batch_size=256,
            shuffle=True,
        )

        # Train the model
        self.model.train()

        for _ in range(self.epochs):
            for X_batch, y_batch in train_loader:
                loss = self.loss_fn(self.model(X_batch), y_batch)

                # Do the backpropagation
                loss.backward()

                self.optimizer.step()
                self.optimizer.zero_grad()

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
