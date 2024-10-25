from __future__ import annotations

from typing import TYPE_CHECKING

import faiss
import numpy as np
import torch
from faiss import merge_knn_results
from sklearn.preprocessing import MultiLabelBinarizer
from torch import Tensor
from torch.nn import BCEWithLogitsLoss, Linear, ReLU, Sequential
from torch.nn.functional import softmax
from torch.optim.adam import Adam
from torch.utils.data import DataLoader

from dynamic_bucket import DynamicBucket
from index import Index
from labeled_dataset import LabeledDataset
from sampling import take_sample
from utils import measure_runtime, np_rng

if TYPE_CHECKING:
    from bucket import Bucket
    from configuration import IndexConfig


class BLISSIndex(Index):
    def __init__(self, config: IndexConfig) -> None:
        super().__init__(config)

        self.bucket_size = config.bucket_shape[0]
        self.sample_size: int = self.bucket_size  # TODO: think about sample_size
        self.k_training: int = 100
        self.top_k_buckets_to_load_balance_between: int = (
            config.n_buckets  # ? Tied to the number of buckets? In the paper it is rather a small number...
        )
        self.n_redistributions: int = 2
        """Number of buckets."""
        self.buckets = {i: DynamicBucket(config.bucket_shape, config.distance.metric) for i in range(config.n_buckets)}

        self.model = Sequential(
            Linear(config.bucket_shape[1], 512),
            ReLU(),
            Linear(512, config.n_buckets),
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
        sample_size = sum(b.get_n_objects() for b in buckets)  # TODO: change
        X_sample, I_sample = take_sample(buckets, sample_size, self.config.bucket_shape[1])

        total_n_objects = sum(b.get_n_objects() for b in buckets)

        # Randomly distribute the dataset into buckets
        bucket_assignment = np_rng.choice(self.config.n_buckets, total_n_objects)

        # Calculate kNN ground truth for the sample on the sample
        ground_truth = self._prepare_ground_truth(X_sample, self.k_training)

        for _ in range(self.n_redistributions):
            # Select the bucket assignment of the sample for training
            bucket_assignment_for_sample = bucket_assignment[I_sample]

            # Calculate the labels for the training
            labels = self._calculate_training_labels(bucket_assignment_for_sample, ground_truth)

            # Train the model
            self._train_model(X_sample, labels)

            offset = 0
            bucket_predictions = torch.empty(
                (total_n_objects, self.top_k_buckets_to_load_balance_between),
                dtype=torch.int32,
            )
            for existing_bucket in buckets:
                # Predict to which bucket each vector belongs
                bucket_predictions[offset : offset + existing_bucket.get_n_objects(), :] = self._predict(
                    existing_bucket.get_data(),
                    self.top_k_buckets_to_load_balance_between,
                )[1]

                offset += existing_bucket.get_n_objects()

            # Redistribute the objects
            n_shifts = self._redistribute(bucket_assignment, bucket_predictions, total_n_objects)
            # logger.info(f'{n_shifts=}')

        # TODO: What if the number of objects in a bucket is larger than the bucket size?
        # bucket_assignment = self._fix_bucket_assignment(bucket_assignment)

        self._assign_objects_to_new_buckets(bucket_assignment, buckets)

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

        # Because we use dynamic bucket size, we do not check for overflowing buckets.

        # Add the vectors to the buckets
        for i, child_bucket in self.buckets.items():
            child_bucket.insert_bulk(X[bucket_ids == i], I[bucket_ids == i])

        return True  # Insertion successful

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

    def _assign_objects_to_new_buckets(self, bucket_assignment: np.ndarray, buckets: list[Bucket]) -> None:
        # Add the vectors to the new buckets
        offset = 0

        for existing_bucket in buckets:
            bucket_data = existing_bucket.get_data()
            bucket_indexes = existing_bucket.get_ids()
            relevant_bucket_assignment = bucket_assignment[offset : offset + existing_bucket.get_n_objects()]

            for i, new_child_bucket in self.buckets.items():
                new_child_bucket.insert_bulk(
                    bucket_data[relevant_bucket_assignment == i],
                    bucket_indexes[np.where(relevant_bucket_assignment == i)],
                )

            offset += existing_bucket.get_n_objects()

    @measure_runtime
    def _prepare_ground_truth(self, sample: Tensor, k: int) -> np.ndarray:
        return faiss.knn(sample, sample, k, metric=faiss.METRIC_L2)[1]

    # For each sample object calculate whether one of the kNNs is contained in a particular bucket
    def _calculate_training_labels(self, bucket_assignment: np.ndarray, ground_truth: np.ndarray) -> np.ndarray:
        # TODO: remove the dependency on MultiLabelBinarizer -> remove scikit-learn dependency
        return MultiLabelBinarizer(classes=np.arange(self.n_buckets)).fit_transform(bucket_assignment[ground_truth])  # type: ignore

    def _get_least_populated_bucket(self, labels: np.ndarray, top_k_bucket_ids: Tensor) -> int:
        # Histogram of the number of objects in each bucket
        bucket_population = np.bincount(labels, minlength=self.config.n_buckets)
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
