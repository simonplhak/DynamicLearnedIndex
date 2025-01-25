from __future__ import annotations

import time
from typing import TYPE_CHECKING

from dli.bucket import Bucket
from dli.config import IndexConfig
from dli.statistic import FrameworkCompactionStatistics

if TYPE_CHECKING:
    from torch import Tensor

    from dli.config.dli import DLIConfig
    from dli.dynamic_learned_index import DynamicLearnedIndex


class BentleySaxe:
    def __init__(self, config: DLIConfig, dli: DynamicLearnedIndex) -> None:
        self.config = config
        self.dli = dli

    def _empty_upper_levels(self, current_level: int) -> int:
        # Empty all levels above the current level as the bucket objects are now in the new index and should be empty
        self.dli.buffer.empty()

        deallocated_spaces = 0

        for i in range(current_level - 1):
            deallocated_spaces += self.dli.levels[i].empty()

        return deallocated_spaces

    def compact(self, X: Tensor, I: int) -> FrameworkCompactionStatistics:
        s = time.time()

        self.dli.buffer.insert_single(X, I)

        if not self.dli.buffer.is_full():
            return FrameworkCompactionStatistics(
                total_model_training_time=0,
                total_compaction_time=time.time() - s,
                allocated_new_level=False,
                n_retrained_indexes=0,
                deallocated_spaces=0,
            )

        # If the buffer is full, we need to merge it with the first level
        return self.compact_recursive(current_level=1, start_time=s)

    def compact_recursive(self, current_level: int, start_time: float) -> FrameworkCompactionStatistics:
        """Compact the data from the buffer into the index.

        Compact the data from the buffer by either creating a new level
        or merging it with an existing one. The buffer will be emptied by this method.
        """
        # Take all data above this level and either create a new index or merge the data into an existing one

        # Option 1 -- We have descended beyond the existing levels, there is no index, we must create one
        if current_level > len(self.dli.levels):
            index = self.dli.config.index_class(
                IndexConfig(
                    n_buckets=pow(self.dli.config.arity, current_level),
                    distance=self.dli.config.distance,
                    bucket_shape=self.dli.config.bucket_shape,
                    sample_threshold=self.dli.config.sample_threshold,
                    shrink_buckets_during_compaction=self.dli.config.shrink_buckets_during_compaction,
                    n_training_samples=sum(map(Bucket.get_n_objects, self.dli.get_buckets(current_level - 1))),
                ),
            )
            model_training_time = index.train(self.dli.get_buckets(current_level - 1))
            self.dli.levels.append(index)
            deallocated_spaces = self._empty_upper_levels(current_level)
            return FrameworkCompactionStatistics(
                total_model_training_time=model_training_time,
                total_compaction_time=time.time() - start_time,
                allocated_new_level=True,
                n_retrained_indexes=0,
                deallocated_spaces=deallocated_spaces,
            )

        # Option 2 -- We are on a level that already exists
        current_index = self.dli.levels[current_level - 1]

        ## Option 2.1 -- The index does not exist at this level, we have to create it
        ##            -- Actually, we are just training it because we have not thrown out the old one...
        if current_index.is_empty():  # ~ the index does not exist
            # TODO: reset the model's weights first
            model_training_time = current_index.train(self.dli.get_buckets(current_level - 1))
            deallocated_spaces = self._empty_upper_levels(current_level)
            return FrameworkCompactionStatistics(
                total_model_training_time=model_training_time,
                total_compaction_time=time.time() - start_time,
                allocated_new_level=False,
                n_retrained_indexes=0,
                deallocated_spaces=deallocated_spaces,
            )

        ## Option 2.2 -- The index exists
        can_be_inserted = current_index.get_free_space() >= sum(
            map(Bucket.get_n_objects, self.dli.get_buckets(current_level - 1)),
        )

        # is_successfully_inserted = current_index.insert(self.dli.get_buckets(current_level - 1))

        ### Option 2.2.1 -- All data has been stored at this level
        # if is_successfully_inserted:
        if can_be_inserted:
            current_index.insert(self.dli.get_buckets(current_level - 1))
            deallocated_spaces = self._empty_upper_levels(current_level)
            return FrameworkCompactionStatistics(
                total_model_training_time=0,
                total_compaction_time=time.time() - start_time,
                allocated_new_level=False,
                n_retrained_indexes=0,
                deallocated_spaces=deallocated_spaces,
            )

        ### Option 2.2.2 -- This level is overflowing, try to fit the objects in the level below
        ### The level is not able to absorb the data = overflow detected ~ `if not is_successfully_inserted`
        ### Three options:
        #### Option 2.2.2.1 -- retrain(level) = if the total number of objects is less than BUCKET_SIZE
        #### -> we can retrain the model and reorganize the data
        #### TODO: implement
        #### Option 2.2.2.2 -- compact(level + 1) = if the total number of objects is equal to BUCKET_SIZE
        #### -> we have no other choice
        return self.compact_recursive(current_level + 1, start_time)
        #### Option 2.2.2.3 -- train for a bit using BLISS training procedure
        #### = use BLISS to reorganize the existing and new data
        #### TODO: implement
