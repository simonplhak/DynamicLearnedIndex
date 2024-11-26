from __future__ import annotations

import time
from typing import TYPE_CHECKING

from bucket import Bucket
from configuration import FrameworkConfig, IndexConfig
from framework import Framework
from statistic import FrameworkCompactionStatistics

if TYPE_CHECKING:
    from torch import Tensor


class BentleySaxe(Framework):
    def __init__(self, config: FrameworkConfig) -> None:
        super().__init__(config)

    def compact(self, X: Tensor, I: int) -> FrameworkCompactionStatistics:
        s = time.time()

        self.buffer.insert_single(X, I)

        if not self.buffer.is_full():
            return FrameworkCompactionStatistics(0, time.time() - s, False, 0)

        # If the buffer is full, we need to merge it with the first level
        return self.compact_recursive(current_level=1, start_time=s)

    def compact_recursive(self, current_level: int, start_time: float) -> FrameworkCompactionStatistics:
        """Compact the data from the buffer into the index.

        Compact the data from the buffer by either creating a new level
        or merging it with an existing one. The buffer will be emptied by this method.
        """
        # Take all data above this level and either create a new index or merge the data into an existing one

        # Option 1 -- We have descended beyond the existing levels, there is no index, we must create one
        if current_level > len(self.levels):
            index = self.config.index_class(
                IndexConfig(
                    n_buckets=pow(self.config.arity, current_level),
                    distance=self.config.distance,
                    bucket_shape=self.config.bucket_shape,
                    sample_threshold=self.config.sample_threshold,
                ),
            )
            model_training_time = index.train(self.get_buckets(current_level - 1))
            self._create_new_level(index)
            self._empty_upper_levels(current_level)
            return FrameworkCompactionStatistics(model_training_time, time.time() - start_time, True, 0)

        # Option 2 -- We are on a level that already exists
        current_index = self.levels[current_level - 1]

        ## Option 2.1 -- The index does not exist at this level, we have to create it
        ##            -- Actually, we are just training it because we have not thrown out the old one...
        if current_index.is_empty():  # ~ the index does not exist
            # TODO: reset the model's weights first
            model_training_time = current_index.train(self.get_buckets(current_level - 1))
            self._empty_upper_levels(current_level)
            return FrameworkCompactionStatistics(model_training_time, time.time() - start_time, False, 0)

        ## Option 2.2 -- The index exists
        can_be_inserted = current_index.get_free_space() >= sum(
            map(Bucket.get_n_objects, self.get_buckets(current_level - 1)),
        )

        # is_successfully_inserted = current_index.insert(self.get_buckets(current_level - 1))

        ### Option 2.2.1 -- All data has been stored at this level
        # if is_successfully_inserted:
        if can_be_inserted:
            current_index.insert(self.get_buckets(current_level - 1))
            self._empty_upper_levels(current_level)
            return FrameworkCompactionStatistics(0, time.time() - start_time, False, 0)

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
