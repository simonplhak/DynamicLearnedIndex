from __future__ import annotations

from typing import TYPE_CHECKING

from framework import Framework

if TYPE_CHECKING:
    from torch import Tensor

    from index import Index

# ! Currently broken as `.exists()` method was removed from `Index`

# class BentleySaxe(Framework):
#     def __init__(
#         self,
#         index_class: type[Index],
#         arity: int,
#         bucket_shape: tuple[int, int],
#         metric: int,
#         keep_max: bool,
#     ) -> None:
#         super().__init__(index_class, arity, bucket_shape, metric, keep_max)

#     def compact(
#         self,
#         X: Tensor,
#         I: int,
#     ) -> None:
#         self.buffer.insert_single(X, I)

#         if not self.buffer.is_full():
#             return

#         # If the buffer is full, we need to merge it with the first level
#         self.compact_recursive(current_level=1)

#     def compact_recursive(self, current_level: int) -> None:
#         """Compact the data from the buffer into the index.

#         Compact the data from the buffer by either creating a new level
#         or merging it with an existing one. The buffer will be emptied by this method.
#         """
#         # Take all data above this level and either create a new index or merge the data into an existing one

#         # Option 1 -- We have descended beyond the existing levels, there is no index, we must create one
#         if current_level > len(self.levels):
#             index = self.index_class(
#                 n_buckets=pow(self.arity, current_level),
#                 metric=self.metric,
#                 bucket_shape=self.bucket_shape,
#             )
#             index.train(self.get_buckets(current_level - 1))
#             self._create_new_level(index)
#             self._empty_upper_levels(current_level)

#         # Option 2 -- We are on a level that already exists
#         else:
#             current_index = self.levels[current_level - 1]

#             # Option 2.1 -- The index does not exist at this level, we have to create it
#             #            -- Actually, we are just training it because we have not thrown out the old one...
#             if not current_index.exists():
#                 # TODO: reset the model's weights first
#                 current_index.train(self.get_buckets(current_level - 1))
#                 self._empty_upper_levels(current_level)

#             # Option 2.2 -- The index exists
#             else:
#                 is_successfully_inserted = current_index.insert(self.get_buckets(current_level - 1))

#                 # Option 2.2.1 -- All data has been stored at this level
#                 if is_successfully_inserted:
#                     self._empty_upper_levels(current_level)
#                     return

#                 # Option 2.2.2 -- This level is overflowing, try to fit the objects in the level below
#                 if not is_successfully_inserted:  # The level is not able to absorb the data = overflow detected
#                     # Option 2.2.2.1 -- retrain(level) = if the total number of objects is less than BUCKET_SIZE -> we can retrain the model and reorganize the data
#                     # TODO: implement
#                     # Option 2.2.2.2 -- comact(level + 1) = if the total number of objects is equal to BUCKET_SIZE -> we have no other choice
#                     self.compact_recursive(current_level + 1)
#                     # Option 2.2.2.3 -- train for a bit using BLISS training procedure = use BLISS to reorganize the existing and new data
#                     # TODO: implement

#                     return
