# New dataset refactor
from .forest_dataset_v2 import (
    ForestDatasetV2,
    collate_fn,
)

__all__ = [
    "ForestDatasetV2",
    "collate_fn",
]
