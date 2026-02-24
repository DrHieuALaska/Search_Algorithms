from __future__ import annotations
import numpy as np


def make_rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(int(seed))


def combine_seeds(instance_seed: int, seed_algo: int, salt: int = 1_000_003) -> int:
    """Deterministic combination for reproducible init solutions across algorithms."""
    return int(instance_seed) * int(salt) + int(seed_algo)
