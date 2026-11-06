from __future__ import annotations

import hashlib

import numpy as np

_SEED_MODULUS = 2**31


def subject_seed(base_seed: int, subject: str) -> int:
    digest = hashlib.blake2b(subject.encode("utf-8"), digest_size=8).digest()
    return (int(base_seed) + int.from_bytes(digest, "big")) % _SEED_MODULUS


def _take(indexes, labels, fraction, rng):
    """Hold out ``fraction`` of each class. Returns the held-out indexes."""
    if fraction <= 0:
        return np.array([], dtype=int)
    if not 0 < fraction < 1:
        raise ValueError(f"fraction must be in (0, 1), got {fraction}")
    held = []
    for label in np.unique(labels):
        pool = np.asarray(indexes)[np.asarray(labels) == label]
        pool = rng.permutation(pool)
        count = int(np.ceil(len(pool) * fraction))
        count = min(count, max(len(pool) - 1, 0)) if len(pool) > 1 else 0
        held.extend(pool[:count].tolist())
    return np.asarray(sorted(held), dtype=int)


def three_way_epoch_split(labels, *, epoch_holdout=0.3, val_fraction=0.2, seed=0):
    """Return train, val, test indexes. They partition ``0 .. len(labels)-1``."""
    labels = np.asarray(labels)
    n = len(labels)
    rng = np.random.default_rng(int(seed))
    everything = np.arange(n)
    test = _take(everything, labels, epoch_holdout, rng)
    rest = np.setdiff1d(everything, test)
    val_local = _take(np.arange(len(rest)), labels[rest], val_fraction, rng)
    val = rest[val_local] if len(val_local) else np.array([], dtype=int)
    train = np.setdiff1d(rest, val)
    if len(train) < 1:
        raise ValueError("split left no training epochs")
    return train, val, test


def chronological_split(n, *, val_fraction=0.14, test_fraction=0.3):
    """Prefix train, then val, then test. Fractions are of ``n``."""
    if n < 3:
        raise ValueError("chronological split needs at least three epochs")
    n_test = int(np.ceil(n * test_fraction))
    n_val = int(np.ceil(n * val_fraction))
    n_train = n - n_val - n_test
    if n_train < 1:
        raise ValueError("chronological split left no training epochs")
    idx = np.arange(n)
    return idx[:n_train], idx[n_train:n_train + n_val], idx[n_train + n_val:]
