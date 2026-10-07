"""Symbol vectors shared by the v3 blocks."""
from dataclasses import dataclass
import numpy as np


def log_probs(values, floor_eps: float = 1e-6) -> np.ndarray:
    # ASSUMPTION[MA-02]
    p = np.asarray(values, dtype=float)
    if p.ndim != 1 or not np.all(np.isfinite(p)) or np.any(p < 0):
        raise ValueError("invalid probability vector")
    p = p + floor_eps
    p /= p.sum()
    return np.log(p)


def normalize_logits(values) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or not np.all(np.isfinite(values)):
        raise ValueError("invalid logits")
    z = values - np.max(values)
    return z - np.log(np.exp(z).sum())


@dataclass(frozen=True)
class Repertoire:
    """Symbol Distribution; NEED-F-04; MA-01, MA-02; Q-07, Q-09."""
    symbols: tuple[str, ...]
    layout: tuple[int, int] = (6, 6)

    def index(self, symbol: str) -> int:
        return self.symbols.index(symbol)


@dataclass(frozen=True)
class SymbolDistribution:
    """Symbol Distribution, a priori; NEED-F-04; MA-02; Q-09."""
    repertoire: Repertoire
    log_p: np.ndarray
    source: str
    context_len: int


@dataclass(frozen=True)
class Posterior:
    """A posteriori distribution; NEED-F-04; MA-02; Q-10."""
    epoch_id: int
    log_p: np.ndarray
    fusion: str
    params: dict
