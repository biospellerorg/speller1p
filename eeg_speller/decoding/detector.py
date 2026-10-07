"""MAP symbol selection with uncertainty and bounded abstention."""
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class Decision:
    epoch_id: int
    symbol: str | None
    confidence: float
    margin: float
    uncertain: bool
    cold_start: bool


class SymbolDetector:
    """Primary Symbol Detector; NEED-F-05; MA-14/15; Q-12."""
    def __init__(self, cfg: dict, rep):
        self.cfg, self.rep = cfg, rep
        self.updates = []

    def decide(self, post, position: int, retry_idx: int = 0):
        # ASSUMPTION[MA-14]
        # IMPROV[Q-12]
        p = np.exp(post.log_p)
        order = np.argsort(-p)
        top, second = map(int, order[:2])
        confidence, margin = float(p[top]), float(p[top] - p[second])
        uncertain = confidence < self.cfg["tau"]
        abstain = self.cfg["mode"] == "abstain" and uncertain and retry_idx < self.cfg["max_retries"]
        # ASSUMPTION[MA-15]
        return Decision(post.epoch_id, None if abstain else self.rep.symbols[top],
                        confidence, margin, uncertain, position < self.cfg["cold_start_symbols"])

    def receive_update(self, batch):
        # STUB[Q-18]
        self.updates.append(batch)
