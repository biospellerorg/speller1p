"""Pseudo-label broadcast without online model training."""
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class PseudoLabel:
    epoch_id: int
    detected: str
    corrected: str
    posterior_log_p: np.ndarray


@dataclass(frozen=True)
class PseudoLabelBatch:
    labels: list[PseudoLabel]
    source_correction_id: int


class RLBlock:
    """Direct/Indirect RL stub; MA-21/22; Q-18."""
    def __init__(self, cfg):
        self.cfg = cfg
        self.sent_positions = set()

    def on_secondary_update(self, primary, secondary, frozen_upto, history, correction_id):
        # ASSUMPTION[MA-21]
        if not self.cfg["enabled"]:
            return None
        labels = []
        limit = min(len(primary), len(secondary))
        for position in range(limit):
            if position in self.sent_positions or position >= len(history):
                continue
            if position < frozen_upto:
                epoch_id, log_p = history[position]
                labels.append(PseudoLabel(epoch_id, primary[position], secondary[position], log_p))
                self.sent_positions.add(position)
        return PseudoLabelBatch(labels, correction_id) if labels else None
