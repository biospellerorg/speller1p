"""Posterior mass summary sent to the presentation module."""
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class PresentationFeedback:
    """Influence on presentation method; MA-16; Q-11."""
    epoch_id: int
    decided: str
    top_k: list[tuple[str, float]]


def make_feedback(post, rep, cfg):
    # ASSUMPTION[MA-16]
    p = np.exp(post.log_p)
    order = np.argsort(-p)
    chosen = []
    for idx in order[:cfg["k_max"]]:
        chosen.append((rep.symbols[int(idx)], float(p[idx])))
        if sum(x[1] for x in chosen) >= cfg["mass"]:
            break
    return PresentationFeedback(post.epoch_id, rep.symbols[int(order[0])], chosen)
