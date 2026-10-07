"""Two configurable prior/physiology fusion strategies."""
import numpy as np
from eeg_speller.core.distributions import Posterior, normalize_logits


class Fusion:
    """Statistical Inference; NEED-F-04/06; MA-12/13; Q-10."""
    def __init__(self, cfg: dict, llm_mode: str):
        self.cfg, self.llm_mode = cfg, llm_mode

    def combine(self, prior, obs):
        # ASSUMPTION[MA-03]
        if obs.semantics != "likelihood":
            raise ValueError("physiology prior must be removed before fusion")
        strategy = self.cfg["strategy"]
        # IMPROV[Q-10]
        alpha = float(self.cfg["alpha"]) if self.llm_mode == "on" else 0.0
        beta = float(self.cfg["beta"])
        if strategy == "log_pool":
            # ASSUMPTION[MA-12]
            out = normalize_logits(alpha * prior.log_p + beta * obs.log_p)
        elif strategy == "linear_pool":
            weight = float(self.cfg["lambda"]) if self.llm_mode == "on" else 0.0
            out = normalize_logits(np.log(weight * np.exp(prior.log_p) + (1 - weight) * np.exp(obs.log_p)))
        else:
            raise ValueError("unknown fusion strategy")
        # ASSUMPTION[MA-13]
        return Posterior(obs.epoch_id, out, strategy,
                         {"alpha": alpha, "beta": beta, "lambda": self.cfg["lambda"]})
