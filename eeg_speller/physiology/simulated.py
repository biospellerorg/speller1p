"""Synthetic P300 score source; no EEG processing."""
import numpy as np
from eeg_speller.core.distributions import normalize_logits
from eeg_speller.physiology.base import PhysiologyObservation


class SimulatedUser:
    """Homo Sapiens stub; copy spelling; MA-06; Q-22."""
    def __init__(self, rng, lapse_rate: float):
        self.rng = rng
        self.lapse_rate = lapse_rate

    def attends(self) -> bool:
        # ASSUMPTION[MA-06]
        return bool(self.rng.random() >= self.lapse_rate)


class SimulatedPhysiology:
    """Physiology Module stub; NEED-F-04; MA-03/04/05/07; Q-01/02/03/06."""
    def __init__(self, cfg: dict, user: SimulatedUser, rng):
        self.cfg, self.user, self.rng = cfg, user, rng
        self.msg = None
        self.target = None
        self.feedback = []
        self.updates = []

    def set_target(self, target: str) -> None:
        self.target = target

    def start_epoch(self, msg) -> None:
        self.msg = msg

    def get_observation(self) -> PhysiologyObservation:
        msg = self.msg
        if msg is None or self.target not in msg.repertoire.symbols:
            raise ValueError("epoch and target must be set")
        n = len(msg.repertoire.symbols)
        rows, cols = msg.repertoire.layout
        if msg.paradigm == "row_column":
            groups = [list(range(r * cols, (r + 1) * cols)) for r in range(rows)]
            groups += [list(range(c, n, cols)) for c in range(cols)]
        elif msg.paradigm == "single_symbol":
            # STUB[Q-06]
            groups = [[i] for i in range(n)]
        else:
            raise ValueError("unknown presentation paradigm")
        # ASSUMPTION[MA-04]
        # ASSUMPTION[MA-05]
        d = float(self.cfg["sim"]["d_prime"])
        # IMPROV[Q-03]
        if self.cfg["sim"].get("adjacency_leak", 0) != 0:
            raise NotImplementedError("adjacency leakage is reserved for EXP-08")
        attentive = self.user.attends()
        logits = np.zeros(n)
        reps = []
        scores = []
        events = []
        target_idx = msg.repertoire.index(self.target)
        for rep in range(msg.timing.n_repetitions):
            row = []
            for flash_idx, group in enumerate(groups):
                is_target = target_idx in group
                x = float(self.rng.normal(d if is_target and attentive else 0.0, 1.0))
                row.append(x)
                llr = d * x - d * d / 2
                logits[group] += llr
                events.append({"rep": rep, "flash_idx": flash_idx, "group": group,
                               "duration_ms": msg.timing.flash_ms, "sim_is_target": is_target})
            scores.append(row)
            reps.append(normalize_logits(logits.copy()))
        flags = [((x > d / 2) == (target_idx in groups[i]))
                 for row in scores for i, x in enumerate(row)]
        means = np.asarray(scores).mean(axis=0)
        mean_flags = [((x > d / 2) == (target_idx in group)) for x, group in zip(means, groups)]
        # ASSUMPTION[MA-23]
        metrics = {"binary_flash_accuracy": float(np.mean(flags)),
                   "binary_mean_accuracy": float(np.mean(mean_flags)),
                   "attentive": attentive}
        return PhysiologyObservation(msg.epoch_id, normalize_logits(logits), "likelihood",
                                     reps, events, metrics, None, scores)

    def receive_feedback(self, fb) -> None:
        # STUB[Q-11]
        self.feedback.append(fb)

    def receive_update(self, batch) -> None:
        # STUB[Q-18]
        self.updates.append(batch)
