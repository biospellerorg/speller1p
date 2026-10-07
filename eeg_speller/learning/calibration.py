"""Offline grid fit of physiology temperature and prior weight."""
import numpy as np
from eeg_speller.core.distributions import normalize_logits


def calibrate_offline(labels, priors, physiology, repertoire, min_labels=100):
    # ASSUMPTION[MA-22]
    if len(labels) < min_labels:
        raise ValueError("not enough pseudo-labels for offline calibration")
    best = None
    for temperature in np.linspace(0.5, 2.0, 16):
        for alpha in np.linspace(0, 2.0, 21):
            loss = 0.0
            for label, prior, phys in zip(labels, priors, physiology):
                p = normalize_logits(alpha * prior + phys / temperature)
                loss -= p[repertoire.index(label)]
            if best is None or loss < best[0]:
                best = (loss, temperature, alpha)
    return {"loss": best[0], "temperature": best[1], "alpha": best[2]}
