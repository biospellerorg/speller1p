"""Participant-level summaries and paired inference."""
import numpy as np
from scipy import stats


def summary(values, confidence=0.95):
    # ASSUMPTION[MA-31]
    # IMPROV[Q-21]
    # IMPROV[Q-23]
    x = np.asarray([v for v in values if v is not None], dtype=float)
    n = len(x)
    if not n:
        return {"n": 0, "mean": None, "variance": None, "ci95": None}
    mean = float(x.mean())
    variance = float(x.var(ddof=1)) if n > 1 else None
    half = float(stats.t.ppf((1 + confidence) / 2, n-1) * x.std(ddof=1) / np.sqrt(n)) if n > 1 else None
    return {"n": n, "mean": mean, "variance": variance,
            "ci95": [mean - half, mean + half] if half is not None else None}


def paired_growth(on, off, bootstrap_n=10000, seed=1):
    # ASSUMPTION[MA-26]
    on, off = np.asarray(on, float), np.asarray(off, float)
    if len(on) != len(off) or np.any(off <= 0):
        raise ValueError("paired speed samples must be positive and aligned")
    growth = float((on.mean() / off.mean() - 1) * 100)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(on), (bootstrap_n, len(on)))
    samples = (on[draws].mean(axis=1) / off[draws].mean(axis=1) - 1) * 100
    t = stats.ttest_rel(on, off)
    try:
        wilcoxon_p = float(stats.wilcoxon(on, off).pvalue)
    except ValueError:
        wilcoxon_p = None
    return {"growth_percent": growth,
            "mean_individual_growth_percent": float(np.mean((on / off - 1) * 100)),
            "bootstrap_ci95": list(map(float, np.percentile(samples, [2.5, 97.5]))),
            "paired_t_p": float(t.pvalue), "wilcoxon_p": wilcoxon_p}
