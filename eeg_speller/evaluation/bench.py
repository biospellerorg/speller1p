"""Offline fast-language-model baseline benchmark on held-out tasks."""
import math
import time
import numpy as np
from eeg_speller.core.distributions import Repertoire
from eeg_speller.epoch.loop import PROJECT, read_tasks
from eeg_speller.llm.fast import create_fast
from eeg_speller.text.normalize import words


def bench_llm(cfg):
    # ASSUMPTION[MA-08]
    tasks = read_tasks(PROJECT / cfg.model_dump().get("bench", {}).get("task_file", "data/tasks_ru.txt"))
    model = create_fast(cfg.fast_llm, PROJECT / "data" / "corpus_ru_fallback.txt")
    rep = Repertoire(tuple(cfg.repertoire["symbols"]))
    logps, latencies = [], []
    hits = {prefix: {n: [] for n in (3, 5, 10)} for prefix in (0, 1, 2)}
    for text in tasks.values():
        for i, char in enumerate(text):
            start = time.perf_counter_ns()
            prior = model.predict(text[:i], rep)
            latencies.append((time.perf_counter_ns() - start) / 1e6)
            logps.append(float(prior.log_p[rep.index(char)]))
        task_words = words(text)
        context = ""
        for word in task_words:
            for prefix in (0, 1, 2):
                if prefix > len(word):
                    continue
                candidates = [w for w, _ in model.top_words(context + word[:prefix], 10)]
                for n in (3, 5, 10):
                    hits[prefix][n].append(word in candidates[:n])
            context += word + " "
    # ASSUMPTION[MA-30]
    return {"model_id": model.model_id, "model_version": model.version,
            "model_hash": model.weights_hash, "n_chars": len(logps),
            "bits_per_char": -float(np.mean(logps)) / math.log(2),
            "mean_logp_target_char": float(np.mean(logps)),
            "top_n": {str(prefix): {str(n): float(np.mean(values)) if values else None
                                    for n, values in ranks.items()} for prefix, ranks in hits.items()},
            "latency_p95_ms": float(np.percentile(latencies, 95)),
            "note": "mock n-gram baseline, not a large language model"}
