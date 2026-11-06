from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.stats import beta

from eeg_speller.datasets.session import epoch_starts, flash_rows, load_recording, source_indexes
from eeg_speller.physiology.session_p300 import P300FlashDecoder


def bits_per_decision(accuracy: float, n_classes: int) -> float:
    if n_classes <= 1:
        return 0.0
    p = float(np.clip(accuracy, 1e-10, 1 - 1e-10))
    return float(np.log2(n_classes) + p * np.log2(p) + (1 - p) * np.log2((1 - p) / (n_classes - 1)))


def clopper_pearson(correct: int, n: int, alpha: float = 0.05) -> list[float]:
    if n <= 0:
        return [0.0, 1.0]
    lower = 0.0 if correct <= 0 else float(beta.ppf(alpha / 2, correct, n - correct + 1))
    upper = 1.0 if correct >= n else float(beta.ppf(1 - alpha / 2, correct + 1, n - correct))
    return [lower, upper]


def flash_record(labels, scores) -> dict:
    y = np.asarray(labels, dtype=bool)
    called = np.asarray(scores, dtype=float) > 0
    n = int(y.size)
    correct = int(np.sum(called == y))
    accuracy = correct / n if n else 0.0
    true_positive = int(np.sum(called & y))
    false_positive = int(np.sum(called & ~y))
    false_negative = int(np.sum(~called & y))
    true_negative = int(np.sum(~called & ~y))
    recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
    specificity = true_negative / (true_negative + false_positive) if true_negative + false_positive else 0.0
    no_positive = not bool(np.any(called))
    if no_positive:
        f1 = None
    else:
        precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if precision + recall else 0.0
    return {"accuracy": accuracy, "f1": f1, "itr": bits_per_decision(accuracy, 2),
            "itr_unit": "bits_per_decision", "n": n, "accuracy_ci95": clopper_pearson(correct, n),
            "balanced_accuracy": 0.5 * (recall + specificity), "majority_class": no_positive}


def character_record(correct_flags, flashes_per_character, soa_s, n_symbols) -> dict:
    flags = np.asarray(correct_flags, dtype=bool)
    n = int(flags.size)
    correct = int(flags.sum())
    accuracy = correct / n if n else 0.0
    flashes = np.asarray(flashes_per_character, dtype=float)
    duration = float(np.mean(flashes) * soa_s) if n and soa_s else 0.0
    itr = bits_per_decision(accuracy, n_symbols) / duration * 60 if duration else None
    return {"accuracy": accuracy, "f1": None, "itr": itr, "itr_unit": "bits_per_min", "n": n,
            "accuracy_ci95": clopper_pearson(correct, n)}


def protocol_name(train_subjects, test_subject: str) -> str:
    subjects = list(dict.fromkeys(train_subjects))
    if not subjects:
        raise ValueError("the model has no training subjects")
    if all(subject == test_subject for subject in subjects):
        return "within"
    if test_subject in subjects:
        raise ValueError("the test subject is also in the training sessions")
    if len(subjects) == 1:
        return "transfer"
    return "loso"


def evaluate(session_dir, model_path) -> dict:
    model = json.loads(Path(model_path).read_text(encoding="utf-8"))
    manifest, events, _eeg, _channels, _fs = load_recording(session_dir)
    dataset = manifest["dataset"]
    if dataset.get("name") != model.get("dataset_name"):
        raise ValueError("dataset name differs from the model")
    test_subject = dataset["subject"]
    test_indexes = set(source_indexes(events))
    for item in model.get("train_sessions", []):
        if item["subject"] == test_subject and test_indexes & set(item.get("source_indexes", [])):
            raise ValueError("test flashes overlap a training session of the same subject")
    protocol = protocol_name(model.get("train_subjects", []), test_subject)
    scores = P300FlashDecoder(model_path).score_flashes(session_dir)
    flashes = flash_rows(events)
    labels = [bool(event["payload"]["is_target"]) for event in flashes]
    record = {"protocol": protocol, "train_subjects": list(model.get("train_subjects", [])),
              "flash": flash_record(labels, scores)}
    characters = _characters(events, scores, manifest)
    if characters is not None:
        record["character"] = characters
    out = Path(session_dir) / "derived" / "metrics.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    return record


def _characters(events, scores, manifest):
    starts = epoch_starts(events)
    if not any(payload.get("target_symbol") for payload in starts.values()):
        return None
    flashes = flash_rows(events)
    if any("indices" not in event["payload"] for event in flashes):
        return None
    repertoire = list(manifest["repertoire"])
    by_epoch: dict[int, list] = {}
    for event, score in zip(flashes, scores):
        by_epoch.setdefault(event["epoch_id"], []).append((event, float(score)))
    correct = []
    counts = []
    for epoch_id, payload in starts.items():
        target = payload.get("target_symbol")
        if not target or epoch_id not in by_epoch:
            continue
        total = np.zeros(len(repertoire))
        for event, score in by_epoch[epoch_id]:
            for index in event["payload"]["indices"]:
                total[int(index)] += score
        predicted = repertoire[int(np.argmax(total))]
        correct.append(predicted == target)
        counts.append(len(by_epoch[epoch_id]))
    if not correct:
        return None
    soa = manifest["dataset"].get("soa_s")
    if soa is None:
        return None
    return character_record(correct, counts, float(soa), len(repertoire))
