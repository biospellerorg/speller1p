from __future__ import annotations

from pathlib import Path

import numpy as np
import scipy.io

from eeg_speller.datasets.session import RUSSIAN_36, load_recording, source_indexes, write_import_session
from eeg_speller.datasets.splits import chronological_split, subject_seed, three_way_epoch_split

CHANNELS = ("P4", "PZ", "P3")


def load_samara(path: Path):
    raw = scipy.io.loadmat(path)
    epochs = np.asarray(raw["epochs"], dtype=np.float64)
    if epochs.ndim != 3 or epochs.shape[1] != 3:
        raise ValueError(f"{path.name}: epochs must have shape (N, 3, time)")
    labels = np.asarray(raw["labels"]).reshape(-1).astype(int)
    if len(labels) != epochs.shape[0]:
        raise ValueError(f"{path.name}: labels and epochs differ in length")
    fs = float(np.asarray(raw["fs"]).reshape(-1)[0]) if "fs" in raw else 250.0
    subject = _text(raw["subj"]) if "subj" in raw else path.name.split("-", 1)[0]
    names = _channel_names(raw) if "eeg_channels" in raw else list(CHANNELS)
    return {"epochs": epochs, "labels": labels, "fs": fs, "subject": subject, "channels": names,
            "path": path}


def _text(value) -> str:
    array = np.asarray(value).reshape(-1)
    if array.dtype.kind in "SU":
        return "".join(str(item) for item in array).strip()
    return str(array.reshape(-1)[0]).strip()


def _channel_names(raw) -> list[str]:
    value = raw["eeg_channels"]
    if getattr(value, "dtype", None) is not None and value.dtype.names:
        return [str(value[name].reshape(-1)[0]).strip() for name in value.dtype.names]
    text = _text(value)
    if text:
        parts = text.replace(",", " ").split()
        if len(parts) == 3:
            return parts
    return list(CHANNELS)


def find_mat(root: Path, subject: str, pattern: str) -> Path:
    root = Path(root)
    if root.is_file():
        return root
    matches = sorted(root.glob(pattern))
    own = [path for path in matches if subject in path.name or _subject_of(path) == subject]
    if len(own) == 1:
        return own[0]
    if len(own) > 1:
        raise ValueError(f"several files match {subject}: {', '.join(path.name for path in own)}")
    raise ValueError(f"no {pattern} for {subject} in {root}")


def _subject_of(path: Path) -> str:
    try:
        return _text(scipy.io.loadmat(path, variable_names=["subj"])["subj"])
    except (KeyError, ValueError, NotImplementedError, OSError):
        return path.name.split("-", 1)[0]


def split_indexes(labels, *, mode: str, seed: int):
    if mode == "chronological":
        return chronological_split(len(labels))
    if mode == "stratified":
        return three_way_epoch_split(labels, seed=seed)
    raise ValueError("split mode must be stratified or chronological")


def import_samara(root, output, *, subject: str, pattern="S*-P300_classic.mat",
                  mode="stratified", base_seed=0) -> list[Path]:
    path = find_mat(Path(root), subject, pattern)
    loaded = load_samara(path)
    mat_subject = loaded["subject"]
    if mat_subject and mat_subject != subject:
        raise ValueError(f"{path.name} belongs to {mat_subject}, not {subject}")
    labels = loaded["labels"]
    seed = subject_seed(base_seed, subject)
    parts = split_indexes(labels, mode=mode, seed=seed)
    names = ("train", "val", "test")
    written = []
    output = Path(output)
    fs = loaded["fs"]
    n_time = loaded["epochs"].shape[2]
    window_s = [0.0, n_time / fs]
    for name, indexes in zip(names, parts):
        if len(indexes) == 0:
            continue
        chosen = loaded["epochs"][indexes]  # (K, 3, T)
        eeg = np.transpose(chosen, (1, 0, 2)).reshape(3, -1)
        epochs = []
        for local, source_index in enumerate(indexes):
            epochs.append({"source_index": int(source_index), "flashes": [{
                "is_target": bool(labels[source_index]),
                "onset_sample": local * n_time,
                "n_samples": n_time,
                "rep": 0,
            }]})
        dataset = {"name": "samara", "subject": subject, "split": name, "role": "import",
                   "label_source": "ground_truth", "source_path": str(path), "fs": fs,
                   "channels": loaded["channels"], "unit": "unknown", "window_s": window_s,
                   "concatenated_epochs": True}
        directory = output / f"samara_{subject}_{name}"
        written.append(write_import_session(
            directory, dataset=dataset, repertoire=list(RUSSIAN_36), paradigm="single_symbol",
            eeg=eeg, fs=fs, channels=loaded["channels"], epochs=epochs, seed=seed))
    _assert_partition(written, len(labels))
    return written


def _assert_partition(directories, n):
    found = []
    for directory in directories:
        _manifest, events, *_rest = load_recording(directory)
        found.extend(source_indexes(events))
    if sorted(found) != list(range(n)):
        raise ValueError("Samara splits do not partition the source epochs")
