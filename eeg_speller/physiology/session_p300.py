from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.signal import butter, sosfiltfilt

from eeg_speller.datasets.session import flash_rows, load_recording, source_indexes
from eeg_speller.physiology.p300 import CalibratedP300


class FeatureSpec:
    def __init__(self, spec: dict):
        self.channels = list(spec["channels"])
        self.spec = spec

    def signature(self):
        return self.spec


def default_intervals(window_s):
    start, stop = float(window_s[0]), float(window_s[1])
    if start < 0:
        baseline = (max(start, -0.2), min(0.0, stop))
    else:
        baseline = (start, min(stop, start + 0.15))
    if baseline[1] <= baseline[0]:
        raise ValueError("the session window is too short for a baseline")
    if start <= 0.25 and 0.5 <= stop and 0.25 >= baseline[1]:
        response = (0.25, 0.5)
    else:
        response = (baseline[1], stop)
    if response[1] <= response[0]:
        raise ValueError("the session window is too short for a response")
    return baseline, response


def extract_vector(window, fs, spec, unit: str):
    x = np.asarray(window, dtype=float)
    if unit == "V":
        x = x * 1e6
    if x.ndim != 2 or not np.all(np.isfinite(x)):
        raise ValueError("invalid EEG window")
    artifact = spec.get("artifact_uv")
    if unit in ("V", "uV") and artifact is not None and np.max(np.ptp(x, axis=1)) > float(artifact):
        return None
    band = tuple(spec["bandpass_hz"])
    if not 0 < band[0] < band[1] < fs / 2:
        raise ValueError("bandpass must be inside the EEG Nyquist frequency")
    filtered = sosfiltfilt(butter(2, band, btype="bandpass", fs=fs, output="sos"), x, axis=1)
    window_s = spec["window_s"]

    def mean_between(interval):
        a = int(round((interval[0] - window_s[0]) * fs))
        b = int(round((interval[1] - window_s[0]) * fs))
        if b <= a:
            raise ValueError("feature interval is empty")
        return filtered[:, a:b].mean(axis=1)

    return mean_between(spec["response_s"]) - mean_between(spec["baseline_s"])


class P300FlashDecoder:
    def __init__(self, path, *, channels=None, baseline_s=None, response_s=None,
                 bandpass_hz=(0.5, 15.0), artifact_uv=150.0):
        self.path = Path(path)
        self.channels = None if channels is None else list(channels)
        self.baseline_s = baseline_s
        self.response_s = response_s
        self.bandpass_hz = tuple(bandpass_hz)
        self.artifact_uv = float(artifact_uv)

    def fit(self, session_dirs) -> dict:
        rows = []
        labels = []
        subjects = []
        train_sessions = []
        dataset_name = None
        spec = None
        for directory in session_dirs:
            manifest, events, eeg, channels, fs = load_recording(directory)
            dataset = manifest["dataset"]
            if dataset.get("label_source") != "ground_truth":
                raise ValueError(f"{directory} has no ground-truth flash labels")
            if dataset_name is None:
                dataset_name = dataset["name"]
            elif dataset["name"] != dataset_name:
                raise ValueError("training sessions must share a dataset name")
            spec = self._spec(dataset, channels)
            kept_indexes = []
            for event in flash_rows(events):
                payload = event["payload"]
                vector = self._vector(eeg, fs, channels, payload, spec, dataset.get("unit", "unknown"))
                if vector is None:
                    continue
                rows.append(vector)
                labels.append(bool(payload["is_target"]))
            for index in source_indexes(events):
                kept_indexes.append(index)
            subjects.append(dataset["subject"])
            train_sessions.append({"subject": dataset["subject"], "session": str(directory),
                                   "source_indexes": kept_indexes})
        if spec is None:
            raise ValueError("no training sessions")
        model = CalibratedP300.fit(np.vstack(rows), np.asarray(labels), FeatureSpec(spec), self.path)
        model["dataset_name"] = dataset_name
        model["train_subjects"] = subjects
        model["train_sessions"] = train_sessions
        self.path.write_text(json.dumps(model, indent=2), encoding="utf-8")
        return model

    def score_flashes(self, session_dir) -> np.ndarray:
        model = json.loads(self.path.read_text(encoding="utf-8"))
        spec = model["features"]
        weights = np.asarray(model["weights"], dtype=float)
        bias = float(model["bias"])
        manifest, events, eeg, channels, fs = load_recording(session_dir)
        unit = manifest["dataset"].get("unit", "unknown")
        scores = []
        for event in flash_rows(events):
            vector = self._vector(eeg, fs, channels, event["payload"], spec, unit)
            if vector is None:
                scores.append(0.0)
                continue
            scores.append(float(np.clip(vector @ weights + bias, -20, 20)))
        return np.asarray(scores, dtype=float)

    def _spec(self, dataset, channels):
        window = list(dataset["window_s"])
        baseline, response = default_intervals(window)
        if self.baseline_s is not None:
            baseline = tuple(self.baseline_s)
        if self.response_s is not None:
            response = tuple(self.response_s)
        chosen = list(channels if self.channels is None else self.channels)
        missing = [name for name in chosen if name not in channels]
        if missing:
            raise ValueError("channels not in the session: " + ", ".join(missing))
        _inside(window, baseline, "baseline")
        _inside(window, response, "response")
        artifact = None if dataset.get("unit") == "unknown" else self.artifact_uv
        return {"channels": chosen, "baseline_s": list(baseline), "response_s": list(response),
                "window_s": list(window), "bandpass_hz": list(self.bandpass_hz), "artifact_uv": artifact}

    def _vector(self, eeg, fs, channels, payload, spec, unit):
        index = [channels.index(name) for name in spec["channels"]]
        onset = int(payload["onset_sample"])
        start, stop = spec["window_s"]
        a = onset + int(round(start * fs))
        b = onset + int(round(stop * fs))
        if a < 0 or b > eeg.shape[1] or b <= a:
            raise ValueError("feature window does not lie inside the recording")
        return extract_vector(eeg[index, a:b], fs, spec, unit)


def _inside(window, interval, name):
    if not window[0] <= interval[0] < interval[1] <= window[1]:
        raise ValueError(f"{name} must lie inside the session window {window}")
