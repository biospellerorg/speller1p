from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from eeg_speller.recording.storage import check_integrity, read_events

RUSSIAN_36 = (
    "а", "б", "в", "г", "д", "е", "ж", "з", "и", "й", "к", "л", "м", "н", "о", "п",
    "р", "с", "т", "у", "ф", "х", "ц", "ч", "ш", "щ", "ъ", "ы", "ь", "э", "ю", "я",
    " ", ".", ",", "⌫",
)


def write_import_session(directory, *, dataset: dict, repertoire: list[str], paradigm: str,
                         eeg, fs: float, channels: list[str], epochs: list[dict],
                         target_text: str = "", seed: int = 0) -> Path:
    """Write one subject split. ``epochs`` are characters or, for Samara, single flashes.

    Each epoch is ``{"target_symbol": str | None, "source_index": int | None,
    "flashes": [{"is_target", "onset_sample", "n_samples", ...}]}``.
    ``eeg`` is ``(C, N)`` in the recording's own unit.
    """
    directory = Path(directory)
    if len(repertoire) < 1:
        raise ValueError("repertoire is empty")
    eeg = np.asarray(eeg, dtype=np.float32)
    if eeg.ndim != 2 or eeg.shape[0] != len(channels):
        raise ValueError("eeg must have one row per channel")
    for name in ("texts", "tables", "derived", "external", "external/physiology"):
        (directory / name).mkdir(parents=True, exist_ok=True)
    session_id = directory.name
    signal = directory / "external" / "physiology" / "lsl_eeg.npz"
    timestamps = np.arange(eeg.shape[1], dtype=np.float64) / float(fs)
    np.savez_compressed(signal, timestamps_s=timestamps, eeg_v=eeg,
                        channels=np.asarray(channels), sample_rate_hz=np.float64(fs))
    digest = hashlib.sha256(signal.read_bytes()).hexdigest()
    (directory / "external" / "refs.json").write_text(json.dumps([{
        "path": "external/physiology/lsl_eeg.npz", "sha256": digest, "kind": "lsl_eeg"}],
        indent=2) + "\n", encoding="utf-8")
    manifest = {
        "schema_version": "eeg-speller/events@1", "session_id": session_id,
        "participant_id": dataset["subject"], "experiment_id": dataset["name"],
        "task_id": "import", "started_utc": None, "ended_utc": None,
        "clock": {"kind": "virtual", "t0_ns": 0}, "run_mode": "semi_synthetic",
        "llm_mode": "off", "paradigm": paradigm, "repertoire": list(repertoire),
        "seed": int(seed), "code_version": "0.1.0",
        "models": {"fast_llm": {"backend": "uniform", "model_id": "uniform", "version": "1", "hash": "none"},
                   "slow_llm": {"backend": "disabled", "model_id": "disabled", "version": "1", "hash": "none"},
                   "physiology": {"backend": "import", "model_id": dataset["name"], "version": "1", "hash": digest}},
        "hardware": {"platform": None, "cpu": None, "gpu": None, "audio_device": None},
        "replay_of": None, "counts": {}, "status": "completed", "dataset": dataset,
    }
    events = []

    def emit(kind, block, payload, epoch_id, t_ns):
        events.append({"schema": "eeg-speller/events@1", "session_id": session_id,
                       "participant_id": dataset["subject"], "seq": len(events), "t_ns": int(t_ns),
                       "block": block, "type": kind, "epoch_id": epoch_id, "payload": payload})

    emit("session_started", "Session Recorder",
         {"run_mode": "semi_synthetic", "llm_mode": "off", "paradigm": paradigm, "seed": int(seed)}, None, 0)
    if target_text:
        emit("task_assigned", "Session Recorder",
             {"task_id": "import", "target_text": target_text}, None, 0)
    labeled = dataset.get("label_source") in ("ground_truth", "simulated")
    for epoch_id, epoch in enumerate(epochs):
        started = {"position": epoch_id, "paradigm": paradigm}
        if epoch.get("target_symbol"):
            started["target_symbol"] = epoch["target_symbol"]
        if epoch.get("source_index") is not None:
            started["source_index"] = int(epoch["source_index"])
        first = epoch["flashes"][0]["onset_sample"] if epoch["flashes"] else 0
        emit("epoch_started", "One Epoch", started, epoch_id, _ns(first, fs))
        for flash_idx, flash in enumerate(epoch["flashes"]):
            payload = {"flash_idx": flash_idx, "rep": flash.get("rep", 0),
                       "onset_sample": int(flash["onset_sample"]), "n_samples": int(flash["n_samples"])}
            if "indices" in flash:
                payload["indices"] = [int(i) for i in flash["indices"]]
            if "kind" in flash:
                payload["kind"] = flash["kind"]
                payload["index"] = int(flash["index"])
            if "stimulus_code" in flash:
                payload["stimulus_code"] = int(flash["stimulus_code"])
            if labeled:
                payload["is_target"] = bool(flash["is_target"])
            emit("stimulus_event", "Presentation Method", payload, epoch_id, _ns(flash["onset_sample"], fs))
    emit("session_ended", "Session Recorder",
         {"status": "completed", "reason": "import", "n_epochs": len(epochs), "n_events": len(events) + 1},
         None, _ns(eeg.shape[1], fs))
    manifest["counts"] = {"epochs": len(epochs), "events": len(events), "corrections": 0}
    (directory / "session.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    with (directory / "events.jsonl").open("w", encoding="utf-8") as fh:
        for event in events:
            fh.write(json.dumps(event, ensure_ascii=False, separators=(",", ":")) + "\n")
    for name, text in (("target", target_text), ("primary_final", ""), ("secondary_final", "")):
        (directory / "texts" / f"{name}.txt").write_text(text, encoding="utf-8")
    integrity = check_integrity(directory)
    (directory / "derived" / "integrity.json").write_text(
        json.dumps(integrity, ensure_ascii=False, indent=2), encoding="utf-8")
    if not integrity["complete"]:
        raise ValueError("imported session is incomplete: " + "; ".join(integrity["errors"]))
    return directory


def _ns(sample, fs):
    return int(round(float(sample) / float(fs) * 1e9))


def load_recording(session_dir):
    session_dir = Path(session_dir)
    manifest = json.loads((session_dir / "session.json").read_text(encoding="utf-8"))
    events = read_events(session_dir)
    with np.load(session_dir / "external" / "physiology" / "lsl_eeg.npz") as data:
        eeg = np.asarray(data["eeg_v"], dtype=np.float64)
        channels = [str(item) for item in np.asarray(data["channels"]).reshape(-1)]
        fs = float(np.asarray(data["sample_rate_hz"]))
    return manifest, events, eeg, channels, fs


def flash_rows(events):
    """Stimulus events in journal order."""
    return [event for event in events if event["type"] == "stimulus_event"]


def epoch_starts(events):
    return {event["epoch_id"]: event["payload"] for event in events if event["type"] == "epoch_started"}


def source_indexes(events):
    found = []
    for event in events:
        if event["type"] == "epoch_started" and "source_index" in event["payload"]:
            found.append(int(event["payload"]["source_index"]))
    return found
