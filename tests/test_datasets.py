import json
from pathlib import Path

import numpy as np
import pytest
import scipy.io

from eeg_speller.datasets.bci3 import BCI3_CHARS, BCI3_FS, flash_onsets, import_bci3
from eeg_speller.evaluation.flash import evaluate
from eeg_speller.physiology.session_p300 import P300FlashDecoder
from eeg_speller.datasets.samara import import_samara
from eeg_speller.datasets.session import flash_rows, load_recording, source_indexes


def _bci_mat(path, *, n_reps=2, n_channels=4, phrase="AB"):
    on, off, step = 20, 16, 36
    flashes = 12 * n_reps
    n_times = flashes * step + int(0.8 * BCI3_FS)
    signal = np.zeros((len(phrase), n_times, n_channels))
    flashing = np.zeros((len(phrase), n_times))
    codes = np.zeros_like(flashing)
    types = np.zeros_like(flashing)
    for char_i, character in enumerate(phrase):
        cell = BCI3_CHARS.index(character)
        targets = {7 + cell // 6, 1 + cell % 6}
        for flash_i in range(flashes):
            code = (flash_i % 12) + 1
            start = flash_i * step
            flashing[char_i, start:start + on] = 1
            codes[char_i, start:start + on] = code
            types[char_i, start:start + on] = float(code in targets)
            if code in targets:
                t = np.arange(int(0.25 * BCI3_FS), int(0.5 * BCI3_FS))
                signal[char_i, start + t, :] = (40 * np.sin(2 * np.pi * 10 * t / BCI3_FS))[:, None]
    scipy.io.savemat(path, {"Signal": signal, "Flashing": flashing, "StimulusCode": codes,
                            "StimulusType": types, "TargetChar": phrase})


def _samara_mat(folder, subject, n=40, n_time=80):
    folder.mkdir(parents=True, exist_ok=True)
    epochs = np.zeros((n, 3, n_time))
    labels = np.zeros((1, n), dtype=np.int64)
    labels[0, : n // 2] = 1
    epochs[:, :, 10] = labels.T
    scipy.io.savemat(folder / f"{subject}-P300_classic.mat",
                     {"epochs": epochs, "labels": labels, "fs": np.array([[250.0]]), "subj": subject})


def test_bci_import_round_trip(tmp_path):
    mat = tmp_path / "Subject_A_Train.mat"
    _bci_mat(mat)
    (tmp_path / "eloc64.txt").write_text("1 0 0 Cz..\n2 0 0 Pz..\n3 0 0 P3..\n4 0 0 P4..\n", encoding="utf-8")
    out = import_bci3(mat, tmp_path / "bci3_A_train", subject="A")
    manifest, events, eeg, channels, _fs = load_recording(out)
    flashes = flash_rows(events)
    raw = scipy.io.loadmat(mat)
    assert manifest["dataset"]["name"] == "bci3" and channels == ["Cz", "Pz", "P3", "P4"]
    assert len(flashes) == sum(len(flash_onsets(row)) for row in raw["Flashing"])
    assert sorted({flash["payload"]["stimulus_code"] for flash in flashes}) == list(range(1, 13))
    first = flashes[0]["payload"]
    assert np.allclose(eeg[:, first["onset_sample"]:first["onset_sample"] + first["n_samples"]],
                       raw["Signal"][0].T[:, :first["n_samples"]])
    assert (out / "texts" / "target.txt").read_text(encoding="utf-8") == "AB"


def test_samara_split_has_no_letters(tmp_path):
    folder = tmp_path / "Samara_data"
    _samara_mat(folder, "S0201")
    written = import_samara(folder, tmp_path / "out", subject="S0201")
    found, classes = [], {}
    for path in written:
        manifest, events, eeg, _channels, _fs = load_recording(path)
        indexes = source_indexes(events)
        found.extend(indexes)
        classes[manifest["dataset"]["split"]] = {event["payload"]["is_target"] for event in flash_rows(events)}
        assert "target_symbol" not in events[1]["payload"]
        raw = scipy.io.loadmat(folder / "S0201-P300_classic.mat")
        assert eeg[0, 10] == raw["epochs"][indexes[0], 0, 10]
    assert [path.name for path in written] == ["samara_S0201_train", "samara_S0201_val", "samara_S0201_test"]
    assert sorted(found) == list(range(40))
    assert classes["train"] == {True, False} and classes["test"] == {True, False}


def test_train_eval_protocols(tmp_path):
    _bci_mat(tmp_path / "Subject_A_Train.mat", phrase="AAAA")
    train = import_bci3(tmp_path / "Subject_A_Train.mat", tmp_path / "train", subject="A")
    held = import_bci3(tmp_path / "Subject_A_Train.mat", tmp_path / "held", subject="A")
    model = tmp_path / "model.json"
    P300FlashDecoder(model, channels=["Ch1"]).fit([train])
    record = evaluate(held, model)
    assert record["protocol"] == "within" and record["flash"]["n"] > 0
    assert record["flash"]["itr_unit"] == "bits_per_decision" and len(record["flash"]["accuracy_ci95"]) == 2
    assert record["character"]["itr_unit"] == "bits_per_min"
    saved = json.loads(model.read_text(encoding="utf-8"))
    saved["weights"] = [0.0] * len(saved["weights"])
    saved["bias"] = -1
    (tmp_path / "negative.json").write_text(json.dumps(saved), encoding="utf-8")
    blank = evaluate(held, tmp_path / "negative.json")
    assert blank["flash"]["f1"] is None and blank["flash"]["majority_class"] is True

    subjects = []
    for subject in ("S1", "S2", "S3"):
        _samara_mat(tmp_path / subject, subject)
        subjects.append(import_samara(tmp_path / subject, tmp_path / f"out_{subject}", subject=subject)[0])
    transfer = tmp_path / "transfer.json"
    P300FlashDecoder(transfer).fit([subjects[0]])
    assert evaluate(subjects[1], transfer)["protocol"] == "transfer"
    loso = tmp_path / "loso.json"
    P300FlashDecoder(loso).fit(subjects[:2])
    assert evaluate(subjects[2], loso)["protocol"] == "loso"
    with pytest.raises(ValueError, match="overlap"):
        evaluate(subjects[0], transfer)
