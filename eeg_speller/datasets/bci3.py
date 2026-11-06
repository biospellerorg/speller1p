from __future__ import annotations

from pathlib import Path

import numpy as np
import scipy.io

from eeg_speller.datasets.session import write_import_session

BCI3_FS = 240.0
BCI3_CHARS = tuple("ABCDEFGHIJKLMNOPQRSTUVWXYZ123456789_")
# Published test keys. Subject B is one character longer than the 100-character file.
TEST_PHRASES = {
    "A": "WQXPLZCOMRKO97YFZDEZ1DPI9NNVGRQDJCUVRMEUOOOJD2UFYPOO6J7LDGYEGOA5VHNEHBTXOO1TDOILUEE5BFAEEXAW_K4R3MRU",
    "B": "MERMIROOMUHJPXJOHUVLEORWRJBTJGGCIFSCGGLFSLEZWNPMPKEZSYSENDWPBDUUJIXZETUZCWOJQQRTVXBMZWHSWPRQDZEQRCOQZ",
}
WINDOW_S = (0.0, 0.8)


def flash_onsets(flashing) -> np.ndarray:
    flash = np.asarray(flashing).ravel()
    changes = np.where(flash[:-1] != flash[1:])[0]
    rises = changes[1::2] + 1
    return np.concatenate([[0], rises])


def code_membership(code: int) -> tuple[str, int, list[int]]:
    if 1 <= code <= 6:
        column = code - 1
        return "column", column, [column + row * 6 for row in range(6)]
    if 7 <= code <= 12:
        row = code - 7
        return "row", row, list(range(row * 6, row * 6 + 6))
    raise ValueError(f"stimulus code {code} is outside 1..12")


def target_is_on(code: int, character: str) -> bool:
    kind, index, _ = code_membership(code)
    cell = BCI3_CHARS.index(character)
    if kind == "column":
        return cell % 6 == index
    return cell // 6 == index


def _as_chars(value) -> str:
    array = np.asarray(value).reshape(-1)
    if array.dtype.kind in "SU":
        return "".join(str(item) for item in array)
    if array.size == 1:
        return str(array.reshape(-1)[0])
    return "".join(str(item) for item in array)


def _channel_names(eloc_path: Path | None, n_channels: int) -> list[str]:
    if eloc_path is None or not eloc_path.is_file():
        return [f"Ch{i + 1}" for i in range(n_channels)]
    names = []
    for line in eloc_path.read_text(encoding="utf-8", errors="replace").splitlines():
        parts = line.split()
        # eloc64.txt is "index angle radius Name" (for example "51 180 0.181 Pz..").
        if parts and parts[-1][:1].isalpha():
            names.append(parts[-1].strip("."))
    if len(names) < n_channels:
        names.extend(f"Ch{i + 1}" for i in range(len(names), n_channels))
    return names[:n_channels]


def load_bci3(path: Path, *, subject: str | None = None, eloc: Path | None = None):
    raw = scipy.io.loadmat(path)
    signal = np.asarray(raw["Signal"], dtype=np.float64)
    if signal.ndim != 3:
        raise ValueError(f"{path.name}: Signal must have shape (characters, time, channels)")
    n_chars, _n_times, n_channels = signal.shape
    flashing = np.asarray(raw["Flashing"])
    codes = np.asarray(raw["StimulusCode"])
    if flashing.shape[0] != n_chars or codes.shape[0] != n_chars:
        raise ValueError(f"{path.name}: Flashing and StimulusCode must have one row per character")
    who = subject or _subject_from_name(path.name)
    if "StimulusType" in raw:
        types = np.asarray(raw["StimulusType"])
        phrase = list(_as_chars(raw["TargetChar"]))[:n_chars] if "TargetChar" in raw else [""] * n_chars
        labeled = "train"
    else:
        types = None
        phrase = list(TEST_PHRASES[who][:n_chars])
        labeled = "test"
    if len(phrase) < n_chars:
        raise ValueError(f"{path.name}: target phrase is shorter than the recording")
    return {
        "signal": signal, "flashing": flashing, "codes": codes, "types": types,
        "phrase": phrase, "subject": who, "split": labeled,
        "channels": _channel_names(eloc, n_channels),
    }


def _subject_from_name(name: str) -> str:
    for token in Path(name).stem.split("_"):
        if token in {"A", "B"}:
            return token
    raise ValueError(f"cannot tell the subject from {name}; pass --subject")


def iter_epochs(loaded, *, window_s=WINDOW_S):
    fs = BCI3_FS
    n_window = int(round((window_s[1] - window_s[0]) * fs))
    signal = loaded["signal"]
    pieces = []
    epochs = []
    cursor = 0
    seen = []
    for char_i, character in enumerate(loaded["phrase"]):
        block = signal[char_i].T  # (channels, time)
        onsets = flash_onsets(loaded["flashing"][char_i])
        codes = np.asarray(loaded["codes"][char_i]).ravel()
        types = None if loaded["types"] is None else np.asarray(loaded["types"][char_i]).ravel()
        counts: dict[int, int] = {}
        flashes = []
        for onset in onsets:
            onset = int(onset)
            if onset + n_window > block.shape[1]:
                raise ValueError(f"character {char_i} flash at {onset} does not fit the {window_s} window")
            code = int(codes[onset])
            kind, index, indices = code_membership(code)
            rep = counts.get(code, 0)
            counts[code] = rep + 1
            is_target = bool(types[onset]) if types is not None else target_is_on(code, character)
            flashes.append({"is_target": is_target, "onset_sample": cursor + onset, "n_samples": n_window,
                            "kind": kind, "index": index, "indices": indices, "stimulus_code": code, "rep": rep})
            seen.append(code)
        pieces.append(block)
        epochs.append({"target_symbol": character, "flashes": flashes})
        cursor += block.shape[1]
    eeg = np.concatenate(pieces, axis=1)
    return eeg, epochs, seen


def import_bci3(path, output, *, subject=None, eloc=None) -> Path:
    path = Path(path)
    loaded = load_bci3(path, subject=subject, eloc=Path(eloc) if eloc else _eloc_near(path))
    eeg, epochs, _codes = iter_epochs(loaded)
    phrase = "".join(epoch["target_symbol"] for epoch in epochs)
    dataset = {"name": "bci3", "subject": loaded["subject"], "split": loaded["split"],
               "role": "import", "label_source": "ground_truth", "source_path": str(path),
               "fs": BCI3_FS, "channels": loaded["channels"], "unit": "unknown",
               "window_s": list(WINDOW_S), "soa_s": 0.175, "concatenated_epochs": True}
    return write_import_session(output, dataset=dataset, repertoire=list(BCI3_CHARS), paradigm="row_column",
                                eeg=eeg, fs=BCI3_FS, channels=loaded["channels"], epochs=epochs,
                                target_text=phrase)


def _eloc_near(path: Path):
    for name in ("eloc64.txt", "eloc64.loc"):
        candidate = path.parent / name
        if candidate.is_file():
            return candidate
    return None
