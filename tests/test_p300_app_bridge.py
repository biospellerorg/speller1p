"""Module p300_app_bridge: an unchanged external program feeds the speller through its event log."""
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "stimulus_kit" / "modules" / "p300_app_bridge"
sys.path.insert(0, str(MODULE))

from bridge import EventsTail, Translator, keyboard_from_config, likelihood_from_top3  # noqa: E402

# The format of p300_app config/defaults.json, with a 2x3 matrix instead of 5x10.
ITEMS = [{"id": "letter_Д", "label": "Д", "type": "character", "value": "Д"},
         {"id": "letter_А", "label": "А", "type": "character", "value": "А"},
         {"id": "letter_Б", "label": "Б", "type": "character", "value": "Б"},
         {"id": "letter_В", "label": "В", "type": "character", "value": "В"},
         {"id": "space", "label": "ПРОБЕЛ", "type": "action", "value": "space"},
         {"id": "backspace", "label": "УДАЛИТЬ", "type": "action", "value": "backspace"}]
CONFIG = {"participant_runtime": {"paradigms": {"flashing": {
    "num_rows": 2, "num_cols": 3, "matrix_items": ITEMS,
    "standalone_items": [{"id": "finish", "label": "ЗАВЕРШИТЬ", "type": "action", "value": "finish"}]}}}}


def public(item, row, col):
    return {**item, "layout": "matrix", "row": row, "col": col}


def letter_events(index, target, t0, sequences=2, step=0.02):
    """Events p300_app writes for one letter (see paradigms/row_column.py)."""
    cells = [public(item, i // 3, i % 3) for i, item in enumerate(ITEMS)]
    events = [("target_cue_started", {"paradigm": "flashing", "target_index": index, "wait_for_space": False,
                                      "lsl_timestamp": t0}),
              ("target_started", {"paradigm": "flashing", "target_index": index, "lsl_timestamp": t0})]
    t, position = t0, 0
    for sequence in range(sequences):
        for kind, n in (("row", 2), ("column", 3)):
            for k in range(n):
                items = [c for c in cells if c["row" if kind == "row" else "col"] == k]
                t += step
                events.append(("stimulus_started", {
                    "paradigm": "flashing", "target_index": index, "stimulus_type": kind, "stimulus_index": k,
                    "sequence_index": sequence, "sequence_position": position, "items": items,
                    "is_target": False, "lsl_timestamp": t}))
                position += 1
        events.append(("stimulus_started", {"paradigm": "flashing", "target_index": index,
                                            "stimulus_type": "standalone", "stimulus_index": 0,
                                            "items": [{"id": "finish", "label": "ЗАВЕРШИТЬ", "value": "finish"}],
                                            "lsl_timestamp": t + step / 2}))
    events += [("stimulation_finished", {"paradigm": "flashing", "target_index": index, "lsl_timestamp": t}),
               ("target_finished", {"paradigm": "flashing", "target_index": index, "lsl_timestamp": t}),
               ("decoding_prediction", {"target_index": index, "predicted_label": target,
                                        "top3": [target, "Б", "В"], "backend": "xlda", "lsl_timestamp": t + 1})]
    return events


class FakeLink:
    def __init__(self):
        self.calls, self.next_id, self.keyboard = [], 0, None

    def wait_epoch(self, timeout=None):
        self.next_id += 1
        return SimpleNamespace(id=self.next_id)

    def flash(self, symbols, timestamp=None, epoch=None):
        self.calls.append(("flash", epoch, symbols, timestamp))

    def send_likelihood(self, values, final=False, epoch=None, usable=True, info=None):
        self.calls.append(("likelihood", epoch, values, usable))

    def done(self, epoch=None):
        self.calls.append(("done", epoch))


def make(start=0.0, **kw):
    symbols, layout, by_label = keyboard_from_config(CONFIG)
    link = FakeLink()
    return link, Translator(link, symbols, by_label, start_clock=start, log=lambda m: None, **kw)


def test_keyboard_is_taken_from_p300_app_config():
    symbols, layout, by_label = keyboard_from_config(CONFIG)
    assert symbols == ["Д", "А", "Б", "В", "_", "⌫"] and layout == (2, 3)
    assert by_label["ПРОБЕЛ"] == "_" and by_label["УДАЛИТЬ"] == "⌫"
    p = likelihood_from_top3(["УДАЛИТЬ", "А"], symbols, by_label)
    assert p["⌫"] == 8 and p["А"] == 3 and p["Д"] == 1


def test_letter_becomes_epoch_with_flip_timestamps_and_decoder_evidence():
    link, tr = make()
    events = letter_events(0, "А", 100.0)
    for kind, payload in events:
        tr.on_event(kind, payload)
    flashes = [c for c in link.calls if c[0] == "flash"]
    sent = [p for k, p in events if k == "stimulus_started" and p["stimulus_type"] != "standalone"]
    assert len(flashes) == len(sent) == 10                      # standalone buttons are not sent
    assert [c[3] for c in flashes] == [p["lsl_timestamp"] for p in sent]
    assert flashes[0][2] == ["Д", "А", "Б"] and flashes[2][2] == ["Д", "В"]
    likelihood = [c for c in link.calls if c[0] == "likelihood"]
    assert len(likelihood) == 1 and likelihood[0][3] is True and likelihood[0][2]["А"] == 8
    assert link.calls[-1] == ("done", 1)


def test_events_before_the_module_started_are_ignored():
    link, tr = make(start=500.0)
    for kind, payload in letter_events(0, "А", 100.0):
        tr.on_event(kind, payload)
    assert link.calls == []


def test_missing_decision_and_interruption_send_unusable_evidence():
    now = [0.0]
    link, tr = make(decision_wait_s=5, clock=lambda: now[0])
    for kind, payload in letter_events(0, "А", 100.0)[:-1]:     # no decoding_prediction
        tr.on_event(kind, payload)
    now[0] = 6.0
    tr.tick()
    assert [c for c in link.calls if c[0] == "likelihood"][-1][3] is False
    assert link.calls[-1] == ("done", 1)
    for kind, payload in letter_events(1, "Д", 200.0)[:5]:
        tr.on_event(kind, payload)
    tr.on_event("target_sequence_interrupted", {"target_index": 1, "lsl_timestamp": 201.0})
    assert link.calls[-2][0] == "likelihood" and link.calls[-2][3] is False
    assert link.calls[-1] == ("done", 2)


def test_log_written_before_start_is_not_replayed(tmp_path):
    """After a reboot LSL clocks restart, so an old log must be skipped by position, not time."""
    old = tmp_path / "data" / "session_2026-11-08_10-00-00_000001"
    old.mkdir(parents=True)
    path = old / "events.jsonl"
    path.write_text('{"event_type": "old", "payload": {"lsl_timestamp": 99999}}\n', encoding="utf-8")
    tail = EventsTail(tmp_path / "data", log=lambda m: None)
    assert tail.poll() == (None, [])
    with path.open("a", encoding="utf-8") as f:
        f.write('{"event_type": "new", "payload": {}}\n')
    assert tail.poll() == (None, [("new", {})])


def test_tail_follows_the_newest_session_and_partial_lines(tmp_path):
    tail = EventsTail(tmp_path / "data", log=lambda m: None)
    assert tail.poll() == (None, [])
    first = tmp_path / "data" / "session_2026-11-09_10-00-00_000001"
    first.mkdir(parents=True)
    path = first / "events.jsonl"
    path.write_text('{"event_type": "a", "payload": {"x": 1}}\n{"event_type": "b", "pay', encoding="utf-8")
    changed, events = tail.poll()
    assert changed == first and events == [("a", {"x": 1})]
    with path.open("a", encoding="utf-8") as f:
        f.write('load": {}}\n')
    assert tail.poll() == (None, [("b", {})])
    second = tmp_path / "data" / "session_2026-11-09_10-05-00_000001"
    second.mkdir()
    (second / "events.jsonl").write_text('{"event_type": "c", "payload": {}}\n', encoding="utf-8")
    assert tail.poll() == (second, [("c", {})])


def test_unchanged_p300_app_types_through_the_real_speller(tmp_path, monkeypatch):
    """End to end over LSL: speller <- module <- events.jsonl written like p300_app."""
    pylsl = pytest.importorskip("pylsl")
    monkeypatch.setenv("LSLAPICFG", str(ROOT / "tests" / "lsl_local.cfg"))
    from eeg_speller.core.config import load_config
    from eeg_speller.external.controller import ExternalSession
    from eeg_speller.recording.storage import read_events

    app = tmp_path / "p300_app"
    (app / "config").mkdir(parents=True)
    (app / "config" / "defaults.json").write_text(json.dumps(CONFIG, ensure_ascii=False), encoding="utf-8")
    settings = tmp_path / "settings.json"
    settings.write_text(json.dumps({"p300_app_dir": str(app), "decision_wait_s": 5}), encoding="utf-8")

    cfg = load_config(ROOT / "configs" / "ext_default.yaml")
    cfg = cfg.model_copy(update={"stimulus": {**cfg.stimulus, "pause_ms": 200}})
    session = ExternalSession(cfg, "да", tmp_path / "runs").start()
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(ROOT / "stimulus_kit"), str(MODULE)])}
    proc = subprocess.Popen([sys.executable, "-u", str(MODULE / "main.py"), "--settings", str(settings)],
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)
    written = []

    def p300_app():                     # writes its log exactly as the unchanged program would
        time.sleep(3)
        folder = app / "data" / "session_2026-11-09_19-40-00_000001"
        folder.mkdir(parents=True)
        (folder / "config_snapshot.json").write_text(json.dumps(CONFIG, ensure_ascii=False), encoding="utf-8")
        with (folder / "events.jsonl").open("a", encoding="utf-8") as f:
            for index, letter in enumerate("ДА"):
                for kind, payload in letter_events(index, letter, pylsl.local_clock()):
                    stamp = payload["lsl_timestamp"]
                    while pylsl.local_clock() < stamp:
                        time.sleep(0.002)
                    f.write(json.dumps({"event_type": kind, "payload": payload}, ensure_ascii=False) + "\n")
                    f.flush()
                    if kind == "stimulus_started" and payload["stimulus_type"] != "standalone":
                        written.append(stamp)
                time.sleep(1.0)

    writer = threading.Thread(target=p300_app, daemon=True)
    writer.start()
    try:
        assert session.wait_finished(60), session.monitor()
    finally:
        session.stop() if hasattr(session, "stop") else None
        proc.terminate()
        output = proc.communicate(timeout=10)[0]
    state = session.monitor()
    assert state["text"] == "да", (state, output)
    received = [e["payload"]["onset_lsl_s"] for e in read_events(session.storage.directory)
                if e["type"] == "stimulus_event"]
    assert len(received) == len(written) == 20, output
    error_ms = max(abs(a - b) for a, b in zip(sorted(received), sorted(written))) * 1000
    assert error_ms < 1.0, error_ms              # p300_app's flip times reach the speller unchanged
