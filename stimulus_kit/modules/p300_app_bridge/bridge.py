"""Translate events of the external P300-LLM program (p300_app) into speller-link@2.

p300_app draws the screen, reads the amplifier and decodes EEG by itself. It does not
listen to the speller. Every event it produces is appended to
``<p300_app>/data/session_*/events.jsonl`` immediately, with ``lsl_timestamp`` taken in
the LSL clock right after ``display.flip()``. This module follows that file and replays
the letters into the speller's epochs:

    target_started          -> take the speller's next epoch
    stimulus_started        -> link.flash(symbols, timestamp=lsl_timestamp)   (rows/columns)
    decoding_prediction     -> link.send_likelihood(from top-3, final=True); link.done()
    interrupted / error     -> link.send_likelihood(usable=False); link.done()

The program is not changed and gets no feedback: it keeps its own target text, cue,
number of repetitions and its own decoded text. Both programs must run on the same
computer (same file system, same LSL clock).
"""
from __future__ import annotations

import json
import time
from pathlib import Path

SPECIAL = {"space": "_", "backspace": "⌫"}       # p300_app item value -> speller symbol
ABORT_EVENTS = ("target_sequence_interrupted", "participant_runtime_error",
                "paradigm_abort_requested", "participant_runtime_finished")


def keyboard_from_config(config: dict) -> tuple[list[str], tuple[int, int], dict[str, str]]:
    """Keyboard of p300_app's flashing paradigm.

    ``config`` is either the whole ``config/defaults.json`` / ``config_snapshot.json``
    or its ``participant_runtime`` block. Returns (symbols, (rows, cols), label -> symbol).
    Standalone buttons ("ЗАВЕРШИТЬ") are not keyboard symbols and are left out.
    """
    runtime = config.get("participant_runtime", config)
    flashing = runtime["paradigms"]["flashing"]
    rows, cols = int(flashing["num_rows"]), int(flashing["num_cols"])
    symbols, by_label = [], {}
    for item in flashing["matrix_items"]:
        value = str(item.get("value", item["label"]))
        symbol = SPECIAL.get(value, value)
        if len(symbol) != 1:
            raise ValueError(f"p300_app item {item.get('id')!r} has value {value!r}: "
                             "add it to SPECIAL in bridge.py as a one-character symbol")
        symbols.append(symbol)
        by_label[str(item["label"])] = symbol
        by_label[value] = symbol
    if len(set(symbols)) != len(symbols):
        raise ValueError("p300_app keyboard has repeated symbols")
    return symbols, (rows, cols), by_label


def item_symbol(item: dict, by_label: dict[str, str]) -> str | None:
    value = str(item.get("value", item.get("label", "")))
    return by_label.get(value) or by_label.get(str(item.get("label", "")))


def likelihood_from_top3(top3: list[str], symbols: list[str], by_label: dict[str, str],
                         weights=(8.0, 3.0, 2.0)) -> dict[str, float]:
    """p300_app reports only its three best cells, so the evidence is coarse:
    the k-th best symbol gets ``weights[k]``, every other symbol 1."""
    p = {s: 1.0 for s in symbols}
    for weight, label in zip(weights, top3):
        symbol = by_label.get(str(label))
        if symbol in p:
            p[symbol] = max(p[symbol], float(weight))
    return p


class EventsTail:
    """Follow the newest ``session_*`` folder of p300_app and return new complete events.

    What is already written when the module starts is history, not live input: the
    session folder that exists at start is read from its current end. (LSL clocks
    restart with the computer, so timestamps alone cannot tell old events from new.)
    Session folders that appear later are read from the beginning.
    """

    def __init__(self, data_dir: Path, log=print, skip_existing: bool = True):
        self.data_dir = Path(data_dir)
        self.log = log
        self.session: Path | None = None
        self.offset = 0
        self.partial = b""
        self.history: Path | None = None
        if skip_existing:
            self.history = self.newest_session()
            if self.history is not None:
                self.session = self.history
                try:
                    self.offset = (self.history / "events.jsonl").stat().st_size
                except OSError:
                    self.offset = 0
                log(f"p300_app session {self.history.name} already exists: only new events are read")

    def newest_session(self) -> Path | None:
        if not self.data_dir.is_dir():
            return None
        sessions = sorted(p for p in self.data_dir.glob("session_*") if p.is_dir())
        return sessions[-1] if sessions else None

    def poll(self) -> tuple[Path | None, list[tuple[str, dict]]]:
        """(session folder if it just changed else None, new events)."""
        changed = None
        newest = self.newest_session()
        if newest is not None and newest != self.session:
            self.session, self.offset, self.partial = newest, 0, b""
            changed = newest
            self.log(f"p300_app session: {newest.name}")
        if self.session is None:
            return changed, []
        path = self.session / "events.jsonl"
        try:
            with path.open("rb") as f:
                f.seek(self.offset)
                chunk = f.read()
        except OSError:
            return changed, []
        self.offset += len(chunk)
        data = self.partial + chunk
        lines = data.split(b"\n")
        self.partial = lines.pop()            # an unfinished last line is read next time
        events = []
        for line in lines:
            if not line.strip():
                continue
            try:
                event = json.loads(line.decode("utf-8"))
            except ValueError:
                self.log(f"p300_app: unreadable line skipped: {line[:80]!r}")
                continue
            events.append((str(event.get("event_type")), dict(event.get("payload") or {})))
        return changed, events


class Translator:
    """One p300_app letter = one speller epoch. ``link`` is a SpellerLink."""

    def __init__(self, link, symbols, by_label, start_clock: float, weights=(8.0, 3.0, 2.0),
                 decision_wait_s: float = 10.0, epoch_wait_s: float = 30.0, log=print,
                 clock=time.monotonic):
        self.link = link
        self.symbols = list(symbols)
        self.by_label = dict(by_label)
        self.start_clock = float(start_clock)
        self.weights = tuple(weights)
        self.decision_wait_s = float(decision_wait_s)
        self.epoch_wait_s = float(epoch_wait_s)
        self.log = log
        self.clock = clock
        self.current: dict | None = None
        self.letters = 0

    def set_keyboard(self, symbols, layout, by_label) -> None:
        """A new p300_app session may use another keyboard: announce it for the next epoch."""
        if list(symbols) != self.symbols:
            self.log("p300_app keyboard changed; the speller gets it at the next epoch")
        self.symbols, self.by_label = list(symbols), dict(by_label)
        self.link.keyboard = {"symbols": list(symbols), "layout": [int(layout[0]), int(layout[1])]}

    def on_event(self, kind: str, payload: dict) -> None:
        stamp = payload.get("lsl_timestamp")
        if isinstance(stamp, (int, float)) and stamp < self.start_clock:
            return                                        # happened before this module started
        if kind == "target_started" and payload.get("paradigm") == "flashing":
            self._begin(int(payload["target_index"]))
        elif kind == "target_started":
            self.log(f"p300_app paradigm {payload.get('paradigm')!r} has no decoder; ignored")
        elif kind == "stimulus_started":
            self._flash(payload)
        elif kind == "stimulation_finished":
            if self._matches(payload):
                self.current["presented_at"] = self.clock()
        elif kind == "decoding_prediction":
            if self._matches(payload):
                p = likelihood_from_top3(payload.get("top3") or [payload.get("predicted_label")],
                                         self.symbols, self.by_label, self.weights)
                self._finish(p, usable=True, info={"source": "p300_app", "backend": payload.get("backend"),
                                                   "top3": payload.get("top3"),
                                                   "p300_app_text": payload.get("text")})
        elif kind in ABORT_EVENTS:
            if self.current is not None:
                self.log(f"p300_app: {kind}; letter {self.current['index'] + 1} has no decision")
                self._finish(None, usable=False, info={"source": "p300_app", "reason": kind})

    def tick(self) -> None:
        """Give up on p300_app's decision some seconds after the last flash."""
        cur = self.current
        if cur and cur.get("presented_at") is not None and self.clock() - cur["presented_at"] > self.decision_wait_s:
            self.log(f"p300_app: no decision for letter {cur['index'] + 1} "
                     f"within {self.decision_wait_s:.0f} s")
            self._finish(None, usable=False, info={"source": "p300_app", "reason": "no_decision"})

    # ------------------------------------------------------------------ steps
    def _matches(self, payload) -> bool:
        return self.current is not None and payload.get("target_index") == self.current["index"]

    def _begin(self, index: int) -> None:
        if self.current is not None:
            self._finish(None, usable=False, info={"source": "p300_app", "reason": "next_letter_started"})
        epoch = self.link.wait_epoch(timeout=self.epoch_wait_s)
        if epoch is None:
            self.log(f"p300_app letter {index + 1}: the speller has no epoch "
                     f"(paused, ended or waiting > {self.epoch_wait_s:.0f} s); letter skipped")
            return
        self.current = {"index": index, "epoch": epoch, "flashes": 0, "presented_at": None}
        self.letters += 1
        self.log(f"p300_app letter {index + 1} -> speller epoch {epoch.id}")

    def _flash(self, payload: dict) -> None:
        if not self._matches(payload) or payload.get("stimulus_type") not in ("row", "column"):
            return                                        # standalone buttons are not symbols
        symbols = [s for s in (item_symbol(i, self.by_label) for i in payload.get("items", [])) if s]
        if not symbols:
            return
        self.link.flash(symbols, timestamp=float(payload["lsl_timestamp"]), epoch=self.current["epoch"].id)
        self.current["flashes"] += 1

    def _finish(self, p, usable: bool, info: dict) -> None:
        cur, self.current = self.current, None
        eid = cur["epoch"].id
        values = p if p is not None else {s: 1.0 for s in self.symbols}
        self.link.send_likelihood(values, final=True, epoch=eid, usable=usable, info=info)
        self.link.done(epoch=eid)
