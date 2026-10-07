from pathlib import Path

from eeg_speller.recording.storage import check_integrity, read_events
from eeg_speller.web.engine import InteractiveSession


def test_keyboard_hits_decode_symbol_and_keep_session_integrity(tmp_path: Path):
    session = InteractiveSession("gui_slow.yaml", "а", tmp_path)
    assert len(session.plan) == 24
    hits = [0 in flash["indices"] for flash in session.plan]
    state = session.submit(hits)
    assert state["text"] == "а"
    assert state["finished"]
    assert check_integrity(session.storage.directory)["complete"]
    events = read_events(session.storage.directory)
    assert sum(event["type"] == "stimulus_event" for event in events) == 24
    assert all(event["payload"]["source"] == "keyboard_space"
               for event in events if event["type"] == "physiology_observation")


def test_no_keyboard_response_abstains_and_unlisted_config_is_rejected(tmp_path: Path):
    from pytest import raises
    with raises(ValueError):
        InteractiveSession("../default.yaml", "", tmp_path)
    session = InteractiveSession("gui_slow.yaml", "а", tmp_path)
    state = session.submit([False] * len(session.plan))
    assert state["text"] == ""
    assert state["retry"] == 1
    assert not state["finished"]
    session.stop()
    assert check_integrity(session.storage.directory)["complete"]
