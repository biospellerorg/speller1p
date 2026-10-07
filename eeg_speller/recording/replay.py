"""Read the physiology stream of a recorded session."""
from pathlib import Path
import json
import numpy as np
from eeg_speller.physiology.base import PhysiologyObservation
from eeg_speller.recording.storage import read_events


class ReplayPhysiology:
    """Physiology Module replay adapter; NEED-M-01; MA-03; Q-02."""
    def __init__(self, session_dir):
        self.source = Path(session_dir)
        self.manifest = json.loads((self.source / "session.json").read_text(encoding="utf-8"))
        self.events = read_events(self.source)
        self.observations = [e for e in self.events if e["type"] == "physiology_observation"]
        self.stimuli = {}
        for event in self.events:
            if event["type"] == "stimulus_event":
                self.stimuli.setdefault(event["epoch_id"], []).append(event["payload"])
        self.targets = [e["payload"].get("sim_target_symbol") for e in self.events if e["type"] == "epoch_started"]
        self.index = 0
        self.feedback = []
        self.updates = []

    def start_epoch(self, msg):
        self.msg = msg

    def get_observation(self):
        event = self.observations[self.index]
        self.index += 1
        p = event["payload"]
        return PhysiologyObservation(self.msg.epoch_id, np.asarray(p["log_p"]),
                                     p["semantics"], [np.asarray(x) for x in p["per_repetition"]] if p.get("per_repetition") else None,
                                     self.stimuli.get(event["epoch_id"]), p.get("module_metrics"),
                                     p.get("t_module_ns"), p.get("sim_flash_scores"))

    def receive_feedback(self, fb):
        self.feedback.append(fb)

    def receive_update(self, batch):
        self.updates.append(batch)
