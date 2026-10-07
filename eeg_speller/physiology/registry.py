"""Selection of physiology implementations by configuration."""
from eeg_speller.physiology.simulated import SimulatedPhysiology
from eeg_speller.physiology.stub_live import StubLivePhysiology


def create(plugin: str, *args):
    # STUB[Q-02]
    if plugin == "simulated":
        return SimulatedPhysiology(*args)
    if plugin == "stub_live":
        return StubLivePhysiology()
    if plugin == "semi_synthetic":
        raise NotImplementedError("Q-02: semi-synthetic physiology is reserved")
    raise ValueError(f"unknown physiology plugin: {plugin}")
