"""Explicit placeholder for external physiology integration."""


class StubLivePhysiology:
    """Physiology Module placeholder; NEED-F-04; MA-03; Q-01."""
    def start_epoch(self, msg):
        # STUB[Q-01]
        raise NotImplementedError("Q-01: external physiology protocol is not connected")

    get_observation = start_epoch
    receive_feedback = start_epoch
    receive_update = start_epoch
