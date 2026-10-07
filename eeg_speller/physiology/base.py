"""Contract offered to the physiology team."""
from dataclasses import dataclass
from typing import Literal, Protocol
import numpy as np
from eeg_speller.core.distributions import Repertoire


@dataclass(frozen=True)
class StimulusTiming:
    flash_ms: int
    isi_ms: int
    pause_ms: int
    n_repetitions: int


@dataclass(frozen=True)
class EpochStart:
    epoch_id: int
    position: int
    repertoire: Repertoire
    paradigm: str
    timing: StimulusTiming


@dataclass(frozen=True)
class PhysiologyObservation:
    epoch_id: int
    log_p: np.ndarray
    semantics: Literal["likelihood", "posterior_with_prior"]
    per_repetition: list[np.ndarray] | None = None
    stimulus_events: list[dict] | None = None
    module_metrics: dict | None = None
    t_module_ns: int | None = None
    sim_flash_scores: list[list[float]] | None = None


class PhysiologyModule(Protocol):
    """Physiology Module; NEED-F-02/03/04; MA-03/04/07; Q-01/02/06."""
    def start_epoch(self, msg: EpochStart) -> None: ...
    def get_observation(self) -> PhysiologyObservation: ...
    def receive_feedback(self, fb) -> None: ...
    def receive_update(self, batch) -> None: ...
