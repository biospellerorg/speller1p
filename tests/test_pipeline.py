from copy import deepcopy
import json
from pathlib import Path
import zipfile
import numpy as np
import pytest
from eeg_speller.core.config import Config, load_config
from eeg_speller.core.distributions import Repertoire, SymbolDistribution, log_probs
from eeg_speller.epoch.fusion import Fusion
from eeg_speller.epoch.loop import EpochRunner
from eeg_speller.llm.slow import validate_correction
from eeg_speller.llm.slow import create_slow
from eeg_speller.physiology.base import PhysiologyObservation
from eeg_speller.recording.export import export
from eeg_speller.recording.replay import ReplayPhysiology
from eeg_speller.recording.storage import check_integrity, read_events


def config(**patch):
    data = load_config("configs/default.yaml").model_dump()
    for key, value in patch.items():
        data[key] = value
    return Config.model_validate(data)


def compact_events(path):
    def clean(value):
        if isinstance(value, dict):
            return {k: clean(v) for k, v in value.items()
                    if k != "session_id" and not k.startswith("wall_")}
        if isinstance(value, list):
            return [clean(v) for v in value]
        return value
    return [clean(e) for e in read_events(path)]


def test_run_replay_and_reproducibility(tmp_path):
    cfg = config(runtime={"realtime": False, "max_symbols": 5})
    first = EpochRunner(cfg, tmp_path / "first").run("привет")
    second = EpochRunner(cfg, tmp_path / "second").run("привет")
    assert compact_events(first) == compact_events(second)
    source = ReplayPhysiology(first)
    data = cfg.model_dump(); data["run_mode"] = "replay"
    replay = EpochRunner(Config.model_validate(data), tmp_path / "replay", source).run("привет")
    decisions = lambda p: [e["payload"]["symbol"] for e in read_events(p) if e["type"] == "symbol_decided"]
    assert decisions(first) == decisions(replay)
    assert check_integrity(first)["complete"]
    assert check_integrity(replay)["complete"]


def test_no_llm_is_uniform_and_has_no_correction_or_candidates(tmp_path):
    cfg = load_config("configs/no_llm.yaml")
    data = cfg.model_dump(); data["runtime"]["max_symbols"] = 5
    path = EpochRunner(Config.model_validate(data), tmp_path).run("привет")
    events = read_events(path)
    prior = next(e for e in events if e["type"] == "prior_computed")["payload"]["log_p"]
    assert np.allclose(prior, np.full(36, -np.log(36)))
    assert not any(e["type"] in {"word_candidates", "correction_requested", "correction_completed"} for e in events)


def test_fusion_rejects_double_counted_prior():
    rep = Repertoire(tuple(load_config("configs/default.yaml").repertoire["symbols"]))
    prior = SymbolDistribution(rep, log_probs(np.ones(36)), "uniform", 0)
    obs = PhysiologyObservation(0, log_probs(np.ones(36)), "posterior_with_prior")
    with pytest.raises(ValueError):
        Fusion(load_config("configs/default.yaml").fusion, "on").combine(prior, obs)


def test_timing_and_correction_validation():
    data = load_config("configs/default.yaml").model_dump()
    data["stimulus"]["flash_ms"] = 105
    with pytest.raises(ValueError):
        Config.model_validate(data)
    assert validate_correction(["два", "слова"], ["два"], True) == (False, "word_count")
    assert validate_correction(["два"], ["слово!"], True) == (False, "invalid_character")
    model = create_slow(load_config("configs/default.yaml").slow_llm, False, "data/corpus_ru_fallback.txt")
    assert model.correct("", ["сосет"]) == ["сосед"]


def test_two_paradigms_and_export(tmp_path):
    data = load_config("configs/default.yaml").model_dump()
    data["stimulus"]["paradigm"] = "single_symbol"
    data["epoch"]["n_repetitions"] = 1
    data["runtime"]["max_symbols"] = 1
    path = EpochRunner(Config.model_validate(data), tmp_path).run("я")
    assert sum(e["type"] == "stimulus_event" for e in read_events(path)) == 36
    archive = export(path)
    with zipfile.ZipFile(archive) as zf:
        names = zf.namelist()
        assert any(name.endswith("events.jsonl") for name in names)
        assert any(name.endswith("derived/integrity.json") for name in names)
        assert any(name.endswith("checksums.sha256") for name in names)
    assert check_integrity(path)["complete"]
