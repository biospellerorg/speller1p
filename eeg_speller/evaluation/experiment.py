"""Paired simulation of with/without language model conditions."""
from copy import deepcopy
from pathlib import Path
import json
from eeg_speller.core.config import Config
from eeg_speller.epoch.loop import EpochRunner, read_tasks
from eeg_speller.evaluation.metrics import save_session_metrics
from eeg_speller.evaluation.stats import paired_growth, summary


def experiment(cfg: Config):
    # IMPROV[Q-26]
    # IMPROV[Q-19]
    settings = cfg.model_dump().get("experiment", {})
    participants = int(settings.get("participants", 10))
    task_ids = settings.get("task_ids", [cfg.task_id])
    task_map = read_tasks()
    output = Path(settings.get("output", "runs/compare_llm"))
    paired = []
    all_sessions = []
    for i in range(participants):
        entries = {"participant_id": f"SIM-{i+1:03d}", "on": [], "off": []}
        for task_id in task_ids:
            for condition in (("on", "off") if i % 2 == 0 else ("off", "on")):
                data = deepcopy(cfg.model_dump())
                data["participant_id"] = entries["participant_id"]
                data["task_id"] = task_id
                data["seed"] = cfg.seed + i * 1000 + int(task_id[1:])
                data["mode"]["llm"] = condition
                data["experiment_id"] = "paired_compare"
                run_cfg = Config.model_validate(data)
                session = EpochRunner(run_cfg, output).run(task_map[task_id])
                metrics = save_session_metrics(session)
                entries[condition].append(metrics)
                all_sessions.append(str(session))
        paired.append(entries)
    modes = {}
    for condition in ("on", "off"):
        modes[condition] = {}
        for key in ("binary_flash_accuracy", "binary_mean_accuracy", "symbol_accuracy", "cer", "speed_primary_cpm", "speed_secondary_cpm", "word_recovery_errors", "semantic_chrf_stub"):
            per_participant = []
            for row in paired:
                values = [m[key] for m in row[condition] if m[key] is not None]
                if values:
                    per_participant.append(sum(values) / len(values))
            modes[condition][key] = summary(per_participant, cfg.stats["confidence"])
    on = [sum(m["speed_primary_cpm"] for m in row["on"]) / len(row["on"]) for row in paired]
    off = [sum(m["speed_primary_cpm"] for m in row["off"]) / len(row["off"]) for row in paired]
    growth = paired_growth(on, off, cfg.stats["bootstrap_n"], cfg.seed)
    thresholds = {"REQ-PERF-01": (modes["on"]["binary_mean_accuracy"], 0.8),
                  "REQ-PERF-02": (modes["off"]["speed_primary_cpm"], 5),
                  "REQ-PERF-03": (modes["on"]["speed_primary_cpm"], 6)}
    thresholds["REQ-LLM-01"] = (modes["on"]["word_recovery_errors"], 0.8)
    checks = {name: {"mean_pass": value["mean"] is not None and value["mean"] >= limit,
                     "strict_ci_pass": value["ci95"] is not None and value["ci95"][0] >= limit,
                     "threshold": limit, "simulation_only": True} for name, (value, limit) in thresholds.items()}
    checks["REQ-PERF-04"] = {"mean_pass": growth["growth_percent"] >= 20,
                             "strict_ci_pass": growth["bootstrap_ci95"][0] >= 20,
                             "threshold": 20}
    result = {"simulation_only": True, "not_tz_verification": True,
              "participants": participants, "task_ids": task_ids, "modes": modes,
              "growth": growth, "threshold_checks": checks, "sessions": all_sessions}
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result
