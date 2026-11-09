"""Внешняя программа P300-LLM (p300_app): буквы и решения её декодера идут в спеллер.

Модуль ничего не рисует. Экран испытуемого, усилитель и классификатор — у p300_app,
которую запускают как обычно (run.bat → «Инференс» → «мигания»). Модуль читает её
``data/session_*/events.jsonl`` и передаёт спеллеру клавиатуру, вспышки с метками
времени после flip и оценку декодера. Обратной связи в p300_app нет.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import pylsl

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from bridge import EventsTail, Translator, keyboard_from_config  # noqa: E402
from module_tools import load_settings, module_args  # noqa: E402
from speller_link import SpellerLink  # noqa: E402


def log(message: str) -> None:
    print(message, flush=True)


def main() -> int:
    args = module_args(__doc__.splitlines()[0], str(HERE / "settings.json"))
    cfg = load_settings(args.settings)
    app_dir = Path(os.path.expandvars(str(cfg.get("p300_app_dir", "")))).expanduser()
    if not str(cfg.get("p300_app_dir", "")).strip() or not (app_dir / "config" / "defaults.json").is_file():
        log(f"Укажите папку p300_app в settings.json (p300_app_dir); сейчас: {str(app_dir)!r}. "
            "В ней должен быть config/defaults.json.")
        return 2
    defaults = json.loads((app_dir / "config" / "defaults.json").read_text(encoding="utf-8"))
    symbols, layout, by_label = keyboard_from_config(defaults)
    log(f"p300_app: {app_dir}; клавиатура {layout[0]}×{layout[1]}, {len(symbols)} символов")

    link = SpellerLink(name="p300_app", role="stimulus+decoder", keyboard=symbols, layout=layout)
    translator = Translator(link, symbols, by_label, start_clock=pylsl.local_clock(),
                            weights=tuple(cfg.get("top3_weights", [8, 3, 2])),
                            decision_wait_s=float(cfg.get("decision_wait_s", 10)),
                            epoch_wait_s=float(cfg.get("epoch_wait_s", 30)), log=log)
    tail = EventsTail(app_dir / "data", log=log)
    poll_s = float(cfg.get("poll_ms", 20)) / 1000
    log("Жду события p300_app. Запустите в ней «Инференс» с парадигмой «мигания».")

    while not link.ended:
        changed, events = tail.poll()
        if changed is not None:
            snapshot = changed / "config_snapshot.json"
            if snapshot.is_file():
                try:
                    translator.set_keyboard(*keyboard_from_config(
                        json.loads(snapshot.read_text(encoding="utf-8"))))
                except (ValueError, KeyError) as exc:
                    log(f"p300_app: config_snapshot.json не прочитан ({exc}); клавиатура прежняя")
        for kind, payload in events:
            translator.on_event(kind, payload)
        translator.tick()
        time.sleep(poll_s)
    log(f"Сессия спеллера завершена; передано букв: {translator.letters}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
