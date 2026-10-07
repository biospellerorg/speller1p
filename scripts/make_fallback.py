"""Create an original, deterministic offline Russian baseline corpus."""
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
subjects = ["сосед", "покупатель", "пассажир", "садовник", "учитель", "курьер", "мастер", "музыкант", "дежурный", "библиотекарь", "повар", "художник"]
times = ["утром", "днем", "вечером", "после работы", "перед обедом", "в свободное время", "на выходных", "после прогулки"]
places = ["в доме", "во дворе", "на кухне", "в мастерской", "в библиотеке", "в парке", "на станции", "в магазине"]
actions = ["проверяет расписание", "готовит простой ужин", "читает новую книгу", "обсуждает планы", "поливает цветы", "собирает вещи", "выбирает удобный маршрут", "слушает тихую музыку", "записывает полезные советы", "открывает окно"]
sentences = []
for i, subject in enumerate(subjects):
    for j, action in enumerate(actions):
        for k, time in enumerate(times):
            place = places[(i + j + k) % len(places)]
            sentences.append(f"{time} {subject} {action} {place}.")
            sentences.append(f"{subject} {action} {place}, когда появляется свободная минута.")
            sentences.append(f"в этот день {subject} спокойно {action} {place}.")
tasks = {line.split("\t", 1)[1].strip() for line in (ROOT / "data" / "tasks_ru.txt").read_text(encoding="utf-8").splitlines() if line and not line.startswith("#")}
sentences = [s for s in sentences if s not in tasks]
text = "\n".join(sentences) + "\n"
assert len(text.encode("utf-8")) >= 50_000
(ROOT / "data" / "corpus_ru_fallback.txt").write_text(text, encoding="utf-8")
print(f"wrote {len(text.encode('utf-8'))} bytes of original offline text")
