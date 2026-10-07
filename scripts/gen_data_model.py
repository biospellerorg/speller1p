"""Check event catalog in code against DATA_MODEL section 6."""
from pathlib import Path
import re
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eeg_speller.core.events import EVENT_TYPES

ROOT = Path(__file__).resolve().parents[1]


def check():
    text = (ROOT / "docs" / "DATA_MODEL.md").read_text(encoding="utf-8")
    section = text.split("## 6. Каталог событий", 1)[1].split("## 7.", 1)[0]
    documented = set()
    for line in section.splitlines():
        if not line.startswith("| `"):
            continue
        first = line.split("|", 2)[1]
        documented.update(re.findall(r"`([a-z_]+)`", first))
    return {"missing_in_docs": sorted(EVENT_TYPES - documented),
            "missing_in_code": sorted(documented - EVENT_TYPES)}


if __name__ == "__main__":
    result = check()
    print(result)
    sys.exit(bool(result["missing_in_docs"] or result["missing_in_code"]))
