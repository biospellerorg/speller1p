"""Download public-domain Russian texts or use the checked-in fallback."""
import argparse
from pathlib import Path
import re
import urllib.request
import yaml

ROOT = Path(__file__).resolve().parents[1]


def normalize(text):
    text = text.lower().translate(str.maketrans({"ё": "е", "ѣ": "е", "і": "и", "ѳ": "ф", "ѵ": "и"}))
    text = re.sub(r"[^а-я .,\n]", "", text)
    return re.sub(r" +", " ", text)


def prepare(offline=False):
    tasks = {normalize(line.split("\t", 1)[1]).strip() for line in
             (ROOT / "data" / "tasks_ru.txt").read_text(encoding="utf-8").splitlines()
             if line and not line.startswith("#")}
    if offline:
        source = (ROOT / "data" / "corpus_ru_fallback.txt").read_text(encoding="utf-8")
    else:
        urls = yaml.safe_load((ROOT / "configs" / "corpus_sources.yaml").read_text(encoding="utf-8"))["urls"]
        chunks = []
        for url in urls:
            if not url.startswith("https://www.gutenberg.org/"):
                raise ValueError("only configured Gutenberg HTTPS URLs are accepted")
            with urllib.request.urlopen(url, timeout=20) as response:
                raw = response.read().decode("utf-8", errors="replace")
                start = re.search(r"\*\*\* START OF THE PROJECT GUTENBERG EBOOK .*?\*\*\*", raw, re.S | re.I)
                end = re.search(r"\*\*\* END OF THE PROJECT GUTENBERG EBOOK", raw, re.I)
                if not start or not end or end.start() <= start.end():
                    raise ValueError(f"missing book boundaries in {url}")
                chunks.append(raw[start.end():end.start()])
        source = "\n".join(chunks)
    sentences = [s.strip() for s in re.split(r"(?<=[.,])\s+", normalize(source))]
    result = "\n".join(s for s in sentences if s and s not in tasks) + "\n"
    (ROOT / "data" / "corpus_ru.txt").write_text(result, encoding="utf-8")
    return len(result.encode("utf-8"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()
    print(f"prepared {prepare(args.offline)} bytes")
