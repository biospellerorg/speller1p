"""Shared text normalization for training and metrics."""
import re


def normalize(text: str, punctuation: bool = True) -> str:
    # ASSUMPTION[MA-01]
    text = text.lower().replace("ё", "е")
    allowed = r"а-я .," if punctuation else r"а-я "
    text = re.sub(fr"[^{allowed}]", "", text)
    return re.sub(r" +", " ", text).strip()


def words(text: str) -> list[str]:
    return normalize(text, punctuation=False).split()
