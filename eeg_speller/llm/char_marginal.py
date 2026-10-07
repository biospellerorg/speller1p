"""Token-prefix marginalization adapter for causal language models."""
from collections import defaultdict


def marginalize_token_paths(paths, prefix: str, symbols: tuple[str, ...], backspace_prob: float = 0.02):
    """Aggregate beam paths (decoded continuation, probability) by next character.

    The caller must heal the trailing partial token and provide beam paths from
    the preceding token boundary. This function refuses an empty beam.
    """
    # ASSUMPTION[MA-09]
    if not paths:
        raise ValueError("token beam is empty")
    masses = defaultdict(float)
    for decoded, probability in paths:
        decoded = decoded.lower().replace("ё", "е")
        if not decoded.startswith(prefix) or len(decoded) <= len(prefix):
            continue
        char = decoded[len(prefix)]
        if char in symbols and char != "⌫":
            masses[char] += probability
    total = sum(masses.values())
    if total <= 0:
        raise ValueError("beam has no repertoire mass")
    return [(1 - backspace_prob) * masses[s] / total if s != "⌫" else backspace_prob for s in symbols]
