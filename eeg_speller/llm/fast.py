"""Fast model factory with explicit capability requirement."""
from eeg_speller.llm.backends.mock_ngram import MockNGram


def create_fast(cfg, corpus):
    backend = cfg["backend"]
    if backend == "mock_ngram":
        return MockNGram(cfg, corpus)
    if backend == "hf_causal":
        from eeg_speller.llm.backends.hf_causal import HFCausal
        return HFCausal(cfg)
    raise ValueError(f"fast backend {backend} does not provide full next-token distribution")
