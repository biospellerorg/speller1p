"""Lazy local Hugging Face causal model adapter."""
import hashlib
import re
from eeg_speller.llm.char_marginal import marginalize_token_paths
from eeg_speller.core.distributions import SymbolDistribution, log_probs


class HFCausal:
    """Source Model (Fast LLM); NEED-F-06; MA-08/09; Q-13."""
    version = "transformers"

    def __init__(self, cfg):
        # IMPROV[Q-13]
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        self.cfg = cfg
        self.model_id = cfg["model_id"]
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, local_files_only=True)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_id, local_files_only=True)
        self.model.eval()
        digest = hashlib.sha256()
        for name, tensor in sorted(self.model.state_dict().items()):
            digest.update(name.encode())
            digest.update(tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
        self.weights_hash = digest.hexdigest()

    def _beam_paths(self, context):
        match = re.search(r"[а-яё]+$", context.lower())
        prefix = match.group().replace("ё", "е") if match else ""
        healed = context[:-len(prefix)] if prefix else context
        ids = self.tokenizer(healed or " ", return_tensors="pt").input_ids
        beams = [(ids, "", 1.0)]
        width = int(self.cfg["char_marginal"]["beam"])
        depth = int(self.cfg["char_marginal"]["depth"])
        for _ in range(depth):
            expanded = []
            for token_ids, decoded, mass in beams:
                with self.torch.no_grad():
                    logits = self.model(token_ids).logits[0, -1]
                probs = self.torch.softmax(logits, dim=-1)
                top = self.torch.topk(probs, min(width, len(probs)))
                for index, prob in zip(top.indices, top.values):
                    piece = self.tokenizer.decode([int(index)], skip_special_tokens=True,
                                                  clean_up_tokenization_spaces=False)
                    continuation = (decoded + piece).lower().replace("ё", "е")
                    if not piece or (prefix and not (prefix.startswith(continuation) or continuation.startswith(prefix))):
                        continue
                    extended = self.torch.cat((token_ids, index.reshape(1, 1)), dim=1)
                    expanded.append((extended, continuation, mass * float(prob)))
            if not expanded:
                break
            beams = sorted(expanded, key=lambda item: -item[2])[:width]
        return prefix, [(decoded, mass) for _, decoded, mass in beams]

    def predict(self, context, repertoire):
        # ASSUMPTION[MA-09]
        # This adapter requires full vocabulary logits; it never uses top-k API logprobs.
        prefix, paths = self._beam_paths(context)
        p = marginalize_token_paths(paths, prefix, repertoire.symbols,
                                    self.cfg["char_marginal"]["backspace_prob"])
        return SymbolDistribution(repertoire, log_probs(p), f"fast_llm:{self.model_id}", len(context))

    def top_words(self, context, n):
        from collections import defaultdict
        prefix, paths = self._beam_paths(context)
        masses = defaultdict(float)
        for decoded, mass in paths:
            if not decoded.startswith(prefix):
                continue
            rest = decoded[len(prefix):]
            end = re.search(r"[ .,!?]", rest)
            if end:
                word = prefix + rest[:end.start()]
                if word and re.fullmatch(r"[а-я]+", word):
                    masses[word] += mass
        total = sum(masses.values())
        if not total:
            return []
        return [(word, mass / total) for word, mass in sorted(masses.items(), key=lambda item: -item[1])[:n]]
