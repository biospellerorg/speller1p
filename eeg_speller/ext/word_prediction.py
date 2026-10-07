"""Top-N word prediction extension."""


class WordPrediction:
    """Word Prediction Top-N; NEED-F-08; MA-11/30; Q-08."""
    def __init__(self, fast, cfg):
        self.fast, self.cfg = fast, cfg

    def predict(self, context):
        # IMPROV[Q-08]
        prefix = context.split(" ")[-1] if context and context[-1] not in " .," else ""
        if len(prefix) < self.cfg["min_prefix"]:
            return []
        return self.fast.top_words(context, self.cfg["n_max"])
