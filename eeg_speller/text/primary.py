"""Uncorrected text as chosen by the symbol detector."""


class PrimaryText:
    """Primary Text; NEED-F-01/05; MA-15; Q-16."""
    def __init__(self):
        self.value = ""

    def apply(self, symbol: str) -> str:
        if symbol == "⌫":
            self.value = self.value[:-1]
        else:
            self.value += symbol
        return self.value
