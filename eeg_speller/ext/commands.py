"""Text command interface with recorded-only actions."""


class CommandHandler:
    """Commands extension stub; NEED-F-09; Q-32."""
    def __init__(self, recorder):
        self.recorder = recorder

    def issue(self, command, text):
        # STUB[Q-32]
        if command not in {"save", "send"}:
            raise ValueError("unsupported command")
        self.recorder.emit("ext:commands", "command_issued",
                           {"command": command, "text": text, "result": "logged_only"})
