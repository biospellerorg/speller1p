"""Minimal console interface; rendering is a recorded event."""


class ConsoleUI:
    """Console UI extension; NEED-F-09; Q-31/32."""
    def __init__(self, recorder, verbose=False):
        self.recorder, self.verbose = recorder, verbose

    def render(self, primary, secondary, candidates, status, epoch_id):
        # STUB[Q-31]
        self.recorder.emit("ext:ui", "ui_rendered", {"primary_text": primary,
                           "secondary_text": secondary, "candidates_shown": candidates,
                           "status": status}, epoch_id)
        if self.verbose:
            print(f"{status}: {primary} | {secondary}")
