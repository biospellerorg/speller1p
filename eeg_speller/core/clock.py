"""Virtual simulation clock and monotonic live clock."""
import time


class Clock:
    def __init__(self, live: bool = False):
        self.live = live
        self.now = 0
        self.start = time.monotonic_ns() if live else 0

    def t_ns(self) -> int:
        return time.monotonic_ns() - self.start if self.live else self.now

    def advance_ms(self, ms: int | float) -> None:
        if not self.live:
            # ASSUMPTION[MA-07]
            self.now += round(ms * 1_000_000)
