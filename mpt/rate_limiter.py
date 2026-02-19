import time
import threading
from collections import deque

class SlidingWindowRateLimiter:
    """Thread-safe sliding-window limiter: max_calls within period_sec."""

    def __init__(self, max_calls: int, period_sec: float = 60.0):
        self.max_calls = int(max_calls)
        self.period_sec = float(period_sec)
        self._lock = threading.Lock()
        self._calls = deque()

    def acquire(self):
        while True:
            with self._lock:
                now = time.monotonic()
                # drop old calls
                cutoff = now - self.period_sec
                while self._calls and self._calls[0] <= cutoff:
                    self._calls.popleft()

                if len(self._calls) < self.max_calls:
                    self._calls.append(now)
                    return

                # need to wait until earliest call falls out of window
                sleep_for = (self._calls[0] + self.period_sec) - now
                sleep_for = max(sleep_for, 0.01)

            time.sleep(sleep_for)
