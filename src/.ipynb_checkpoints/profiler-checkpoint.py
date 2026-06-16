import os
import time
import psutil


class RoundProfiler:
    """Measure wall time + process CPU usage (proxy)."""

    def __init__(self):
        self.proc = psutil.Process(os.getpid())
        self.t0 = None

    def start(self) -> None:
        self.proc.cpu_percent(interval=None)  # warm-up
        self.t0 = time.perf_counter()

    def stop(self) -> tuple[float, float]:
        wall_s = float(time.perf_counter() - self.t0)
        cpu_pct = float(self.proc.cpu_percent(interval=None))
        return wall_s, cpu_pct
