"""Hardware, memory, and latency logging for detector eval (plan §13 E5 / §14)."""

from __future__ import annotations

import os
import platform
import subprocess
import sys
import time
from dataclasses import dataclass, field
from typing import Any


def _cpu_brand() -> str | None:
    try:
        return subprocess.check_output(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return None


def hardware_info() -> dict[str, Any]:
    info: dict[str, Any] = {
        "platform": platform.platform(),
        "system": platform.system(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python": sys.version.split()[0],
        "cpu_count": os.cpu_count(),
        "cpu_brand": _cpu_brand(),
    }
    try:
        import psutil

        mem = psutil.virtual_memory()
        info["ram_total_bytes"] = int(mem.total)
        info["ram_available_bytes"] = int(mem.available)
    except Exception:
        info["ram_total_bytes"] = None
    try:
        import torch

        info["torch"] = torch.__version__
        info["mps_built"] = bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_built())
        info["mps_available"] = bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())
        info["cuda_available"] = bool(torch.cuda.is_available())
    except Exception:
        info["torch"] = None
        info["mps_available"] = False
        info["cuda_available"] = False
    return info


def pick_device(explicit: str | None = None) -> str:
    if explicit:
        return explicit
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


def pick_dtype(device: str, explicit: str | None = None):
    import torch

    if explicit:
        mapping = {
            "fp16": torch.float16,
            "float16": torch.float16,
            "bf16": torch.bfloat16,
            "bfloat16": torch.bfloat16,
            "fp32": torch.float32,
            "float32": torch.float32,
        }
        return mapping[explicit]
    if device == "cuda":
        return torch.bfloat16
    if device == "mps":
        return torch.float16
    return torch.float32


@dataclass
class MemoryTracker:
    peak_rss_bytes: int = 0
    peak_mps_bytes: int = 0
    _proc: Any = field(default=None, repr=False)

    def __post_init__(self) -> None:
        try:
            import psutil

            self._proc = psutil.Process()
        except Exception:
            self._proc = None
        self.sample()

    def sample(self) -> None:
        if self._proc is not None:
            try:
                rss = int(self._proc.memory_info().rss)
                self.peak_rss_bytes = max(self.peak_rss_bytes, rss)
            except Exception:
                pass
        try:
            import torch

            if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                allocated = int(torch.mps.current_allocated_memory())
                self.peak_mps_bytes = max(self.peak_mps_bytes, allocated)
        except Exception:
            pass


def percentile(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    idx = (len(ordered) - 1) * (q / 100.0)
    lo = int(idx)
    hi = min(lo + 1, len(ordered) - 1)
    frac = idx - lo
    return float(ordered[lo] * (1.0 - frac) + ordered[hi] * frac)


def summarize_latencies(latencies_s: list[float]) -> dict[str, float]:
    if not latencies_s:
        return {
            "n": 0.0,
            "mean_s": float("nan"),
            "median_s": float("nan"),
            "p95_s": float("nan"),
            "min_s": float("nan"),
            "max_s": float("nan"),
        }
    return {
        "n": float(len(latencies_s)),
        "mean_s": float(sum(latencies_s) / len(latencies_s)),
        "median_s": percentile(latencies_s, 50),
        "p95_s": percentile(latencies_s, 95),
        "min_s": float(min(latencies_s)),
        "max_s": float(max(latencies_s)),
    }


class Timer:
    def __init__(self) -> None:
        self.start = time.perf_counter()

    def elapsed(self) -> float:
        return time.perf_counter() - self.start
