"""Peak GPU memory per GPU and peak host RAM of this job, sampled in a background thread of the RLinf driver.

GPU memory: `nvidia-smi --query-gpu=memory.used` (every process on the job's GPUs, including the EGL
renderers of the simulators). Host RAM: the job's cgroup (Slurm), working set = memory.current minus
inactive_file, the figure Ray's memory monitor uses, against memory.max; without a cgroup limit (a
workstation), the whole machine's used/total RAM. `take_peaks()` returns the peaks since the last call.
"""

import subprocess
import sys
import threading
import time
from pathlib import Path

GIB = 2**30


def job_cgroup() -> Path | None:
    """The cgroup v2 directory of the Slurm job (job_<id>) or, failing that, of this process; None if unlimited."""
    try:
        relative = Path("/proc/self/cgroup").read_text().strip().split("::", 1)[1].lstrip("/")
    except (OSError, IndexError):
        return None
    path = Path("/sys/fs/cgroup") / relative
    for candidate in [path, *path.parents]:
        if candidate.name.startswith("job_") and (candidate / "memory.max").is_file():
            return candidate
    limited = (path / "memory.max").is_file() and (path / "memory.max").read_text().strip() != "max"
    return path if limited else None


def host_ram(cgroup: Path | None) -> tuple[float, float]:
    """(used GiB, limit GiB) of the job, or of the machine without a cgroup limit."""
    if cgroup is None:
        import psutil

        memory = psutil.virtual_memory()
        return (memory.total - memory.available) / GIB, memory.total / GIB
    stat = dict(line.split() for line in (cgroup / "memory.stat").read_text().splitlines())
    used = int((cgroup / "memory.current").read_text()) - int(stat.get("inactive_file", 0))
    return used / GIB, int((cgroup / "memory.max").read_text()) / GIB


def gpu_memory() -> list[float]:
    """Used memory of every visible GPU in GiB, by nvidia-smi index."""
    out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True, check=True, timeout=30).stdout
    return [float(line) / 1024 for line in out.split()]


class ResourceSampler:
    """Samples every `interval` seconds; `take_peaks()` returns and resets the peaks."""

    def __init__(self, interval: float = 2.0):
        self.interval, self.cgroup, self.lock = interval, job_cgroup(), threading.Lock()
        self.ram_limit = host_ram(self.cgroup)[1]
        self.gpu_peaks: list[float] = []
        self.ram_peak = 0.0
        self.errors = 0
        threading.Thread(target=self._run, name="a2u-resources", daemon=True).start()

    def _run(self) -> None:
        while True:
            try:
                gpus, (ram, _) = gpu_memory(), host_ram(self.cgroup)
                with self.lock:
                    self.gpu_peaks = [max(a, b) for a, b in zip(gpus, self.gpu_peaks or gpus)]
                    self.ram_peak = max(self.ram_peak, ram)
            except (OSError, subprocess.SubprocessError, ValueError) as error:  # counted and logged; sampling continues
                self.errors += 1
                sys.stderr.write(f"[a2u resources] WARNING: sampling failed: {error!r}\n")
            time.sleep(self.interval)

    def take_peaks(self) -> dict[str, float]:
        with self.lock:
            gpu_peaks, ram_peak, errors = self.gpu_peaks, self.ram_peak, self.errors
            self.gpu_peaks, self.ram_peak, self.errors = [], 0.0, 0
        metrics = {f"resources/peak_gpu_mem_gib/gpu{i}": value for i, value in enumerate(gpu_peaks)}
        if gpu_peaks:
            metrics["resources/peak_gpu_mem_gib/max"] = max(gpu_peaks)
        metrics |= {"resources/peak_host_ram_gib": ram_peak, "resources/host_ram_limit_gib": self.ram_limit,
                    "resources/peak_host_ram_frac": ram_peak / self.ram_limit, "resources/sampling_errors": errors}
        return metrics
