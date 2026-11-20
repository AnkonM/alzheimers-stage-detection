"""Memory floor for the batch runner. SynthSeg needs about 13 to 14 GiB resident (measured: 12.62 GiB by /usr/bin/time -v
on one ADNI scan; up to 13.61 GiB sampled over a 10-scan ADNI soak; peak memory varies by scan), so before every SynthSeg
call the runner checks that enough memory is available and otherwise stops cleanly. Messages carry sizes only, never
subject identifiers, paths or scores."""

# Largest measured SynthSeg resident peak: 13,940 MiB = 13.61 GiB (10-scan ADNI soak, VM with 19,999 MB) + 1.5 GiB margin,
# rounded up to 0.1 GiB. The first single-scan measurement (12.62 GiB) gave 14.5; the soak showed that was too low.
SYNTHSEG_PEAK_GB = 13.61
DEFAULT_FLOOR_GB = 15.2


class MemoryFloorError(RuntimeError):
    """MemAvailable is below the configured floor. The message contains sizes only."""

    def __init__(self, available_gb, floor_gb):
        super().__init__(f"MemAvailable {available_gb:.1f} GiB is below the floor {floor_gb:.1f} GiB")
        self.available_gb, self.floor_gb = available_gb, floor_gb


def mem_available_gb():
    """MemAvailable from /proc/meminfo in GiB."""
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) / 1048576.0
    raise OSError("MemAvailable not found in /proc/meminfo")


def floor_check(floor_gb=DEFAULT_FLOOR_GB, probe=None):
    """A callable that raises MemoryFloorError when the probe reads below the floor. floor_gb <= 0 disables it."""
    def check():
        if floor_gb and floor_gb > 0:
            a = (probe or mem_available_gb)()
            if a < floor_gb:
                raise MemoryFloorError(a, floor_gb)
    return check
