"""The four stage workers, each started once in its own venv and reused across jobs."""
import os

from ..paths import LYP, REPO
from .client import WorkerClient

VENVS = {"mri": ".venv-mri", "dbn": ".venv-deepbrainnet", "streams": ".venv-fusion", "kan": ".venv-kan"}
TIMEOUT = {"mri": 900, "dbn": 900, "streams": 300, "kan": 300}


class WorkerPool:
    def __init__(self, log_dir):
        os.makedirs(log_dir, exist_ok=True)
        self.log_dir = log_dir
        self._w = {}

    def __getattr__(self, name):
        if name not in VENVS:
            raise AttributeError(name)
        if name not in self._w:
            self._w[name] = WorkerClient(str(LYP / VENVS[name] / "bin" / "python"),
                                         f"src.assessment.workers.{name}_worker", str(REPO),
                                         os.path.join(self.log_dir, f"{name}.log"), TIMEOUT[name],
                                         env={"CUDA_VISIBLE_DEVICES": ""} if os.environ.get("AD_FORCE_CPU") else None)
        return self._w[name]

    def release(self, *names):
        """Stop the named workers so their memory goes back to the OS. They are started again lazily on next use
        (models reload). Names that are not running are ignored; an unknown name raises."""
        for n in names:
            if n not in VENVS:
                raise ValueError(f"unknown worker {n!r}")
            w = self._w.pop(n, None)
            if w is not None:
                w.close()

    def close(self):
        for w in self._w.values():
            w.close()
        self._w.clear()
