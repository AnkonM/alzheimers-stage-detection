"""Memory safety around SynthSeg: the floor check, WorkerPool.release (real subprocesses and fakes) and the engine's
release -> check -> SynthSeg order. No scans, no models, no GPU."""
import pathlib
import sys

import pytest

from src.assessment import engine as E
from src.assessment import memguard
from src.assessment.workers import pool as P
from src.assessment.workers.client import WorkerClient

ROOT = str(pathlib.Path(__file__).resolve().parents[1])


def test_floor_check_raises_below_floor_with_sizes_only_and_passes_above():
    memguard.floor_check(14.5, lambda: 16.0)()
    memguard.floor_check(14.5, lambda: 14.5)()                      # exactly at the floor is allowed
    with pytest.raises(memguard.MemoryFloorError) as e:
        memguard.floor_check(14.5, lambda: 9.04)()
    assert str(e.value) == "MemAvailable 9.0 GiB is below the floor 14.5 GiB"
    assert (e.value.available_gb, e.value.floor_gb) == (9.04, 14.5)
    memguard.floor_check(0, lambda: 0.0)()                          # 0 disables
    memguard.floor_check(-1, lambda: 0.0)()


def test_default_floor_follows_the_measured_peak_rule_and_real_probe_reads_meminfo():
    assert memguard.SYNTHSEG_PEAK_GB == 13.61                       # 13,940 MiB, the largest sampled peak (single scan: 12.62)
    assert memguard.DEFAULT_FLOOR_GB >= memguard.SYNTHSEG_PEAK_GB + 1.5
    a = memguard.mem_available_gb()
    assert 0.1 < a < 4096


class FakeClient:
    made, closed = [], []

    def __init__(self, python, module, cwd, log_path, timeout=0, env=None):
        self.module = module
        FakeClient.made.append(module)

    def close(self):
        FakeClient.closed.append(self.module)


@pytest.fixture
def fake_pool(tmp_path, monkeypatch):
    FakeClient.made, FakeClient.closed = [], []
    monkeypatch.setattr(P, "WorkerClient", FakeClient)
    return P.WorkerPool(str(tmp_path / "logs"))


def test_release_stops_named_workers_and_they_restart_lazily(fake_pool):
    a, b, c = fake_pool.mri, fake_pool.dbn, fake_pool.streams
    assert FakeClient.made == ["src.assessment.workers.mri_worker", "src.assessment.workers.dbn_worker",
                               "src.assessment.workers.streams_worker"]
    fake_pool.release("mri", "dbn")
    assert FakeClient.closed == ["src.assessment.workers.mri_worker", "src.assessment.workers.dbn_worker"]
    assert fake_pool.streams is c                                    # untouched worker is the same object
    assert fake_pool.mri is not a and fake_pool.mri is fake_pool.mri  # a new one, created once, on first use
    assert FakeClient.made.count("src.assessment.workers.mri_worker") == 2
    assert FakeClient.made.count("src.assessment.workers.dbn_worker") == 1   # dbn not touched again: lazy


def test_release_ignores_workers_that_are_not_running_and_rejects_unknown_names(fake_pool):
    fake_pool.release("kan", "mri")                                  # nothing running: no error, nothing closed
    assert FakeClient.closed == []
    with pytest.raises(ValueError):
        fake_pool.release("synthseg")
    fake_pool.streams
    fake_pool.close()
    assert FakeClient.closed == ["src.assessment.workers.streams_worker"]


def test_release_really_stops_a_worker_process_and_a_new_one_answers(tmp_path, monkeypatch):
    def real(python, module, cwd, log_path, timeout=0, env=None):
        return WorkerClient(sys.executable, "tests.echo_worker", ROOT, log_path, timeout=5)
    monkeypatch.setattr(P, "WorkerClient", real)
    pool = P.WorkerPool(str(tmp_path / "logs"))
    w1 = pool.mri
    assert w1.call("add", a=1, b=2) == 3
    pid1 = w1._p.pid
    pool.release("mri")
    assert w1._p.poll() is not None                                  # the process is gone, its memory returned
    w2 = pool.mri
    assert w2._p.pid != pid1 and w2.call("add", a=2, b=3) == 5
    pool.close()
    assert w2._p.poll() is not None


class RecordingPool:
    def __init__(self, events):
        self.events = events

    def release(self, *names):
        self.events.append(("release", names))


def _engine(tmp_path, monkeypatch, events, *, release, hook, fail_hook=False):
    from src.assessment.volumes import synthseg
    monkeypatch.setattr(synthseg, "run", lambda native, work: events.append(("synthseg", native)) or {"volumes": {}})

    def check():
        events.append(("check",))
        if fail_hook:
            raise memguard.MemoryFloorError(3.0, 14.5)
    eng = E.AssessmentEngine(run_root=tmp_path / "runs", release_around_synthseg=release,
                             before_synthseg=check if hook else None)
    eng.pool = RecordingPool(events)
    return eng


def test_engine_releases_mri_and_dbn_then_checks_then_runs_synthseg(tmp_path, monkeypatch):
    ev = []
    eng = _engine(tmp_path, monkeypatch, ev, release=True, hook=True)
    tm = E.Timer()
    out = eng._volumes_synthseg(tm, "/x/scan.nii.gz", tmp_path / "out")
    assert out == {"volumes": {}}
    assert [e[0] for e in ev] == ["release", "check", "synthseg"] and ev[0][1] == ("mri", "dbn")
    assert "release_workers" in tm.steps and "volumes_synthseg" in tm.steps


def test_engine_defaults_change_nothing_and_a_failed_check_prevents_synthseg(tmp_path, monkeypatch):
    ev = []
    eng = _engine(tmp_path, monkeypatch, ev, release=False, hook=False)
    assert eng.release_around_synthseg is False and eng.before_synthseg is None
    tm = E.Timer()
    eng._volumes_synthseg(tm, "/x/scan.nii.gz", tmp_path / "out")
    assert [e[0] for e in ev] == ["synthseg"] and "release_workers" not in tm.steps      # default path: SynthSeg only
    ev.clear()
    eng = _engine(tmp_path, monkeypatch, ev, release=True, hook=True, fail_hook=True)
    with pytest.raises(memguard.MemoryFloorError):
        eng._volumes_synthseg(E.Timer(), "/x/scan.nii.gz", tmp_path / "out")
    assert [e[0] for e in ev] == ["release", "check"]                # released first, then refused; SynthSeg never ran


def test_engine_source_default_call_path_is_unchanged_without_options():
    import inspect
    src = inspect.getsource(E.AssessmentEngine.predict)
    assert "self._volumes_synthseg(tm, native_path, out)" in src and "self.pool.release" not in src
