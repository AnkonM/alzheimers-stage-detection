import pathlib
import sys

import pytest

from src.assessment.workers.client import WorkerClient, WorkerError

ROOT = str(pathlib.Path(__file__).resolve().parents[1])


@pytest.fixture
def w(tmp_path):
    with WorkerClient(sys.executable, "tests.echo_worker", ROOT, str(tmp_path / "w.log"), timeout=3) as c:
        yield c


def test_roundtrip_error_and_stray_print(w):
    assert w.call("add", a=1, b=2) == 3
    assert w.call("noisy") == 1          # stray stdout does not break the protocol
    with pytest.raises(WorkerError, match="ValueError: bad input"):
        w.call("boom")
    assert w.call("add", a=2, b=2) == 4  # worker survives an error


def test_timeout_kills_worker(w):
    with pytest.raises(WorkerError, match="timed out"):
        w.call("sleep", s=10)
    with pytest.raises(WorkerError):
        w.call("add", a=1, b=1)
