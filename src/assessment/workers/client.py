"""Persistent stage worker, driven over JSON lines.

Each worker is a long-lived process in its own venv. A request is one JSON object per line on stdin
({"id", "op", "args"}); the reply is one line on stdout ({"id", "ok", "result"|"error"}). Arrays are never
sent inline: they travel as .npy files in the job directory and are referenced by path. stderr goes to the
worker's log file. Worker code prints replies only through serve().
"""
import json
import os
import subprocess
import sys
import threading


class WorkerError(RuntimeError):
    pass


class WorkerClient:
    def __init__(self, python: str, module: str, cwd: str, log_path: str, timeout: float = 600, env: dict | None = None):
        self.timeout = timeout
        self._log = open(log_path, "ab")
        self._p = subprocess.Popen([python, "-u", "-m", module], cwd=cwd, stdin=subprocess.PIPE,
                                   stdout=subprocess.PIPE, stderr=self._log, text=True,
                                   env={**os.environ, **(env or {})})
        self._n = 0
        self._lock = threading.Lock()

    def call(self, op: str, **args):
        with self._lock:
            self._n += 1
            rid = self._n
            if self._p.poll() is not None:
                raise WorkerError(f"worker exited with code {self._p.returncode}")
            self._p.stdin.write(json.dumps({"id": rid, "op": op, "args": args}) + "\n")
            self._p.stdin.flush()
            line = self._read_line()
        msg = json.loads(line)
        if msg.get("id") != rid:
            raise WorkerError(f"reply id mismatch: {msg.get('id')} != {rid}")
        if not msg["ok"]:
            raise WorkerError(msg["error"])
        return msg["result"]

    def _read_line(self) -> str:
        out = []
        t = threading.Thread(target=lambda: out.append(self._p.stdout.readline()), daemon=True)
        t.start()
        t.join(self.timeout)
        if t.is_alive():
            self._p.kill()
            raise WorkerError(f"worker timed out after {self.timeout}s")
        if not out or not out[0]:
            raise WorkerError("worker closed its output (crashed?); see its log")
        return out[0]

    def close(self):
        try:
            self._p.stdin.close()
            self._p.wait(timeout=10)
        except Exception:
            self._p.kill()
        self._log.close()

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()


def serve(handlers: dict):
    """Worker main loop. Handlers are loaded once by the caller and reused for every request."""
    real_out = sys.stdout
    sys.stdout = sys.stderr  # stray prints from libraries must not corrupt the protocol
    for line in sys.stdin:
        req = json.loads(line)
        try:
            res = {"id": req["id"], "ok": True, "result": handlers[req["op"]](**req["args"])}
        except Exception as e:  # noqa: BLE001
            res = {"id": req["id"], "ok": False, "error": f"{type(e).__name__}: {e}"}
        real_out.write(json.dumps(res) + "\n")
        real_out.flush()
