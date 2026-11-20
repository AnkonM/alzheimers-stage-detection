"""SQLite job store. One row per job; a job's id is also its report id."""
import json
import sqlite3
import threading
import time

from . import config

SCHEMA = """
create table if not exists jobs (
  id text primary key, status text not null, step text, steps text not null default '[]', plan text not null default '[]', mode text not null,
  subject_label text, error text, created_at real not null, finished_at real, precomputed integer not null default 0
);
"""


class Store:
    def __init__(self, path=None):
        config.VAR.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(str(path or config.DB), check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        with self._lock:
            self._db.executescript(SCHEMA)
            self._db.commit()

    def _run(self, sql, *args):
        with self._lock:
            cur = self._db.execute(sql, args)
            self._db.commit()
            return cur

    def create(self, job_id, mode, subject_label, plan=(), precomputed=False, status="queued"):
        self._run("insert or replace into jobs (id,status,mode,subject_label,created_at,precomputed,finished_at,plan) "
                  "values (?,?,?,?,?,?,?,?)", job_id, status, mode, subject_label, time.time(), int(precomputed),
                  time.time() if status == "done" else None, json.dumps(list(plan)))

    def step(self, job_id, step):
        row = self.get(job_id)
        steps = json.loads(row["steps"]) + [{"step": step, "t": time.time()}]
        self._run("update jobs set status='running', step=?, steps=? where id=?", step, json.dumps(steps), job_id)

    def finish(self, job_id, error=None):
        self._run("update jobs set status=?, error=?, finished_at=?, step=null where id=?",
                  "failed" if error else "done", error, time.time(), job_id)

    def get(self, job_id):
        with self._lock:
            return self._db.execute("select * from jobs where id=?", (job_id,)).fetchone()

    def list(self):
        with self._lock:
            return self._db.execute("select * from jobs order by created_at desc").fetchall()

    def delete(self, job_id):
        self._run("delete from jobs where id=?", job_id)

    def expired(self, hours):
        cutoff = time.time() - hours * 3600
        with self._lock:
            return [r["id"] for r in self._db.execute(
                "select id from jobs where precomputed=0 and created_at < ?", (cutoff,)).fetchall()]
