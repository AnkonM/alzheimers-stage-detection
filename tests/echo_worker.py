import sys
import time

from src.assessment.workers.client import serve


def boom():
    raise ValueError("bad input")


serve({"add": lambda a, b: a + b, "noisy": lambda: print("stray print") or 1, "boom": boom,
       "sleep": lambda s: time.sleep(s)})
