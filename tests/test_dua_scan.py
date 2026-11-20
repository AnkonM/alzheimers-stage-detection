"""Tests for tools/dua_scan.py. Runs under pytest or `python3 -m unittest`."""
import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))
import dua_scan as D  # noqa: E402


class DuaScanTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        r = self.tmp.name
        files = {
            "a/x.nii.gz": b"", "b/w.pt": b"", "b/m.cbm": b"", "b/arr.npz": b"", "b/fig.png": b"",
            "q/123_S_4567.png": b"", "q/123_S_4568.png": b"",
            "b/t.csv": b"RID,MMSE\n1,30\n", "b/notes.md": b"subject 002_S_0001 here\n",
            "b/ADNIMERGE.csv": b"a,b\n1,2\n",
            "b/n.ipynb": json.dumps({"cells": [{"cell_type": "code", "outputs": [{"x": 1}]}]}).encode(),
            "b/ok.py": b"clean\n", "b/agg.csv": b"stage,f1\nAD,0.9\n",
        }
        for p, data in files.items():
            os.makedirs(os.path.join(r, os.path.dirname(p)), exist_ok=True)
            with open(os.path.join(r, p), "wb") as f:
                f.write(data)

    def tearDown(self):
        self.tmp.cleanup()

    def scan(self, allow=()):
        return D.run_scan(D.scan_tree(self.tmp.name), list(allow))

    def test_every_rule_fires(self):
        _, f = self.scan()
        for rule in ["EXT_IMAGE", "EXT_WEIGHTS", "EXT_ARRAY", "IMAGE_REVIEW", "PATH_ID",
                     "CSV_SUBJECT", "CONTENT_ID", "ADNI_TABLE", "NB_OUTPUT"]:
            self.assertIn(rule, f, rule)
        self.assertEqual(len(f["EXT_WEIGHTS"]), 2)

    def test_clean_files_pass(self):
        _, f = self.scan()
        flagged = {p for v in f.values() for p in v}
        self.assertNotIn("b/ok.py", flagged)
        self.assertNotIn("b/agg.csv", flagged)

    def test_ids_redacted_but_counted(self):
        _, f = self.scan()
        self.assertEqual(len(f["PATH_ID"]), 2)  # two files, not collapsed into one
        self.assertFalse(any("4567" in p or "4568" in p for v in f.values() for p in v))

    def test_allowlist(self):
        _, f = self.scan(allow=["a/*.nii.gz"])
        self.assertNotIn("EXT_IMAGE", f)


if __name__ == "__main__":
    unittest.main()
