"""API settings (environment overrides, safe defaults). Everything the API writes lives under var/ (gitignored)."""
import os
import pathlib

REPO = pathlib.Path(__file__).resolve().parents[2]
VAR = pathlib.Path(os.environ.get("AD_VAR", REPO / "var"))
RUNS = VAR / "runs"          # one folder per report: report.json, brain.nii.gz, overlay.nii.gz, PDFs
UPLOADS = VAR / "uploads"    # removed as soon as the job finishes, and by retention
DEMO = pathlib.Path(os.environ.get("AD_DEMO_DIR", REPO / "demo_cases"))
DB = VAR / "app.db"
DIST = REPO / "frontend" / "dist"
MODEL_CARD = REPO / "docs" / "model_card.md"

HOST = "127.0.0.1"           # the API refuses to run on any other address
MAX_UPLOAD_BYTES = int(os.environ.get("AD_MAX_UPLOAD_MB", "300")) * 1024 * 1024
RETENTION_HOURS = float(os.environ.get("AD_RETENTION_HOURS", "24"))   # uploads and outputs older than this are removed
ALLOWED_HOSTS = {"127.0.0.1", "localhost"}
