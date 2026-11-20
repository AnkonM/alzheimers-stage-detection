"""Tiny static server for frontend/dist with SPA fallback (tests and local checks)."""
import http.server
import pathlib
import threading

DIST = pathlib.Path(__file__).resolve().parents[1] / "frontend" / "dist"
FIX = pathlib.Path(__file__).resolve().parent / "fixtures" / "reports"
RUNS = pathlib.Path(__file__).resolve().parents[1] / "var" / "runs"


class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *a, **k):
        super().__init__(*a, directory=str(DIST), **k)

    def do_GET(self):
        path = self.path.split("?")[0]
        if path.startswith("/fixtures/"):
            f = FIX / path.split("/")[-1]
            if f.exists():
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(f.read_bytes())
                return
        if path.startswith("/api/reports/"):             # local check harness: serve var/runs like the real API
            parts = path[len("/api/reports/"):].split("/")
            f = RUNS / parts[0] / ("report.json" if len(parts) == 1 else parts[-1])
            if f.exists():
                self.send_response(200)
                self.send_header("Content-Type", "application/json" if f.suffix == ".json" else "application/gzip")
                self.end_headers()
                self.wfile.write(f.read_bytes())
                return
            self.send_error(404)
            return
        if not (DIST / path.lstrip("/")).is_file():
            self.path = "/index.html"
        return super().do_GET()

    def log_message(self, *a):
        pass


def start():
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, f"http://127.0.0.1:{srv.server_address[1]}"
