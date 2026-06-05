#!/usr/bin/env python3
"""LinkedIn Optimizer local server.

The server is intentionally local-first. The dashboard only reads processed
artifacts and triggers skill runs through this gateway; it never calls Apify or
LLMs directly from the browser.
"""

from __future__ import annotations

import json
import os
import threading
import traceback
import webbrowser
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse


def _load_dotenv():
    """Load .env file if it exists (no third-party dependency needed)."""
    env_path = Path(__file__).resolve().parent / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip()
        if key and value and key not in os.environ:
            os.environ[key] = value


_load_dotenv()

from linkedin_orchestrator import SkillGateway
from linkedin_orchestrator.common import ROOT, read_json


PORT = int(os.environ.get("PORT", "8080"))
gateway = SkillGateway(ROOT)


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(ROOT), **kwargs)

    def json_response(self, status: int, payload):
        body = json.dumps(payload, indent=2, ensure_ascii=False, default=str).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def read_body(self):
        length = int(self.headers.get("Content-Length", "0") or "0")
        if length <= 0:
            return {}
        raw = self.rfile.read(length)
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return {}

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self):
        path = urlparse(self.path).path
        try:
            if path == "/api/dashboard/state":
                return self.json_response(200, gateway.dashboard_state())
            if path.startswith("/api/runs/"):
                run_id = path.rsplit("/", 1)[-1]
                run_path = ROOT / "data" / "runs" / run_id / "run.json"
                if not run_path.exists():
                    return self.json_response(404, {"error": "run_not_found", "run_id": run_id})
                return self.json_response(200, read_json(run_path, {}))
        except Exception as exc:
            traceback.print_exc()
            return self.json_response(500, {"error": str(exc)})
        return super().do_GET()

    def do_POST(self):
        path = urlparse(self.path).path
        try:
            if path.startswith("/api/skills/") and path.endswith("/run"):
                skill = path.split("/")[3]
                return self.json_response(200, gateway.run(skill))
            if path == "/api/manual/log":
                return self.json_response(200, gateway.log_manual_action(self.read_body()))
        except KeyError as exc:
            return self.json_response(404, {"error": str(exc)})
        except Exception as exc:
            traceback.print_exc()
            return self.json_response(500, {"error": str(exc)})
        return self.json_response(404, {"error": "unknown_endpoint", "path": path})

    def log_message(self, fmt, *args):
        message = fmt % args if args else fmt
        if "/api/" in message:
            print(message)


if __name__ == "__main__":
    print(f"LinkedIn Optimizer: http://localhost:{PORT}/dashboard.html")
    print("Backend: evidence-backed skill gateway")
    print("Ctrl+C to stop\n")
    gateway.bootstrap_processed()
    server = ThreadingHTTPServer(("", PORT), Handler)
    threading.Timer(1, lambda: webbrowser.open(f"http://localhost:{PORT}/dashboard.html")).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
        server.server_close()
