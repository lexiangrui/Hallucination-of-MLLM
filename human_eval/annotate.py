#!/usr/bin/env python3
"""
Minimal annotation server — serves the HTML frontend and CSV/image APIs.
"""

import csv
import http.server
import json
import os
import socketserver
import sys
import urllib.parse
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
HUMAN_EVAL_DIR = REPO_ROOT / "human_eval"

SAMPLE_FIELDS = [
    "sample_id", "model", "pid", "image", "question",
    "gt_answer", "model_response", "human_label", "human_type",
]

CSV_PATH = HUMAN_EVAL_DIR / "annotations.csv"
HTML_PATH = HUMAN_EVAL_DIR / "index.html"


TEXT_FIELDS = {"question", "gt_answer", "model_response"}


def load_csv() -> list[dict]:
    if not CSV_PATH.exists():
        return []
    with open(CSV_PATH, newline="") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        for field in TEXT_FIELDS:
            val = row.get(field, "")
            if val:
                row[field] = val.replace("\\n", "\n")
    return rows


def save_csv(rows: list[dict]) -> None:
    with open(CSV_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SAMPLE_FIELDS)
        writer.writeheader()
        for row in rows:
            out = {}
            for k in SAMPLE_FIELDS:
                val = row.get(k, "")
                if k in TEXT_FIELDS:
                    val = val.replace("\n", "\\n")
                out[k] = val
            writer.writerow(out)


def update_row(sample_id: str, label: str, hal_type: str) -> dict | None:
    rows = load_csv()
    for row in rows:
        if row["sample_id"] == sample_id:
            row["human_label"] = label
            row["human_type"] = hal_type
            save_csv(rows)
            return row
    return None


class Handler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        params = urllib.parse.parse_qs(parsed.query)

        if parsed.path == "/api/samples":
            rows = load_csv()
            stats = {"total": len(rows), "annotated": sum(1 for r in rows if r.get("human_label"))}
            payload = json.dumps({"samples": rows, "stats": stats}, ensure_ascii=False)
            self._json_response(payload)
            return

        if parsed.path == "/api/image":
            path = params.get("path", [None])[0]
            if path and os.path.exists(path):
                self.send_response(200)
                self.send_cache_headers()
                ext = Path(path).suffix.lower()
                self._guess_mime(ext)
                self.end_headers()
                with open(path, "rb") as f:
                    self.wfile.write(f.read())
            else:
                self._json_response(json.dumps({"error": "image not found"}), status=404)
            return

        if parsed.path == "/":
            self._serve_html()
            return

        super().do_GET()

    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path == "/api/save":
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length))
            sample_id = body.get("sample_id", "")
            label = body.get("human_label", "")
            hal_type = body.get("human_type", "")
            result = update_row(sample_id, label, hal_type)
            if result:
                self._json_response(json.dumps({"ok": True}))
            else:
                self._json_response(json.dumps({"ok": False, "error": "sample not found"}), status=404)
            return

        self.send_error(404)

    def _serve_html(self):
        if HTML_PATH.exists():
            with open(HTML_PATH) as f:
                html = f.read()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            self.wfile.write(html.encode("utf-8"))
        else:
            self.send_error(500, "index.html not found")

    def _json_response(self, payload: str, status: int = 200):
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_cache_headers()
        self.end_headers()
        self.wfile.write(payload.encode("utf-8"))

    def send_cache_headers(self):
        self.send_header("Cache-Control", "no-cache")

    def _guess_mime(self, ext):
        mime_map = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png",
                    ".gif": "image/gif", ".webp": "image/webp"}
        self.send_header("Content-Type", mime_map.get(ext, "application/octet-stream"))

    def log_message(self, fmt, *args):
        pass  # quiet


def main():
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8765
    url = f"http://localhost:{port}"
    print(f"Annotation server started: {url}")
    print(f"CSV path: {CSV_PATH}")
    print("Press Ctrl+C to stop.")
    with socketserver.TCPServer(("", port), Handler) as httpd:
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            httpd.shutdown()


if __name__ == "__main__":
    main()