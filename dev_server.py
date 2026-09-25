"""Local stand-in for Vercel: serves index.html + photos, and routes POST /api/count to the same
function Vercel runs. Usage: python dev_server.py [port]

Only reachable from this computer (127.0.0.1), and only serves the page and the sample photos, never
.env or the source files.
"""
import http.server
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "api"))
from count import handler as ApiHandler  # noqa: E402

ALLOWED = ("/", "/index.html", "/favicon.ico")


class Dev(http.server.SimpleHTTPRequestHandler):
    _send = ApiHandler._send  # the API handler's JSON reply helper

    def __init__(self, *a, **kw):
        super().__init__(*a, directory=ROOT, **kw)

    def do_GET(self):
        path = self.path.split("?", 1)[0]
        if path in ALLOWED or (path.startswith("/photos/") and "/." not in path and ".." not in path):
            return super().do_GET()
        self.send_error(404)

    def do_HEAD(self):
        self.do_GET()

    def do_POST(self):
        if self.path.startswith("/api/count"):
            return ApiHandler.do_POST(self)
        self.send_error(404)


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 3000
    print(f"http://localhost:{port}")
    http.server.ThreadingHTTPServer(("127.0.0.1", port), Dev).serve_forever()
