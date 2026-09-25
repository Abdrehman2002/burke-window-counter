"""Local stand-in for Vercel: serves index.html + photos, and routes POST /api/count to the same
function Vercel runs. Usage: python dev_server.py [port]"""
import http.server
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "api"))
from count import handler as ApiHandler  # noqa: E402


class Dev(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *a, **kw):
        super().__init__(*a, directory=ROOT, **kw)

    def do_POST(self):
        if self.path.startswith("/api/count"):
            return ApiHandler.do_POST(self)
        self.send_error(404)


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 3000
    print(f"http://localhost:{port}")
    http.server.ThreadingHTTPServer(("", port), Dev).serve_forever()
