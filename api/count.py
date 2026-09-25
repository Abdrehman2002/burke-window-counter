"""Vercel Python function: POST /api/count

Body (JSON): {"image": "<base64 JPEG>", "thresh": 0.4, "floors": "ai" | "geometric"}
Returns:     {"width", "height", "boxes": [{x0,y0,x1,y1,conf,floor}], "n_floors", "note", "seconds"}

Pricing and drawing happen in the browser, so changing a price never calls this again.
"""
import base64
import io
import json
import os
import sys
import time
from http.server import BaseHTTPRequestHandler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _core import load_env, run_side  # noqa: E402

from PIL import Image, ImageOps  # noqa: E402

load_env()
MAX_SIDE = 1600


def process(body: bytes) -> tuple[int, dict]:
    try:
        req = json.loads(body or b"{}")
        raw = req["image"].split(",", 1)[-1]  # accept a data: URL or bare base64
        img = ImageOps.exif_transpose(Image.open(io.BytesIO(base64.b64decode(raw)))).convert("RGB")
    except Exception as e:
        return 400, {"error": f"Could not read the image ({e})"}
    img.thumbnail((MAX_SIDE, MAX_SIDE))
    thresh = float(req.get("thresh", 0.4))
    floors = req.get("floors", "ai") if req.get("floors") in ("ai", "geometric") else "ai"
    t = time.time()
    try:
        res = run_side(img, floors=floors, thresh=thresh)
    except Exception as e:
        return 502, {"error": f"Window detection failed ({str(e)[:200]})"}
    return 200, {
        "width": img.width, "height": img.height,
        "boxes": [{"x0": b.x0, "y0": b.y0, "x1": b.x1, "y1": b.y1, "conf": b.conf, "floor": int(b.floor), "edge": bool(b.edge), "other_side": bool(b.other_side), "weak": bool(b.weak)}
                  for b in res.boxes],
        "n_floors": len(res.floors), "note": res.note, "floors_check": bool(res.floors_check), "seconds": round(time.time() - t, 1),
    }


class handler(BaseHTTPRequestHandler):
    def do_POST(self):
        body = self.rfile.read(int(self.headers.get("content-length") or 0))
        status, payload = process(body)
        out = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(out)))
        self.end_headers()
        self.wfile.write(out)

    def do_GET(self):
        out = json.dumps({"ok": True, "keys": {k: bool(os.environ.get(k)) for k in
                                               ("ROBOFLOW_API_KEY", "OPENAI_API_KEY", "GEMINI_API_KEY")}}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(out)
