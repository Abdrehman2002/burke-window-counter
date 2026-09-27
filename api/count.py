"""Vercel Python function: POST /api/count

Count a new photo:
  {"image": "<base64 JPEG>", "thresh": 0.4, "floors": "ai" | "geometric"}
Count only a drawn area of a photo already counted (no new detection):
  {"image": ..., "area": [x0, y0, x1, y1], "boxes": [<boxes from the first answer>], "floors": ...}

Returns {"width", "height", "boxes": [{x0,y0,x1,y1,conf,floor,edge,other_side,weak,region}],
         "regions": [{id,label,polygon,windows,n_floors,note,floors_check}], "selected": [ids],
         "n_floors", "note", "floors_check", "seconds"}
or {"error": "...", "code": "..."} with a 4xx/5xx status.

Every request must send the password in the X-Access-Code header. The password is checked against the
ACCESS_CODE environment variable if set, else against ACCESS_CODE_SHA256 (a SHA-256 hex digest), else
against the built-in digest below. Only a hash is kept in the code, never the password itself.
GET /api/count answers 200 for the right password and 401 otherwise (the page uses it to unlock).
Pricing and drawing happen in the browser, so changing a price never calls this again.
"""
import base64
import hashlib
import hmac
import io
import json
import os
import sys
import time
from collections import defaultdict, deque
from http.server import BaseHTTPRequestHandler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _core import MAX_BOXES, Box, load_env, run_area, run_side  # noqa: E402

from PIL import Image, ImageOps  # noqa: E402

load_env()
DEFAULT_CODE_SHA256 = "c2178a776b95281b1a7c47470d0c80bfa97f9d83bd225694d4164129c1795246"  # set 2026-09-26; change with ACCESS_CODE or ACCESS_CODE_SHA256 in Vercel
MAX_SIDE = 1600
MAX_PIXELS = 40_000_000  # ~40 MP: bigger than any phone photo; stops decompression bombs
Image.MAX_IMAGE_PIXELS = MAX_PIXELS

# Best effort per-address limit. Serverless instances are short-lived and separate, so this only slows
# down a burst from one address on one warm instance; the ACCESS_CODE is the real gate.
RATE_LIMIT, RATE_WINDOW_S = 40, 600
_hits: dict[str, deque] = defaultdict(deque)


def _rate_ok(ip: str) -> bool:
    now = time.time()
    q = _hits[ip]
    while q and now - q[0] > RATE_WINDOW_S:
        q.popleft()
    if len(q) >= RATE_LIMIT:
        return False
    q.append(now)
    return True


def _load_image(raw_b64: str) -> Image.Image:
    data = base64.b64decode(raw_b64.split(",", 1)[-1])  # accept a data: URL or bare base64
    im = Image.open(io.BytesIO(data))
    w, h = im.size
    if w * h > MAX_PIXELS:
        raise ValueError(f"the image is {w}x{h}, too large; send a photo under 40 megapixels")
    im.draft("RGB", (MAX_SIDE, MAX_SIDE))  # JPEGs decode straight at a smaller size: fast, low memory
    im.thumbnail((MAX_SIDE, MAX_SIDE))      # shrink before rotating, never after
    return ImageOps.exif_transpose(im).convert("RGB")


def code_ok(code: str | None) -> bool:
    want = os.environ.get("ACCESS_CODE", "")
    if want:
        return hmac.compare_digest(want, code or "")
    digest = os.environ.get("ACCESS_CODE_SHA256", DEFAULT_CODE_SHA256)
    return hmac.compare_digest(hashlib.sha256((code or "").encode()).hexdigest(), digest)


def process(body: bytes, access_code: str | None = None, ip: str = "local") -> tuple[int, dict]:
    if not code_ok(access_code):
        return 401, {"error": "Wrong or missing password.", "code": "auth"}
    if not _rate_ok(ip):
        return 429, {"error": "Too many photos in a short time. Wait a few minutes and try again.", "code": "rate"}
    try:
        req = json.loads(body or b"{}")
        if not isinstance(req, dict) or not isinstance(req.get("image"), str):
            raise ValueError("send the photo as base64 in \"image\"")
        img = _load_image(req["image"])
        thresh = min(max(float(req.get("thresh", 0.4)), 0.2), 0.8)
        floors = req.get("floors") if req.get("floors") in ("ai", "geometric") else "ai"
        area = req.get("area")
        boxes = None
        if area is not None:
            if not (isinstance(area, list) and len(area) == 4):
                raise ValueError("\"area\" must be [x0, y0, x1, y1]")
            area = [float(v) for v in area]
            raw = req.get("boxes")
            if not isinstance(raw, list):
                raise ValueError("\"area\" needs the \"boxes\" from the first count")
            if len(raw) > MAX_BOXES:
                raise ValueError(f"too many boxes (over {MAX_BOXES})")
            boxes = [Box(float(b["x0"]), float(b["y0"]), float(b["x1"]), float(b["y1"]), float(b["conf"]),
                         weak=bool(b.get("weak"))) for b in raw]
    except Exception as e:
        return 400, {"error": f"Could not read the request: {e}", "code": "bad_request"}
    t = time.time()
    try:
        res = run_area(img, boxes, area, floors=floors) if area is not None else run_side(img, floors=floors, thresh=thresh)
    except Exception as e:
        return 502, {"error": f"Window detection failed ({str(e)[:200]}). Try again in a moment.", "code": "upstream"}
    return 200, {
        "width": img.width, "height": img.height,
        "boxes": [{"x0": b.x0, "y0": b.y0, "x1": b.x1, "y1": b.y1, "conf": b.conf, "floor": int(b.floor),
                   "edge": bool(b.edge), "other_side": bool(b.other_side), "weak": bool(b.weak), "region": int(b.region),
                   "kind": b.kind, "gy": round(float(b.gy), 4)}
                  for b in res.boxes],
        "regions": [{"id": r["id"], "label": r["label"], "polygon": r["polygon"], "windows": int(r["windows"]),
                     "n_floors": int(r.get("n_floors") or 0), "note": r.get("note", ""),
                     "floors_check": bool(r.get("floors_check")), "glass": bool(r.get("glass")),
                     "storeys": int(r.get("storeys") or 0)} for r in res.regions],
        "selected": list(res.selected),
        "n_floors": len(res.floors), "note": res.note, "floors_check": bool(res.floors_check),
        "seconds": round(time.time() - t, 1),
    }


class handler(BaseHTTPRequestHandler):
    def _send(self, status: int, payload: dict):
        out = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(out)))
        self.end_headers()
        self.wfile.write(out)

    def do_POST(self):
        length = int(self.headers.get("content-length") or 0)
        if length > 6_000_000:
            return self._send(413, {"error": "The photo is too large to upload.", "code": "too_large"})
        body = self.rfile.read(length)
        ip = (self.headers.get("x-forwarded-for") or self.client_address[0] or "?").split(",")[0].strip()
        status, payload = process(body, self.headers.get("x-access-code"), ip)
        self._send(status, payload)

    def do_GET(self):  # password check for the page's unlock screen
        ip = (self.headers.get("x-forwarded-for") or self.client_address[0] or "?").split(",")[0].strip()
        if not _rate_ok(ip):
            return self._send(429, {"error": "Too many tries. Wait a few minutes.", "code": "rate"})
        if not code_ok(self.headers.get("x-access-code")):
            return self._send(401, {"error": "Wrong password.", "code": "auth"})
        self._send(200, {"ok": True})
