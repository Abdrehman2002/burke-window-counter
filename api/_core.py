"""Window counter for Burke: detect windows -> group into floors -> apply per-floor pricing.

Usage:  python counter.py photos/06_apt_karlsruhe.jpg [--crop x0,y0,x1,y1] [--thresh 0.4]
Writes output/<name>_marked.jpg and prints the per-floor table.
"""
import argparse
import base64
import io
import json
import os
import sys
import time
from dataclasses import dataclass, field

import numpy as np
import requests
from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repo root

# One time budget per request. Vercel stops the function at 60 s, and a call still running when it is
# stopped is still billed, so every outside call sizes its timeout from what is left of this budget.
BUDGET_S = float(os.environ.get("BUDGET_S", "50"))
_DEADLINE = 0.0  # set at the start of each run (one request per function instance on Vercel)
MAX_BOXES = 400  # more than any real facade; caps the work a single request can cause


def start_budget():
    global _DEADLINE
    _DEADLINE = time.time() + BUDGET_S


def remaining() -> float:
    return (_DEADLINE - time.time()) if _DEADLINE else 1e9


def _read_timeout(cap: float) -> float:
    left = remaining() - 2
    if left < 3:
        raise TimeoutError("out of time for this photo")
    return min(cap, left)

# Burke's pricing: floor n costs BASE_PRICE * MULTIPLIER ** (n - 1) per window.
BASE_PRICE = 3.0
MULTIPLIER = 2.0

# Free tier: only the Flash preview has quota on this key (2.5 is retired for new users, Pro has none).
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3-flash-preview")

FLOOR_COLOURS = [(21, 75, 255), (230, 57, 70), (42, 157, 143), (244, 162, 97),
                 (131, 56, 236), (255, 190, 11), (58, 134, 255), (6, 214, 160)]


def load_env():
    path = os.path.join(HERE, ".env")
    if os.path.exists(path):
        for line in open(path, encoding="utf-8"):
            if "=" in line and not line.startswith("#"):
                k, v = line.strip().split("=", 1)
                os.environ.setdefault(k, v)


@dataclass
class Box:
    x0: float
    y0: float
    x1: float
    y1: float
    conf: float
    floor: int = 0
    edge: bool = False  # cut off by the left/right edge of the photo: almost always the building next door
    other_side: bool = False  # not on the one wall this photo is quoting (a second side, or next door)
    weak: bool = False  # the detector was under WEAK_BELOW sure: shown as a suggestion, not counted
    region: int = 0  # which building / wall of the photo this window belongs to (see Region)
    kind: str = "window"  # which prompt found it: "window", a glass-door prompt, or "door" (solid door, not counted)

    @property
    def counted(self): return not (self.edge or self.other_side or self.weak)

    @property
    def w(self): return self.x1 - self.x0
    @property
    def h(self): return self.y1 - self.y0
    @property
    def cy(self): return (self.y0 + self.y1) / 2
    @property
    def area(self): return max(self.w, 0) * max(self.h, 0)


# ---------- 1. detect ----------

def _jpeg_b64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=92)
    return base64.b64encode(buf.getvalue()).decode()


# Things whose "windows" are not building windows. Asked for in the same SAM 3 call; any window
# whose centre sits on one of these is dropped.
EXCLUDE_PROMPTS = ["car", "vehicle"]
# Glass to count. Extra prompts ride in the same request, so they cost nothing more.
GLASS_PROMPTS = [p.strip() for p in os.environ.get("GLASS_PROMPTS", "window").split(",") if p.strip()]
# Doors are found only to drop a small window that sits inside a solid door (a door light is not a window).
# Tested 2026-09-26: adding "door" to the same SAM 3 request changes what it returns for "window" (fake
# floors on Karlsruhe and the rowhouses): 85% -> 70%. Off by default. Extra glass prompts ("glass door",
# "french door") scored 82% vs 85% for "window" alone, so GLASS_PROMPTS stays "window".
DOOR_RULE = os.environ.get("DOOR_RULE", "0") == "1"


def _sam3_boxes(pred_list, kind="window") -> list[Box]:
    boxes = []
    for p in pred_list:
        pts = np.array([pt for poly in p["masks"] for pt in poly], dtype=float)
        if len(pts):
            boxes.append(Box(pts[:, 0].min(), pts[:, 1].min(), pts[:, 0].max(), pts[:, 1].max(), p["confidence"], kind=kind))
    return boxes


def _roboflow_post(payload: dict, tries=2) -> dict:
    """POST to Roboflow's hosted SAM 3, retrying server errors and timeouts (they happen now and then,
    and with 6 calls per photo one blip would otherwise sink the whole side)."""
    last = None
    for attempt in range(tries):
        try:
            r = requests.post("https://serverless.roboflow.com/sam3/concept_segment",
                              headers={"Authorization": f"Bearer {os.environ['ROBOFLOW_API_KEY']}"},
                              json=payload, timeout=(5, _read_timeout(20)))
            if r.status_code >= 500 or r.status_code == 429:
                last = RuntimeError(f"Roboflow {r.status_code}")
            else:
                r.raise_for_status()
                return r.json()
        except requests.exceptions.RequestException as e:
            last = e
        if remaining() < 10:
            break
        time.sleep(0.8 * (attempt + 1))
    raise RuntimeError(f"window detector unavailable ({last})")


def _sam3_call(img: Image.Image, thresh=0.4, prompt="window") -> list[Box]:
    """One SAM 3 request for every kind at once: the glass prompts (counted), cars and vehicles (anything on
    them is dropped) and, with DOOR_RULE, doors (returned as kind "door", never counted)."""
    glass = GLASS_PROMPTS if prompt == "window" else [prompt]
    names = glass + EXCLUDE_PROMPTS + (["door"] if DOOR_RULE else [])
    data = _roboflow_post({"image": {"type": "base64", "value": _jpeg_b64(img)},
                           "prompts": [{"type": "text", "text": t} for t in names],
                           "output_prob_thresh": thresh, "format": "polygon"})
    results = sorted(data["prompt_results"], key=lambda x: x["prompt_index"])
    windows, blockers, doors = [], [], []
    for name, res in zip(names, results):
        if name in glass:
            windows += _sam3_boxes(res["predictions"], "window" if name == "window" else name)
        elif name == "door":
            doors += _sam3_boxes(res["predictions"], "door")
        else:
            blockers += _sam3_boxes(res["predictions"])
    def on_blocker(w: Box) -> bool:
        cx = (w.x0 + w.x1) / 2
        return any(k.x0 <= cx <= k.x1 and k.y0 <= w.cy <= k.y1 for k in blockers)
    return [w for w in windows if not on_blocker(w)] + doors


TILE_GRID = int(os.environ.get("TILE_GRID", "2"))


def detect_sam3(img: Image.Image, thresh=0.4, prompt="window") -> list[Box]:
    """SAM 3 shrinks every image to about 1000px inside, so on a big building each window is only a few
    pixels and many are missed. So we also cut the photo into overlapping tiles, enlarge each one, and
    detect on every tile at the same time. Boxes cut off by a tile's INNER edge are dropped (the
    neighbouring tile sees that window whole); the whole-photo pass keeps big windows no tile holds.
    clean() then merges the duplicates."""
    from concurrent.futures import ThreadPoolExecutor
    W, H = img.size
    jobs = [None]  # None = the whole photo
    if TILE_GRID > 1:
        tw, th = W / TILE_GRID, H / TILE_GRID
        ow, oh = 0.2 * tw, 0.2 * th
        for gy in range(TILE_GRID):
            for gx in range(TILE_GRID):
                jobs.append((int(max(0, gx * tw - ow)), int(max(0, gy * th - oh)),
                             int(min(W, (gx + 1) * tw + ow)), int(min(H, (gy + 1) * th + oh))))

    def run(job):
        if job is None:
            return _sam3_call(img, thresh, prompt)
        x0, y0, x1, y1 = job
        crop = img.crop(job)
        k = 1600 / max(crop.size)
        crop = crop.resize((max(1, int(crop.width * k)), max(1, int(crop.height * k))))
        out, m = [], 3 / k  # a 3px margin in the tile, in photo pixels
        for b in _sam3_call(crop, thresh, prompt):
            bx = Box(x0 + b.x0 / k, y0 + b.y0 / k, x0 + b.x1 / k, y0 + b.y1 / k, b.conf)
            cut = ((x0 > 0 and bx.x0 <= x0 + m) or (x1 < W and bx.x1 >= x1 - m) or
                   (y0 > 0 and bx.y0 <= y0 + m) or (y1 < H and bx.y1 >= y1 - m))
            if not cut:
                out.append(bx)
        return out

    failed = []

    def safe(job):
        try:
            return run(job)
        except Exception:
            if job is None:  # the whole-photo pass is the backbone: if it fails, the side fails
                raise
            failed.append(job)
            return []        # one tile failing costs a little recall, never the whole quote

    with ThreadPoolExecutor(len(jobs)) as ex:
        parts = list(ex.map(safe, jobs))
    detect_sam3.failed_tiles = len(failed)
    return [b for p in parts for b in p]


detect_sam3.failed_tiles = 0


def detect_gemini(img: Image.Image, model=GEMINI_MODEL) -> list[Box]:
    """Fallback detector. Gemini returns boxes as [ymin, xmin, ymax, xmax] on a 0-1000 grid."""
    prompt = ("Detect every window on the building facade in this photo. Count each whole window once "
              "(a window split into several panes is ONE window). Do not include doors or shop entrances. "
              'Return JSON only: [{"box_2d": [ymin, xmin, ymax, xmax]}] with coordinates normalised to 0-1000.')
    text = _gemini_json(model, img, prompt)
    W, H = img.size
    boxes = []
    for item in json.loads(text):
        y0, x0, y1, x1 = item["box_2d"]
        boxes.append(Box(x0 / 1000 * W, y0 / 1000 * H, x1 / 1000 * W, y1 / 1000 * H, 1.0))
    return boxes


def _gemini_json(model: str, img: Image.Image, prompt: str, tries=3, timeout=75) -> str:
    """One Gemini call returning JSON text. The free preview model throws 503s and stalls, so:
    short timeout, low thinking, a few retries. Raises if every try fails."""
    last = None
    if remaining() < 20:
        raise TimeoutError("no time left for the fallback model")
    for attempt in range(tries):
        if remaining() < 15:
            break
        try:
            r = requests.post(
                f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
                headers={"x-goog-api-key": os.environ["GEMINI_API_KEY"]},
                json={"contents": [{"parts": [{"inline_data": {"mime_type": "image/jpeg", "data": _jpeg_b64(img)}},
                                              {"text": prompt}]}],
                      "generationConfig": {"response_mime_type": "application/json", "temperature": 0,
                                           "thinkingConfig": {"thinkingLevel": "low"}}},
                timeout=(5, _read_timeout(timeout)))
            if r.status_code in (429, 500, 503):
                last = RuntimeError(f"Gemini {r.status_code}")
                time.sleep(2 + 3 * attempt)
                continue
            r.raise_for_status()
            parts = r.json()["candidates"][0]["content"]["parts"]
            return next(p["text"] for p in parts if "text" in p and not p.get("thought"))
        except requests.exceptions.Timeout as e:
            last = e
    raise RuntimeError(f"Gemini failed after {tries} tries: {last}")


# ---------- clean-up ----------

def _inter(a: Box, b: Box) -> float:
    iw = min(a.x1, b.x1) - max(a.x0, b.x0)
    ih = min(a.y1, b.y1) - max(a.y0, b.y0)
    return max(iw, 0) * max(ih, 0)


def _iou(a: Box, b: Box) -> float:
    i = _inter(a, b)
    return i / (a.area + b.area - i) if i else 0.0


def clean(boxes: list[Box]) -> list[Box]:
    """Sort out overlapping boxes so each real window is counted once.

    - A big box around two or more window-sized boxes is a GROUP (a pair of windows boxed together):
      drop the big one, keep the windows.
    - A small box inside a bigger one is a PANE or a duplicate: keep the big one.
    - Two boxes that half-overlap are the same window seen twice: keep the more confident one.
    """
    boxes = [b for b in boxes if b.area > 0]
    if not boxes:
        return []
    # "A normal window's size" comes from confident boxes only; low-confidence specks would shrink it.
    sure = [b.area for b in boxes if b.conf >= 0.5] or [b.area for b in boxes]
    typical = float(np.median(sure))
    groups = set()
    for i, big in enumerate(boxes):
        inside = [s for j, s in enumerate(boxes) if j != i and s.area < big.area and _inter(s, big) / s.area > 0.4]
        window_sized = [s for s in inside if s.area > 0.45 * typical]
        if len(window_sized) >= 2 and big.area > 1.3 * typical:
            groups.add(i)
    boxes = [b for i, b in enumerate(boxes) if i not in groups]
    # same window detected twice, partly offset
    boxes.sort(key=lambda b: b.conf, reverse=True)
    deduped: list[Box] = []
    for b in boxes:
        if all(_iou(b, k) < 0.35 for k in deduped):
            deduped.append(b)
    boxes = sorted(deduped, key=lambda b: b.area, reverse=True)
    kept: list[Box] = []
    for b in boxes:
        if b.area <= 0:
            continue
        # contained mostly inside an already-kept (larger) box -> it is a pane or a duplicate
        if any(_inter(b, k) / b.area > 0.6 for k in kept):
            continue
        kept.append(b)
    kept = _merge_stacked(kept)
    # drop outliers far bigger than a typical window (shop fronts, whole walls)
    if len(kept) >= 4:
        med = np.median([k.area for k in kept])
        kept = [k for k in kept if k.area < med * 8]
    return kept


# Tested 2026-09-26 on the 10-building benchmark: close-up verification of low-confidence candidates found
# a missed floor (Wien) but let in more false windows (panes, porch lattice) than it recovered: 83% -> 69-73%.
# Off by default; kept as a switch for a better verifier.
VERIFY = os.environ.get("VERIFY", "0") == "1"
CAND_THRESH = float(os.environ.get("CAND_THRESH", "0.2"))  # SAM 3 candidates this sure or more get a look
VERIFY_MAX = 48

VERIFY_PROMPT = (
    "Each numbered tile is a close-up from a photo of a building, with a RED box drawn on it. For each "
    "number, answer true if the red box is ONE window or ONE glass door that a window cleaner would clean "
    "(including French doors and windows behind railings or bars), and false if it is anything else: wall, "
    "roof, trim, a vent, a solid door, a door's small window, a sign, a light, a tree or leaves, a vehicle, "
    "a reflection, a boarded-up opening with no glass, or a box around several windows at once. "
    "Tiles: {nums}. Return JSON only: {{\"windows\": {{\"1\": true, \"2\": false}}}}")


def _candidate_sheet(img: Image.Image, cands: list[Box], tile=192, cols=8) -> Image.Image:
    """A contact sheet: each candidate as a close-up with its box in red and a number, so the model judges
    one small, clear question per tile instead of searching the whole photo."""
    rows = (len(cands) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * tile, rows * tile), (255, 255, 255))
    d = ImageDraw.Draw(sheet)
    f = _font(22)
    for i, b in enumerate(cands):
        m = 0.6 * max(b.w, b.h) + 8
        x0, y0 = max(0, b.x0 - m), max(0, b.y0 - m)
        x1, y1 = min(img.width, b.x1 + m), min(img.height, b.y1 + m)
        crop = img.crop((int(x0), int(y0), int(x1), int(y1)))
        k = (tile - 8) / max(crop.width, crop.height)
        crop = crop.resize((max(1, int(crop.width * k)), max(1, int(crop.height * k))))
        cd = ImageDraw.Draw(crop)
        cd.rectangle([(b.x0 - x0) * k, (b.y0 - y0) * k, (b.x1 - x0) * k, (b.y1 - y0) * k], outline=(255, 0, 0), width=3)
        ox, oy = (i % cols) * tile + 4, (i // cols) * tile + 4
        sheet.paste(crop, (ox, oy))
        d.rectangle([ox, oy, ox + 34, oy + 28], fill=(0, 0, 0))
        d.text((ox + 4, oy + 2), str(i + 1), font=f, fill=(255, 255, 0))
    return sheet


def verify_candidates(img: Image.Image, cands: list[Box], provider="openai"):
    """Ask the model, 3 votes, whether each doubtful candidate is a window or glass door. A unanimous yes
    counts it; a split yes counts it but leaves it amber for the viewer; a no leaves it uncounted."""
    if not cands:
        return
    cands = sorted(cands, key=lambda b: -b.conf)[:VERIFY_MAX]
    picture = _candidate_sheet(img, cands)
    prompt = VERIFY_PROMPT.format(nums=", ".join(str(i + 1) for i in range(len(cands))))
    from concurrent.futures import ThreadPoolExecutor
    ask = (lambda: _openai_json(picture, prompt)) if provider == "openai" else (lambda: _gemini_json(GEMINI_MODEL, picture, prompt))
    with ThreadPoolExecutor(VOTES) as ex:
        futs = [ex.submit(ask) for _ in range(VOTES)]
    votes = []
    for f in futs:
        try:
            a = json.loads(f.result())
            if isinstance(a, dict) and isinstance(a.get("windows"), dict):
                votes.append(a["windows"])
        except Exception:
            pass
    if not votes:
        return
    for i, b in enumerate(cands):
        yes = sum(1 for v in votes if v.get(str(i + 1)) is True)
        if yes * 2 > len(votes):
            b.weak = False
            b.conf = max(b.conf, CHECK_BELOW if yes == len(votes) else min(b.conf, CHECK_BELOW - 0.01))


def _drop_door_lights(windows: list[Box], doors: list[Box]) -> list[Box]:
    """A small window inside a solid door (a door light or a grille) is not a window to clean. A box that
    fills most of a door is the door itself seen as glass (a French or glass door) and is kept."""
    out = []
    for w in windows:
        cx = (w.x0 + w.x1) / 2
        inside = [d for d in doors if d.x0 <= cx <= d.x1 and d.y0 <= w.cy <= d.y1 and d.area > 0]
        if inside and all(w.area < 0.45 * d.area for d in inside) and w.kind == "window":
            continue
        out.append(w)
    return out


def _merge_stacked(boxes: list[Box]) -> list[Box]:
    """A tall window sometimes comes back as two boxes, top half and bottom half. Merge two boxes that
    share almost the same left and right edges and touch vertically. Windows on different floors have
    a band of wall between them, so they never touch like this. One sweep in x order: only boxes that
    overlap horizontally are compared."""
    boxes = sorted(boxes, key=lambda b: b.x0)
    i = 0
    while i < len(boxes):
        a = boxes[i]
        j = i + 1
        while j < len(boxes) and boxes[j].x0 < a.x1:
            b = boxes[j]
            overlap_w = min(a.x1, b.x1) - max(a.x0, b.x0)
            gap = max(a.y0, b.y0) - min(a.y1, b.y1)  # negative = they overlap
            if overlap_w >= 0.8 * max(a.w, b.w) and gap < 0.08 * min(a.h, b.h):
                a = boxes[i] = Box(min(a.x0, b.x0), min(a.y0, b.y0), max(a.x1, b.x1), max(a.y1, b.y1),
                                   min(a.conf, b.conf))
                del boxes[j]
                j = i + 1  # the grown box may now touch one we already passed
                continue
            j += 1
        i += 1
    return boxes


# ---------- 2. floors ----------

def _same_floor_pairs(boxes: list[Box]) -> list[tuple[int, int]]:
    """Each window paired with its most level neighbour on each side (overlap > half its height)."""
    cx = [(b.x0 + b.x1) / 2 for b in boxes]
    pairs = set()
    for i, a in enumerate(boxes):
        for side in (-1, 1):
            best, best_score = -1, 0.5
            for j, c in enumerate(boxes):
                if j == i or side * (cx[j] - cx[i]) <= 0:
                    continue
                ov = (min(a.y1, c.y1) - max(a.y0, c.y0)) / min(a.h, c.h)
                if ov <= 0.5:
                    continue
                score = ov - 0.02 * abs(cx[j] - cx[i]) / max(a.w, 1)
                if score > best_score:
                    best, best_score = j, score
            if best >= 0:
                pairs.add((min(i, best), max(i, best)))
    return sorted(pairs)


def _floor_vanishing_point(boxes: list[Box], pairs) -> np.ndarray | None:
    """Every floor of a wall runs along a line, and in a photo all those lines meet at one point (the
    vanishing point); straight on, they are parallel and it sits at infinity. Estimate it from the
    lines through side-by-side windows, ignoring stray pairs (RANSAC), then refine with least squares.
    Returned as a homogeneous 3-vector, or None when there are too few pairs."""
    if len(pairs) < 4:
        return None
    pts = [np.array([(b.x0 + b.x1) / 2, b.cy, 1.0]) for b in boxes]
    lines, segs = [], []
    for i, j in pairs:
        l = np.cross(pts[i], pts[j])
        nrm = np.hypot(l[0], l[1])
        if nrm < 1e-9:
            continue
        lines.append(l / nrm)
        segs.append((pts[i][:2], pts[j][:2]))
    if len(lines) < 4:
        return None
    L = np.array(lines)

    def inliers(v):
        out = []
        for k, (p, q) in enumerate(segs):
            d = q - p
            mid = (p + q) / 2
            if abs(v[2]) > 1e-9:
                to = v[:2] / v[2] - mid
            else:
                to = v[:2]
            c = abs(np.dot(d, to)) / (np.linalg.norm(d) * np.linalg.norm(to) + 1e-12)
            if c > np.cos(np.radians(2.5)):
                out.append(k)
        return out

    rng = np.random.default_rng(0)  # fixed seed: the same photo always gives the same answer
    best = []
    for _ in range(min(400, len(lines) * (len(lines) - 1) // 2)):
        a, b = rng.choice(len(lines), 2, replace=False)
        v = np.cross(L[a], L[b])
        if np.linalg.norm(v) < 1e-12:
            continue
        inl = inliers(v)
        if len(inl) > len(best):
            best = inl
    if len(best) < 3:
        return None
    _, _, vt = np.linalg.svd(L[best])
    return vt[-1]


def vp_rows(boxes: list[Box]) -> tuple[list[list[Box]], bool]:
    """Group windows into floors using the wall's own perspective.

    Each window is slid along the line toward the floors' vanishing point until it reaches one vertical
    reference line. Windows on the same floor land on top of each other there, however steeply the photo
    was taken; different floors land apart. Windows whose landed spans overlap form a row.
    Deterministic. Returns (rows bottom-first, is_grid)."""
    n = len(boxes)
    if n == 0:
        return [], False
    pairs = _same_floor_pairs(boxes)
    v = _floor_vanishing_point(boxes, pairs)
    xref = float(np.median([(b.x0 + b.x1) / 2 for b in boxes]))

    def land(x, y):
        if v is None:
            return y
        if abs(v[2]) > 1e-9:
            vx, vy = v[0] / v[2], v[1] / v[2]
            if abs(x - vx) < 1e-6:
                return y
            return vy + (y - vy) * (xref - vx) / (x - vx)
        if abs(v[0]) < 1e-9:
            return y
        return y + (xref - x) * v[1] / v[0]

    spans = []
    for b in boxes:
        cx = (b.x0 + b.x1) / 2
        t0, t1 = land(cx, b.y0), land(cx, b.y1)
        spans.append((min(t0, t1), max(t0, t1)))
    # Windows that sit on none of this wall's floor lines (another wall seen at a corner, a stray box)
    # would bridge floors together. Build the floors from the windows that do, then hang the rest on the
    # floor whose landed position is nearest.
    core = set(range(n))
    # Only on a big facade: on a small house a window with no side-by-side partner (an attic window)
    # is normal and must keep its own floor.
    if v is not None and n >= 20 and len(pairs) >= 6:
        fits = _pair_fits(boxes, pairs, v)
        on_line = {i for (i, j), ok in zip(pairs, fits) if ok for i in (i, j)}
        if len(on_line) >= 0.6 * n:
            core = on_line
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    core_l = sorted(core)
    for a_i, i in enumerate(core_l):
        for j in core_l[a_i + 1:]:
            (a0, a1), (b0, b1) = spans[i], spans[j]
            ov = min(a1, b1) - max(a0, b0)
            if ov > 0.5 * min(a1 - a0, b1 - b0):
                parent[find(i)] = find(j)
    mid = [(t0 + t1) / 2 for t0, t1 in spans]
    for i in range(n):
        if i not in core and core_l:
            j = min(core_l, key=lambda c: abs(mid[c] - mid[i]))
            parent[find(i)] = find(j)
    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    order = sorted(groups.values(), key=lambda g: np.median([(spans[i][0] + spans[i][1]) / 2 for i in g]), reverse=True)
    rows = [[boxes[i] for i in g] for g in order]
    # A "grid" is a tall regular facade (apartment block, office): many rows of several windows each.
    # There, the geometry beats the model's reading. A house with a few uneven rows is not a grid; there
    # the model's judgement of which rows share a storey is the better one.
    big = [r for r in rows if len(r) >= 3]
    is_grid = n >= 24 and len(big) >= 6 and sum(len(r) for r in big) >= 0.7 * n
    return rows, is_grid


def _rows_picture(img: Image.Image, rows: list[list[Box]]) -> Image.Image:
    """The photo with each row's windows outlined in its own colour and a big R1, R2... label beside it."""
    out = img.convert("RGB").copy()
    d = ImageDraw.Draw(out)
    f = _font(max(18, img.width // 45))
    lw = max(3, img.width // 400)
    for k, row in enumerate(rows):
        col = FLOOR_COLOURS[k % len(FLOOR_COLOURS)]
        for b in row:
            d.rectangle([b.x0, b.y0, b.x1, b.y1], outline=col, width=lw)
        x = min(b.x0 for b in row)
        y = float(np.median([b.cy for b in row])) - f.size / 2
        label = f"R{k + 1}"
        tw = d.textlength(label, font=f)
        lx = max(0, x - tw - 14)
        d.rectangle([lx, y - 3, lx + tw + 10, y + f.size + 5], fill=col)
        d.text((lx + 5, y), label, font=f, fill=(255, 255, 255))
    # vertical grid 0..10 so the model can say where the wall starts and ends without guessing pixels
    W, H = out.size
    g = _font(max(16, img.width // 55))
    for k in range(11):
        x = min(W - 2, k * W / 10)
        d.line([(x, 0), (x, H)], fill=(255, 255, 0), width=max(2, W // 700))
        t = str(k)
        tw = d.textlength(t, font=g)
        tx = min(max(0, x - tw / 2), W - tw - 8)
        d.rectangle([tx - 4, 2, tx + tw + 4, g.size + 8], fill=(0, 0, 0))
        d.text((tx, 4), t, font=g, fill=(255, 255, 0))
    return out


OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-5.4-mini")
AI_REGIONS = 3  # buildings per photo whose storeys the model reads; the rest use geometry
VOTES = max(1, int(os.environ.get("VOTES", "3")))


def _openai_json(img: Image.Image, prompt: str, model=OPENAI_MODEL, timeout=90) -> str:
    r = requests.post(
        "https://api.openai.com/v1/chat/completions",
        headers={"Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"},
        json={"model": model,
              "reasoning_effort": os.environ.get("OPENAI_EFFORT", "low"),
              "response_format": {"type": "json_object"},
              "messages": [{"role": "user", "content": [
                  {"type": "text", "text": prompt},
                  {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{_jpeg_b64(img)}",
                                                      "detail": "high"}}]}]},
        timeout=(5, _read_timeout(min(timeout, 40))))
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


def _set_floors(rows: list[list[Box]], storeys: list[int]) -> int:
    """Apply one storey per row, never going down as rows go up, then renumber 1..n using ONLY levels
    that have a counted window. A level whose windows are all on another wall or next door is not a floor
    of this side; if one of those windows is clicked back in, it joins the nearest floor at or below it."""
    fixed, cur = [], 1
    for s in storeys:
        cur = max(cur, int(s) if s else cur)
        fixed.append(cur)
    # a floor needs at least one counted window the detector was SURE about; a level made only of unsure
    # boxes (often a shadow or a porch rail) joins the nearest real floor instead of inventing one
    levels = (sorted({lvl for row, lvl in zip(rows, fixed) if any(b.counted and b.conf >= CHECK_BELOW for b in row)})
              or sorted({lvl for row, lvl in zip(rows, fixed) if any(b.counted for b in row)}) or sorted(set(fixed)))
    remap = {old: new for new, old in enumerate(levels, start=1)}
    for row, lvl in zip(rows, fixed):
        target = min(levels, key=lambda l: (abs(l - lvl), l))  # nearest real floor, lower one on a tie
        for b in row:
            b.floor = remap[target]
    return len(levels)


STOREYS_PROMPT = (
    "This photo shows ONE building wall for a window-cleaning quote. Its windows have been grouped into "
    "horizontal rows by height; each row's windows are outlined in one colour and labelled R1, R2 ... from "
    "the BOTTOM up. (Ignore the yellow grid lines.) "
    "1) STOREYS: for each row, which storey of the wall is it on? Storey 1 is the lowest level with windows. "
    "Rows on the same level are the same storey (a small window and a tall bay window beside it, a porch "
    "window and a main window). A dormer, gable or turret-top window is on the storey it sits at. A higher row "
    "is never on a lower storey than a row below it. Rows: {rows}. "
    "2) TOTAL: how many storeys does this wall have, counting every level that has windows? "
    'Return JSON only: {{"storeys": {{"R1": 1, "R2": 2}}, "total_storeys": 2}}')


def _ask_votes(picture: Image.Image, prompt: str, provider: str) -> list[dict]:
    """Ask VOTES times in parallel; one reading varies run to run, the median of three barely does."""
    from concurrent.futures import ThreadPoolExecutor
    ask = (lambda: _openai_json(picture, prompt)) if provider == "openai" else (lambda: _gemini_json(GEMINI_MODEL, picture, prompt))
    with ThreadPoolExecutor(VOTES) as ex:
        futs = [ex.submit(ask) for _ in range(VOTES)]
    answers = []
    for f in futs:
        try:
            a = json.loads(f.result())
        except Exception:
            continue
        if isinstance(a, dict) and isinstance(a.get("storeys"), dict):
            answers.append(a)
    if not answers:
        raise RuntimeError("no usable answer from the vision model")
    return answers


def _as_int(v, default=0) -> int:
    try:
        return int(round(float(v)))
    except (TypeError, ValueError):
        return default


def floors_for_wall(img: Image.Image, boxes: list[Box], provider="openai") -> tuple[int, int | None]:
    """Floors for ONE wall the viewer picked: geometry makes the rows, the model says which rows share a
    storey (a tall regular facade trusts the geometry instead). The model never adds or removes a window."""
    live = [b for b in boxes if b.counted]
    if not live:
        for b in boxes:
            b.floor = 1
        return 0, None
    rows, is_grid = vp_rows(live)
    if len(rows) == 1:
        n = _set_floors(rows, [1])
        seen = None
    else:
        med = lambda xs: float(np.median(xs))
        seen = None
        storeys = list(range(1, len(rows) + 1))
        if not is_grid:
            xs0 = min(b.x0 for b in live); xs1 = max(b.x1 for b in live)
            ys0 = min(b.y0 for b in live); ys1 = max(b.y1 for b in live)
            mx, my = 0.25 * (xs1 - xs0) + 40, 0.25 * (ys1 - ys0) + 40
            crop = (int(max(0, xs0 - mx)), int(max(0, ys0 - my)), int(min(img.width, xs1 + mx)), int(min(img.height, ys1 + my)))
            picture = _rows_picture(img, rows).crop(crop)
            prompt = STOREYS_PROMPT.format(rows=", ".join(f"R{k + 1}" for k in range(len(rows))))
            answers = _ask_votes(picture, prompt, provider)
            storeys = [int(round(med([_as_int(a["storeys"].get(f"R{k + 1}")) for a in answers])))
                       for k in range(len(rows))]
            totals = [_as_int(a.get("total_storeys")) for a in answers if _as_int(a.get("total_storeys")) > 0]
            seen = int(round(med(totals))) if totals else None
        n = _set_floors(rows, storeys)
    if n >= 3:
        n = _merge_orphan_floors(live)
        if n >= 3:
            n = _merge_close_floors(live)
    if seen and seen < n:
        n = _merge_to(live, seen)
    # windows not counted (edge, weak) still need a floor in case someone clicks them in
    for b in boxes:
        if not b.counted:
            near = min(live, key=lambda c: abs(c.cy - b.cy) + 0.2 * abs((c.x0 + c.x1) / 2 - (b.x0 + b.x1) / 2))
            b.floor = near.floor
    return n, seen


def _floor_the_rest(boxes: list[Box]):
    """Windows not counted (edge, low confidence) still get the floor of the nearest counted window, so
    if the viewer clicks one back in it is priced on a real floor."""
    live = [b for b in boxes if b.counted and b.floor > 0]
    for b in boxes:
        if (not b.counted or b.floor <= 0) and live:
            near = min(live, key=lambda c: abs(c.cy - b.cy) + 0.2 * abs((c.x0 + c.x1) / 2 - (b.x0 + b.x1) / 2))
            b.floor = near.floor
        elif b.floor <= 0:
            b.floor = 1


def segment_buildings(img: Image.Image, thresh=0.4) -> list[list[list[float]]]:
    """SAM 3 'building' instances: one polygon (list of [x, y]) per separate building, biggest first.
    Near-duplicates are dropped. Walls of the SAME building are not split here (see _split_walls)."""
    prompts = [{"type": "text", "text": "building"}]
    data = _roboflow_post({"image": {"type": "base64", "value": _jpeg_b64(img)}, "prompts": prompts,
                           "output_prob_thresh": thresh, "format": "polygon"})
    polys = []
    for p in data["prompt_results"][0]["predictions"]:
        rings = [m for m in p["masks"] if len(m) >= 3]
        if rings:
            polys.append(max(rings, key=lambda m: _poly_area(m)))
    polys.sort(key=_poly_area, reverse=True)
    kept, masks = [], []
    for poly in polys:
        m = _raster(poly, img.size)
        if any((m & k).sum() > 0.6 * min(m.sum(), k.sum()) for k in masks):
            continue
        kept.append(poly)
        masks.append(m)
    return kept


def _poly_area(poly) -> float:
    a = np.array(poly, dtype=float)
    return 0.5 * abs(np.dot(a[:, 0], np.roll(a[:, 1], 1)) - np.dot(a[:, 1], np.roll(a[:, 0], 1)))


_RSCALE = 4  # masks are rasterised at 1/4 size: plenty to decide which building a window sits on


def _raster(poly, size) -> np.ndarray:
    W, H = size
    im = Image.new("L", (W // _RSCALE + 1, H // _RSCALE + 1), 0)
    ImageDraw.Draw(im).polygon([(x / _RSCALE, y / _RSCALE) for x, y in poly], fill=1)
    return np.array(im, dtype=bool)


def _hull(points) -> list[list[float]]:
    """Convex hull (monotone chain), used as the outline of a wall made of its windows."""
    pts = sorted(set((float(x), float(y)) for x, y in points))
    if len(pts) < 3:
        return [list(p) for p in pts]
    cross = lambda o, a, b: (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    lower, upper = [], []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return [list(p) for p in lower[:-1] + upper[:-1]]


def _pair_fits(boxes: list[Box], pairs, v, tol_deg=4.0) -> list[bool]:
    """Does each side-by-side pair run toward vanishing point v (i.e. lie along a floor of that wall)?"""
    out = []
    for i, j in pairs:
        p = np.array([(boxes[i].x0 + boxes[i].x1) / 2, boxes[i].cy])
        q = np.array([(boxes[j].x0 + boxes[j].x1) / 2, boxes[j].cy])
        d, mid = q - p, (p + q) / 2
        to = v[:2] / v[2] - mid if abs(v[2]) > 1e-9 else v[:2]
        c = abs(np.dot(d, to)) / (np.linalg.norm(d) * np.linalg.norm(to) + 1e-12)
        out.append(bool(c > np.cos(np.radians(tol_deg))))
    return out


def _corner_x(boxes: list[Box]) -> float | None:
    """Where does a corner photo turn from one wall to the other?

    Fit the floors' vanishing point for the whole building. On one wall the side-by-side pairs fit it
    everywhere (a few stray diagonal pairs aside), so there is no corner. At a real corner there is a
    vertical line with pairs that fit on one side and pairs that do not on the other, and those others
    agree on a vanishing point of their own. None when the building shows a single wall."""
    pairs = _same_floor_pairs(boxes)
    if len(pairs) < 8:
        return None
    v1 = _floor_vanishing_point(boxes, pairs)
    if v1 is None:
        return None
    fit1 = np.array(_pair_fits(boxes, pairs, v1))
    mids = np.array([((boxes[i].x0 + boxes[i].x1) / 2 + (boxes[j].x0 + boxes[j].x1) / 2) / 2 for i, j in pairs])
    order = np.argsort(mids)
    mids, fit1 = mids[order], fit1[order]
    pairs = [pairs[k] for k in order]
    best = None
    for k in range(4, len(pairs) - 3):  # at least 4 pairs each side
        for main_left in (True, False):
            A = fit1[:k] if main_left else fit1[k:]
            B = fit1[k:] if main_left else fit1[:k]
            score = A.mean() - B.mean()
            if best is None or score > best[0]:
                best = (score, k, main_left, A.mean(), B.mean(), len(B))
    if best is None:
        return None
    score, k, main_left, ra, rb, nb = best
    if not (ra >= 0.6 and rb <= 0.35 and nb >= 4):
        return None
    other = pairs[k:] if main_left else pairs[:k]
    v2 = _floor_vanishing_point(boxes, other)
    if v2 is None or np.mean(_pair_fits(boxes, other, v2)) < 0.6:
        return None
    return float((mids[k - 1] + mids[k]) / 2)


def _clip_x(poly, sx: float, keep_left: bool):
    """Cut a polygon along the vertical line x = sx (Sutherland-Hodgman), keeping one side."""
    inside = (lambda p: p[0] <= sx) if keep_left else (lambda p: p[0] >= sx)
    out = []
    n = len(poly)
    for k in range(n):
        cur, nxt = poly[k], poly[(k + 1) % n]
        ci, ni = inside(cur), inside(nxt)
        if ci:
            out.append(cur)
        if ci != ni:
            t = (sx - cur[0]) / ((nxt[0] - cur[0]) or 1e-9)
            out.append([sx, cur[1] + t * (nxt[1] - cur[1])])
    return out


def _split_walls(boxes: list[Box]) -> tuple[list[list[Box]], float | None]:
    """Split one building's windows at the corner line, if the photo shows two of its walls."""
    firm = [b for b in boxes if not b.weak] or boxes
    if len(firm) < 8:
        return [boxes], None
    sx = _corner_x(firm)
    if sx is None:
        return [boxes], None
    left = [b for b in boxes if (b.x0 + b.x1) / 2 < sx]
    right = [b for b in boxes if (b.x0 + b.x1) / 2 >= sx]
    if sum(not b.weak for b in left) < 4 or sum(not b.weak for b in right) < 4:
        return [boxes], None
    return [left, right], sx


def find_regions(img: Image.Image, boxes: list[Box], polys) -> list[dict]:
    """Every building (and, in a corner shot, every wall) in the photo, numbered left to right, each with
    its windows. Sets box.region. The viewer picks which ones this side's quote covers."""
    masks = [_raster(p, img.size) for p in polys]
    mask_pts = []
    for m in masks:  # every other pixel is plenty for "which building is nearest"
        ys, xs = np.nonzero(m[::2, ::2])
        mask_pts.append((xs * 2, ys * 2))
    groups: dict[int, list[Box]] = {}
    for b in boxes:
        k = -1
        if masks:
            x = int(((b.x0 + b.x1) / 2) / _RSCALE)
            y = int(b.cy / _RSCALE)
            hits = [i for i, m in enumerate(masks) if 0 <= y < m.shape[0] and 0 <= x < m.shape[1] and m[y, x]]
            if hits:
                k = min(hits, key=lambda i: masks[i].sum())  # the smaller building in front wins
            else:
                def dist(i):
                    xs, ys = mask_pts[i]
                    return float(np.min((xs - x) ** 2 + (ys - y) ** 2)) if len(xs) else 1e18
                k = min(range(len(masks)), key=dist)
        groups.setdefault(k, []).append(b)
    regions = []
    for k, bs in groups.items():
        walls, sx = _split_walls(bs)
        for wi, wall in enumerate(walls):
            if k >= 0:
                poly = polys[k] if sx is None else _clip_x(polys[k], sx, keep_left=(wi == 0))
            else:
                pts = [(x, y) for b in wall for x, y in ((b.x0, b.y0), (b.x1, b.y0), (b.x1, b.y1), (b.x0, b.y1))]
                poly = _hull(pts)
            if len(poly) < 3:
                pts = [(x, y) for b in wall for x, y in ((b.x0, b.y0), (b.x1, b.y0), (b.x1, b.y1), (b.x0, b.y1))]
                poly = _hull(pts)
            regions.append({"boxes": wall, "polygon": [[float(x), float(y)] for x, y in poly],
                            "wall": (wi if len(walls) > 1 else None), "building": k})
    regions.sort(key=lambda r: np.median([(b.x0 + b.x1) / 2 for b in r["boxes"]]))
    bnum = {}
    for i, r in enumerate(regions):
        r["id"] = i + 1
        for b in r["boxes"]:
            b.region = i + 1
    for r in regions:
        bnum.setdefault(r["building"], len(bnum) + 1)
    multi_b = len(bnum) > 1
    for r in regions:
        label = f"Building {bnum[r['building']]}" if multi_b else "Building"
        if r["wall"] is not None:
            label += f", wall {'AB'[r['wall']]}"
        r["label"] = label
        r["windows"] = sum(1 for b in r["boxes"] if b.counted)
    return regions


def _merge_close_floors(boxes: list[Box]) -> int:
    """Two 'floors' much closer together than the building's usual floor height are one storey seen at
    two heights (a gable window a little below a turret window). Merge them. Uses the building's own
    spacing between floors, so a tall facade with even floors is never touched."""
    while True:
        fl = sorted({b.floor for b in boxes if b.counted})
        if len(fl) < 3:
            return len(fl)
        cy = {f: float(np.median([b.cy for b in boxes if b.counted and b.floor == f])) for f in fl}
        gaps = [abs(cy[fl[k]] - cy[fl[k + 1]]) for k in range(len(fl) - 1)]
        pitch = float(np.median(gaps))
        k = int(np.argmin(gaps))
        if gaps[k] >= 0.45 * pitch:
            break
        a, b = fl[k], fl[k + 1]
        na = sum(1 for x in boxes if x.counted and x.floor == a)
        nb = sum(1 for x in boxes if x.counted and x.floor == b)
        small, into = (a, b) if na < nb else (b, a)
        for x in boxes:
            if x.floor == small:
                x.floor = into
    fl = sorted({b.floor for b in boxes if b.counted})
    remap = {f: i + 1 for i, f in enumerate(fl)}
    for b in boxes:
        b.floor = remap.get(b.floor, 1)
    return len(fl)


def _merge_orphan_floors(boxes: list[Box]) -> int:
    """A 'floor' of one or two windows sandwiched between two full floors (4+ windows each) is almost
    always a small side window at an in-between height, not a storey. Merge it into the nearer neighbour.
    Top and bottom floors are never touched: a one-window attic or a two-window basement is real."""
    while True:
        fl = sorted({b.floor for b in boxes if b.counted})
        cnt = {f: sum(1 for b in boxes if b.counted and b.floor == f) for f in fl}
        cy = {f: float(np.median([b.cy for b in boxes if b.counted and b.floor == f])) for f in fl}
        hit = None
        for k in range(1, len(fl) - 1):
            f, lo, hi = fl[k], fl[k - 1], fl[k + 1]
            if cnt[f] <= 2 and cnt[lo] >= 4 and cnt[hi] >= 4:
                hit = (f, min((lo, hi), key=lambda g: abs(cy[g] - cy[f])))
                break
        if not hit:
            break
        for b in boxes:
            if b.floor == hit[0]:
                b.floor = hit[1]
    fl = sorted({b.floor for b in boxes if b.counted})
    remap = {f: i + 1 for i, f in enumerate(fl)}
    for b in boxes:
        b.floor = remap.get(b.floor, 1)
    return len(fl)


def _merge_to(boxes: list[Box], target: int) -> int:
    """The building reads as fewer storeys than we found: merge the floor with the fewest counted windows
    into whichever neighbouring floor sits closer to it, and repeat until the counts agree."""
    def floors_now():
        return sorted({b.floor for b in boxes if b.counted})
    fl = floors_now()
    while len(fl) > max(1, target):
        counts = {f: sum(1 for b in boxes if b.counted and b.floor == f) for f in fl}
        small = min(fl, key=lambda f: (counts[f], f))
        if counts[small] > 2:  # the model's storey count never outvotes a floor of real windows
            break
        k = fl.index(small)
        cy = lambda f: float(np.median([b.cy for b in boxes if b.counted and b.floor == f]))
        nbrs = [fl[j] for j in (k - 1, k + 1) if 0 <= j < len(fl)]
        into = min(nbrs, key=lambda f: abs(cy(f) - cy(small)))
        for b in boxes:
            if b.floor == small:
                b.floor = into
        fl = floors_now()
    remap = {f: i + 1 for i, f in enumerate(fl)}
    for b in boxes:
        b.floor = remap.get(b.floor, min(remap.values(), key=lambda v: v) if remap else 1)
    return len(fl)


def assign_floors_rows(boxes: list[Box]) -> int:
    """No-AI fallback: every row is its own storey."""
    rows, _ = vp_rows(boxes)
    return _set_floors(rows, list(range(1, len(rows) + 1))) if rows else 0


# ---------- 3. price ----------

# A box below this detector confidence is shown dashed and its floor is flagged "check this".
CHECK_BELOW = 0.55
# Below this, a box is a suggestion only (vents, trim, leaves): drawn, not counted until clicked in.
WEAK_BELOW = float(os.environ.get("WEAK_BELOW", "0.5"))


@dataclass
class FloorLine:
    floor: int
    windows: int
    price_each: float
    subtotal: float
    check: bool = False


@dataclass
class SideResult:
    name: str
    boxes: list[Box]
    floors: list[FloorLine] = field(default_factory=list)
    note: str = ""
    floors_check: bool = False  # the photo reads as a different number of storeys than were found
    regions: list = field(default_factory=list)   # buildings / walls in the photo (see find_regions)
    selected: list = field(default_factory=list)  # region ids the quote covers by default

    @property
    def total(self): return sum(f.subtotal for f in self.floors)
    @property
    def windows(self): return sum(f.windows for f in self.floors)


def price(boxes: list[Box], n_floors: int, base=BASE_PRICE, mult=MULTIPLIER, selected=None) -> list[FloorLine]:
    if selected is not None:
        boxes = [b for b in boxes if b.region in selected]
        n_floors = max([b.floor for b in boxes if b.counted] or [0])
    lines = []
    for n in range(1, n_floors + 1):
        fb = [b for b in boxes if b.floor == n and b.counted]
        each = base * mult ** (n - 1)
        lines.append(FloorLine(n, len(fb), each, len(fb) * each, any(b.conf < CHECK_BELOW for b in fb)))
    return lines


# ---------- 4. draw ----------

def _font(size):
    for f in ("arialbd.ttf", "arial.ttf", "DejaVuSans-Bold.ttf"):
        try:
            return ImageFont.truetype(f, size)
        except OSError:
            pass
    return ImageFont.load_default()


def _is_level(fb: list[Box]) -> bool:
    """A floor is 'level' in the photo when its windows' tops line up; on a photo taken looking up, a
    floor slopes and a horizontal band across the picture would point at the wrong windows."""
    if len(fb) < 2:
        return True
    h = float(np.median([b.h for b in fb]))
    return (max(b.cy for b in fb) - min(b.cy for b in fb)) < 0.8 * h


def draw(img: Image.Image, res: SideResult) -> Image.Image:
    """Bands first, then every box on top, so no floor's band can paint over another floor's boxes.
    A sloped floor gets a line through its windows instead of a band."""
    out = img.convert("RGBA").copy()
    W, H = out.size
    lw = max(2, W // 500)
    f = _font(max(14, W // 60))
    layers = []
    chosen = set(res.selected) if res.regions else None
    for fl in res.floors:
        fb = [b for b in res.boxes if b.floor == fl.floor and b.counted and (chosen is None or b.region in chosen)]
        if fb:
            layers.append((fl, fb, FLOOR_COLOURS[(fl.floor - 1) % len(FLOOR_COLOURS)]))
    bands = Image.new("RGBA", out.size, (0, 0, 0, 0))
    bd = ImageDraw.Draw(bands)
    for fl, fb, col in layers:
        if _is_level(fb):
            bd.rectangle([0, min(b.y0 for b in fb), W, max(b.y1 for b in fb)], fill=col + (38,))
    out = Image.alpha_composite(out, bands)
    d = ImageDraw.Draw(out)
    for fl, fb, col in layers:
        if not _is_level(fb):
            pts = [((b.x0 + b.x1) / 2, b.cy) for b in sorted(fb, key=lambda b: b.x0 + b.x1)]
            d.line(pts, fill=col + (230,), width=lw * 2)
    for b in res.boxes:
        if not b.counted or (chosen is not None and b.region not in chosen):
            d.rectangle([b.x0, b.y0, b.x1, b.y1], outline=(255, 255, 255, 200), width=lw)
    for fl, fb, col in layers:
        for b in fb:
            if b.conf < CHECK_BELOW:  # unsure: amber, thicker, so it stands out for a human check
                d.rectangle([b.x0, b.y0, b.x1, b.y1], outline=(255, 170, 0, 255), width=lw * 2)
            else:
                d.rectangle([b.x0, b.y0, b.x1, b.y1], outline=col + (255,), width=lw)
    for fl, fb, col in layers:
        lead = min(fb, key=lambda b: b.x0)
        top = lead.y0 if not _is_level(fb) else min(b.y0 for b in fb)
        lx = 0 if _is_level(fb) else max(0, lead.x0 - 4)
        label = f" Floor {fl.floor}: {fl.windows} x ${fl.price_each:g} = ${fl.subtotal:g}{'  CHECK' if fl.check else ''} "
        tw = d.textlength(label, font=f)
        lx = min(lx, W - tw - 10)
        ty = max(0, top - (0 if _is_level(fb) else f.size + 10))
        d.rectangle([lx, ty, lx + tw + 8, ty + f.size + 8], fill=col + (235,))
        d.text((lx + 4, ty + 3), label, font=f, fill=(255, 255, 255, 255))
    out = out.convert("RGB")
    d = ImageDraw.Draw(out)
    total = f" {res.windows} windows | total ${res.total:g} "
    tw = d.textlength(total, font=f)
    d.rectangle([W - tw - 12, H - f.size - 16, W, H], fill=(8, 20, 45))
    d.text((W - tw - 6, H - f.size - 10), total, font=f, fill=(255, 255, 255))
    return out


# ---------- pipeline ----------

def run_side(img: Image.Image, name="side", detector="sam3", thresh=0.4,
             base=BASE_PRICE, mult=MULTIPLIER, floors="ai") -> SideResult:
    """Windows -> buildings/walls in the photo -> floors worked out separately for each one.

    The photo can show several buildings, or two walls of one building at a corner. Each becomes a
    numbered region with its own floors (so two buildings of different heights never mix). The quote
    covers the region(s) the viewer picks; by default the one with the most windows."""
    from concurrent.futures import ThreadPoolExecutor
    start_budget()
    detect_sam3.failed_tiles = 0
    with ThreadPoolExecutor(2) as ex:
        f_win = ex.submit(lambda: detect_sam3(img, min(thresh, CAND_THRESH) if VERIFY else thresh) if detector == "sam3" else detect_gemini(img))
        f_bld = ex.submit(lambda: segment_buildings(img))
        raw = f_win.result()
        try:
            polys = f_bld.result()
        except Exception:
            polys = []  # no building outlines: the whole photo is one region
    doors = [b for b in raw if b.kind == "door"]
    raw = [b for b in raw if b.kind != "door"]
    if doors:
        raw = _drop_door_lights(raw, doors)
    boxes = clean(raw)
    if len(boxes) > MAX_BOXES:  # more than any real facade: keep the most confident
        boxes = sorted(boxes, key=lambda b: b.conf, reverse=True)[:MAX_BOXES]
    # A window sliced by the left or right edge of the photo belongs to the neighbour nearly every time.
    # Kept and drawn, but not counted unless clicked back.
    W = img.width
    for b in boxes:
        b.edge = bool(b.x0 < 0.012 * W or b.x1 > 0.988 * W)
        b.weak = bool(b.conf < WEAK_BELOW)
    if VERIFY and floors == "ai" and os.environ.get("OPENAI_API_KEY"):
        # Doubtful detections (between the candidate threshold and WEAK_BELOW) get a close-up check.
        strong = [b for b in boxes if not b.weak]
        # Only genuinely new candidates: anything overlapping a confident window is a pane, a duplicate
        # or a frame of it, and a close-up of a pane looks like a window, so those are never asked about.
        doubtful = [b for b in boxes if b.weak and not b.edge and b.conf >= CAND_THRESH
                    and all(_inter(b, t) < 0.1 * min(b.area, t.area) for t in strong)]
        try:
            verify_candidates(img, doubtful)
        except Exception:
            pass  # unverified doubtful boxes just stay uncounted, as before
    regions = find_regions(img, boxes, polys) if boxes else []
    res = SideResult(name, boxes)
    res.regions = regions

    # The model reads storeys for the few biggest buildings only; small ones in the background get
    # geometry. Caps what one photo can cost however many buildings it shows.
    ai_ids = {r["id"] for r in sorted(regions, key=lambda r: -r["windows"])[:AI_REGIONS] if r["windows"] > 0}

    def do(region):
        bs = region["boxes"]
        if floors != "ai" or region["id"] not in ai_ids:
            live = [b for b in bs if b.counted]
            n = assign_floors_rows(live) if live else 0
            _floor_the_rest(bs)
            return n, None, ""
        errs = []
        for provider in ("openai", "gemini"):
            if not os.environ.get("OPENAI_API_KEY" if provider == "openai" else "GEMINI_API_KEY"):
                continue
            try:
                n, seen = floors_for_wall(img, bs, provider)
                _floor_the_rest(bs)
                return n, seen, ""
            except Exception as e:  # never let one flaky API sink the quote
                errs.append(f"{provider}: {str(e)[:60]}")
        live = [b for b in bs if b.counted]
        n = assign_floors_rows(live) if live else 0
        _floor_the_rest(bs)
        return n, None, "Floors from geometry only (the AI check was unavailable: " + "; ".join(errs) + ")."

    with ThreadPoolExecutor(max(1, len(regions))) as ex:
        outs = list(ex.map(do, regions))
    for r, (n, seen, note) in zip(regions, outs):
        r["n_floors"] = n
        r["floors_check"] = bool(seen and n and seen < n)
        if r["floors_check"]:
            note += (f" Check the floors: this reads as {seen} storeys but {n} floors were found. "
                     "If two colours are one storey, use merge on the upper one.")
        elif seen and n and seen > n:
            note += f" This may have {seen} storeys. If a floor was missed, use + Add a floor."
        r["note"] = note.strip()
    with_windows = [r for r in regions if r["windows"] > 0]
    res.selected = [max(with_windows, key=lambda r: r["windows"])["id"]] if with_windows else []
    sel = [r for r in regions if r["id"] in res.selected]
    if getattr(detect_sam3, "failed_tiles", 0):
        for r in regions:
            r["note"] = (r["note"] + " Part of the photo could not be checked closely, so small windows "
                         "may be missing. Count again, or check the floors.").strip()
    res.floors_check = any(r["floors_check"] for r in sel) or bool(getattr(detect_sam3, "failed_tiles", 0))
    res.note = " ".join(r["note"] for r in sel if r["note"])
    res.floors = price(boxes, 0, base, mult, selected=set(res.selected))
    return res


def run_area(img: Image.Image, boxes: list[Box], area, base=BASE_PRICE, mult=MULTIPLIER, floors="ai") -> SideResult:
    """The viewer drew the exact area to count (e.g. one wall of a corner shot). Reuse the windows already
    found, count only those whose centre is inside the area, and work out floors for just them."""
    x0, y0, x1, y1 = [float(v) for v in area]
    x0, x1 = sorted((x0, x1)); y0, y1 = sorted((y0, y1))
    inside = [b for b in boxes if x0 <= (b.x0 + b.x1) / 2 <= x1 and y0 <= b.cy <= y1]
    outside = [b for b in boxes if b not in inside]
    for b in inside:
        b.region, b.edge, b.other_side = 1, False, False
    for b in outside:
        b.region = 2
    start_budget()
    res = SideResult("area", boxes)
    n, seen, note = 0, None, ""
    live = [b for b in inside if b.counted]
    try:
        if floors != "ai":
            n = assign_floors_rows(live) if live else 0
        else:
            n, seen = floors_for_wall(img, inside) if inside else (0, None)
    except Exception as e:
        n = assign_floors_rows(live) if live else 0
        note = f"Floors from geometry only (the AI check was unavailable: {str(e)[:60]})."
    _floor_the_rest(inside)
    rect = [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]
    reg = {"id": 1, "label": "Your area", "polygon": rect, "boxes": inside, "windows": sum(1 for b in inside if b.counted),
           "n_floors": n, "floors_check": bool(seen and n and seen < n), "note": note}
    if reg["floors_check"]:
        reg["note"] += f" Check the floors: this reads as {seen} storeys but {n} floors were found."
    res.regions = [reg]
    if outside:
        res.regions.append({"id": 2, "label": "Outside your area", "polygon": [], "boxes": outside, "windows": 0,
                            "n_floors": 0, "floors_check": False, "note": ""})
        for b in outside:
            b.floor = 1
    res.selected = [1]
    res.floors_check = reg["floors_check"]
    res.note = reg["note"].strip()
    res.floors = price(boxes, 0, base, mult, selected={1})
    return res

