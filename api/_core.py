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

    @property
    def counted(self): return not (self.edge or self.other_side)

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


def _sam3_boxes(pred_list) -> list[Box]:
    boxes = []
    for p in pred_list:
        pts = np.array([pt for poly in p["masks"] for pt in poly], dtype=float)
        if len(pts):
            boxes.append(Box(pts[:, 0].min(), pts[:, 1].min(), pts[:, 0].max(), pts[:, 1].max(), p["confidence"]))
    return boxes


def detect_sam3(img: Image.Image, thresh=0.4, prompt="window") -> list[Box]:
    prompts = [{"type": "text", "text": prompt}] + [{"type": "text", "text": t} for t in EXCLUDE_PROMPTS]
    r = requests.post(
        "https://serverless.roboflow.com/sam3/concept_segment",
        headers={"Authorization": f"Bearer {os.environ['ROBOFLOW_API_KEY']}"},
        json={"image": {"type": "base64", "value": _jpeg_b64(img)},
              "prompts": prompts, "output_prob_thresh": thresh, "format": "polygon"},
        timeout=120)
    r.raise_for_status()
    results = sorted(r.json()["prompt_results"], key=lambda x: x["prompt_index"])
    windows = _sam3_boxes(results[0]["predictions"])
    blockers = [b for res in results[1:] for b in _sam3_boxes(res["predictions"])]
    def on_blocker(w: Box) -> bool:
        cx = (w.x0 + w.x1) / 2
        return any(k.x0 <= cx <= k.x1 and k.y0 <= w.cy <= k.y1 for k in blockers)
    return [w for w in windows if not on_blocker(w)]


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
    for attempt in range(tries):
        try:
            r = requests.post(
                f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
                headers={"x-goog-api-key": os.environ["GEMINI_API_KEY"]},
                json={"contents": [{"parts": [{"inline_data": {"mime_type": "image/jpeg", "data": _jpeg_b64(img)}},
                                              {"text": prompt}]}],
                      "generationConfig": {"response_mime_type": "application/json", "temperature": 0,
                                           "thinkingConfig": {"thinkingLevel": "low"}}},
                timeout=timeout)
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
    typical = float(np.median([b.area for b in boxes]))
    groups = set()
    for i, big in enumerate(boxes):
        inside = [s for j, s in enumerate(boxes) if j != i and s.area < big.area and _inter(s, big) / s.area > 0.6]
        window_sized = [s for s in inside if s.area > 0.45 * typical]
        if len(window_sized) >= 2 and big.area > 1.6 * typical:
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


def _merge_stacked(boxes: list[Box]) -> list[Box]:
    """A tall window sometimes comes back as two boxes, top half and bottom half. Merge two boxes that
    share almost the same left and right edges and touch vertically. Windows on different floors have
    a band of wall between them, so they never touch like this."""
    merged = True
    while merged:
        merged = False
        for i, a in enumerate(boxes):
            for j, b in enumerate(boxes):
                if j <= i:
                    continue
                overlap_w = min(a.x1, b.x1) - max(a.x0, b.x0)
                if overlap_w < 0.8 * max(a.w, b.w):
                    continue
                gap = max(a.y0, b.y0) - min(a.y1, b.y1)  # negative = they overlap
                if gap < 0.08 * min(a.h, b.h):
                    boxes[i] = Box(min(a.x0, b.x0), min(a.y0, b.y0), max(a.x1, b.x1), max(a.y1, b.y1),
                                   min(a.conf, b.conf))
                    del boxes[j]
                    merged = True
                    break
            if merged:
                break
    return boxes


# ---------- 2. floors ----------

def assign_floors(boxes: list[Box]) -> int:
    """Group boxes into rows by vertical position. Bottom row = floor 1. Returns number of floors.

    A box joins a row when its vertical centre sits inside that row's typical vertical span.
    Never assumes a grid: a row of 3 and a row of 11 are both just rows.
    """
    if not boxes:
        return 0
    rows: list[list[Box]] = []
    for b in sorted(boxes, key=lambda b: b.cy):
        placed = False
        for row in rows:
            top = np.median([r.y0 for r in row])
            bot = np.median([r.y1 for r in row])
            pad = 0.15 * (bot - top)
            if top - pad <= b.cy <= bot + pad:
                row.append(b)
                placed = True
                break
        if not placed:
            rows.append([b])
    rows.sort(key=lambda row: np.median([r.cy for r in row]), reverse=True)  # bottom first
    for n, row in enumerate(rows, start=1):
        for b in row:
            b.floor = n
    return len(rows)


def build_rows(boxes: list[Box], min_overlap=0.5) -> list[list[Box]]:
    """Windows on the same storey form a row. Each window links only to its BEST neighbour on each side:
    the box beside it whose height range overlaps its own the most. A row that slopes (a photo looking
    up at a tall building) still chains along itself, but can't jump diagonally to the next storey.
    Pure geometry, so the same photo always gives the same rows. Returned bottom row first."""
    n = len(boxes)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def overlap(a: Box, c: Box) -> float:
        return (min(a.y1, c.y1) - max(a.y0, c.y0)) / min(a.h, c.h)

    for i, a in enumerate(boxes):
        acx = (a.x0 + a.x1) / 2
        for side in (-1, 1):
            best, best_score = -1, min_overlap
            for j, c in enumerate(boxes):
                if j == i or side * ((c.x0 + c.x1) / 2 - acx) <= 0:
                    continue
                ov = overlap(a, c)
                if ov <= min_overlap:
                    continue
                # prefer the most level neighbour, then the closest one
                score = ov - 0.02 * abs((c.x0 + c.x1) / 2 - acx) / max(a.w, 1)
                if score > best_score:
                    best, best_score = j, score
            if best >= 0:
                parent[find(i)] = find(best)
    groups: dict[int, list[Box]] = {}
    for i, b in enumerate(boxes):
        groups.setdefault(find(i), []).append(b)
    return sorted(groups.values(), key=lambda row: np.median([b.cy for b in row]), reverse=True)


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


ROWS_PROMPT = (
    "This photo is meant to show ONE side of a building for a window-cleaning quote. Its windows have been "
    "grouped into horizontal rows by height; each row's windows are outlined in one colour and labelled "
    "R1, R2 ... from the BOTTOM up. Yellow vertical lines numbered 0 to 10 mark positions across the photo. "
    "1) SIDE: which part of the photo is the one wall being quoted, meaning the wall that faces the camera "
    "most directly? Give the grid positions where it starts and ends. If the photo is taken at a corner and "
    "shows a second wall going off at an angle, leave that second wall out. Leave out separate buildings "
    "next door. Attached rowhouses or townhouses sharing one continuous front wall are ONE side, so include "
    "all of them. If the whole photo is one wall, answer 0 and 10. "
    "2) STOREYS: for each row, which storey of that wall is it on? Storey 1 is the lowest level with windows. "
    "Rows on the same level are the same storey (a small window and a tall bay window beside it, a porch "
    "window and a main window). A dormer, gable or turret-top window is on the storey it sits at. A higher row "
    "is never on a lower storey than a row below it. Rows: {rows}. "
    'Return JSON only: {{"side_from": 0, "side_to": 10, "storeys": {{"R1": 1, "R2": 2}}}}')

OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-5.4-mini")


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
        timeout=timeout)
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
    levels = sorted({lvl for row, lvl in zip(rows, fixed) if any(b.counted for b in row)}) or sorted(set(fixed))
    remap = {old: new for new, old in enumerate(levels, start=1)}
    for row, lvl in zip(rows, fixed):
        at_or_below = [l for l in levels if l <= lvl]
        target = at_or_below[-1] if at_or_below else levels[0]
        for b in row:
            b.floor = remap[target]
    return len(levels)


def assign_floors_llm(img: Image.Image, boxes: list[Box], provider="openai") -> int:
    """Geometry makes the rows; a vision model answers two small questions: which part of the photo is the
    one wall being quoted, and which storey each ROW is on. It never deletes or adds a window: windows off
    that wall are only marked other_side, and a click brings any of them back."""
    if not boxes:
        return 0
    rows = build_rows(boxes)
    prompt = ROWS_PROMPT.format(rows=", ".join(f"R{k + 1}" for k in range(len(rows))))
    picture = _rows_picture(img, rows)
    text = _openai_json(picture, prompt) if provider == "openai" else _gemini_json(GEMINI_MODEL, picture, prompt)
    ans = json.loads(text)
    lo = min(max(float(ans.get("side_from", 0)), 0), 10) / 10 * img.width
    hi = min(max(float(ans.get("side_to", 10)), 0), 10) / 10 * img.width
    if hi - lo < 0.15 * img.width:  # an implausibly narrow answer: ignore it rather than drop most windows
        lo, hi = 0, img.width
    for b in boxes:
        cx = (b.x0 + b.x1) / 2
        b.other_side = bool(cx < lo or cx > hi)
    return _set_floors(rows, [ans.get("storeys", {}).get(f"R{k + 1}", 0) for k in range(len(rows))])


def assign_floors_rows(boxes: list[Box]) -> int:
    """No-AI fallback: every row is its own storey."""
    rows = build_rows(boxes)
    return _set_floors(rows, list(range(1, len(rows) + 1))) if rows else 0


# ---------- 3. price ----------

# A box below this detector confidence is shown dashed and its floor is flagged "check this".
CHECK_BELOW = 0.55


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

    @property
    def total(self): return sum(f.subtotal for f in self.floors)
    @property
    def windows(self): return sum(f.windows for f in self.floors)


def price(boxes: list[Box], n_floors: int, base=BASE_PRICE, mult=MULTIPLIER) -> list[FloorLine]:
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


def draw(img: Image.Image, res: SideResult) -> Image.Image:
    out = img.convert("RGB").copy()
    overlay = Image.new("RGBA", out.size, (0, 0, 0, 0))
    od = ImageDraw.Draw(overlay)
    W, H = out.size
    lw = max(2, W // 500)
    f = _font(max(14, W // 60))
    for fl in res.floors:
        fb = [b for b in res.boxes if b.floor == fl.floor]
        if not fb:
            continue
        for b in [b for b in fb if not b.counted]:  # not counted: thin white
            od.rectangle([b.x0, b.y0, b.x1, b.y1], outline=(255, 255, 255, 200), width=lw)
        fb = [b for b in fb if b.counted]
        if not fb:
            continue
        col = FLOOR_COLOURS[(fl.floor - 1) % len(FLOOR_COLOURS)]
        top, bot = min(b.y0 for b in fb), max(b.y1 for b in fb)
        od.rectangle([0, top, W, bot], fill=col + (38,))
        for b in fb:
            if b.conf < CHECK_BELOW:  # unsure: amber, thicker, so it stands out for a human check
                od.rectangle([b.x0, b.y0, b.x1, b.y1], outline=(255, 170, 0, 255), width=lw * 2)
            else:
                od.rectangle([b.x0, b.y0, b.x1, b.y1], outline=col + (255,), width=lw)
        label = f" Floor {fl.floor}: {fl.windows} x ${fl.price_each:g} = ${fl.subtotal:g}{'  CHECK' if fl.check else ''} "
        tw = od.textlength(label, font=f)
        od.rectangle([0, top, tw + 8, top + f.size + 8], fill=col + (235,))
        od.text((4, top + 3), label, font=f, fill=(255, 255, 255, 255))
    out = Image.alpha_composite(out.convert("RGBA"), overlay).convert("RGB")
    d = ImageDraw.Draw(out)
    total = f" {res.windows} windows | total ${res.total:g} "
    tw = d.textlength(total, font=f)
    d.rectangle([W - tw - 12, H - f.size - 16, W, H], fill=(8, 20, 45))
    d.text((W - tw - 6, H - f.size - 10), total, font=f, fill=(255, 255, 255))
    return out


# ---------- pipeline ----------

def run_side(img: Image.Image, name="side", detector="sam3", thresh=0.4,
             base=BASE_PRICE, mult=MULTIPLIER, floors="ai") -> SideResult:
    raw = detect_sam3(img, thresh) if detector == "sam3" else detect_gemini(img)
    boxes = clean(raw)
    # A window sliced by the left or right edge of the photo belongs to the neighbour nearly every time
    # (the photo is meant to frame the whole side). Kept and drawn, but not counted unless clicked back.
    W = img.width
    for b in boxes:
        b.edge = bool(b.x0 < 0.012 * W or b.x1 > 0.988 * W)
    res_note = ""
    if floors == "ai":
        n = None
        for provider in ("openai", "gemini"):
            if not os.environ.get("OPENAI_API_KEY" if provider == "openai" else "GEMINI_API_KEY"):
                continue
            try:
                n = assign_floors_llm(img, boxes, provider)
                break
            except Exception as e:  # never let one flaky API sink the quote
                res_note += f"{provider} floors failed ({str(e)[:80]}). "
        if n is None:
            res_note += "Used geometric floors."
            n = assign_floors_rows(boxes)
    elif floors == "geometric":
        n = assign_floors_rows(boxes)
    else:
        n = assign_floors(boxes)
    res = SideResult(name, boxes)
    res.note = res_note
    res.floors = price(boxes, n, base, mult)
    return res
