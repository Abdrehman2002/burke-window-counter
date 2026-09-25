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


def clean(boxes: list[Box]) -> list[Box]:
    """Drop duplicates and boxes that sit inside a bigger one (a pane inside its window)."""
    boxes = sorted(boxes, key=lambda b: b.area, reverse=True)
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


def assign_floors_geometric(boxes: list[Box], k=0.9) -> int:
    """Looser rule: sort by vertical centre, start a new floor when the jump from the previous
    window's centre exceeds k x the median window height. Bottom row = floor 1."""
    if not boxes:
        return 0
    med_h = float(np.median([b.h for b in boxes]))
    order = sorted(boxes, key=lambda b: b.cy, reverse=True)  # bottom first
    floor = 1
    order[0].floor = 1
    for prev, b in zip(order, order[1:]):
        if prev.cy - b.cy > k * med_h:
            floor += 1
        b.floor = floor
    return floor


def _numbered(img: Image.Image, boxes: list[Box]) -> Image.Image:
    out = img.convert("RGB").copy()
    d = ImageDraw.Draw(out)
    f = _font(max(14, img.width // 70))
    for i, b in enumerate(boxes):
        d.rectangle([b.x0, b.y0, b.x1, b.y1], outline=(255, 0, 0), width=max(2, img.width // 600))
        d.rectangle([b.x0, b.y0, b.x0 + f.size * 1.6, b.y0 + f.size + 4], fill=(255, 0, 0))
        d.text((b.x0 + 2, b.y0 + 1), str(i), font=f, fill=(255, 255, 255))
    return out


def _enforce_same_level(boxes: list[Box], min_overlap=0.6):
    """Geometry guard on the model's answer: two windows whose heights mostly overlap are on the
    same floor, whatever the model said. Each group takes its most common floor (ties go lower)."""
    parent = list(range(len(boxes)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    for i, a in enumerate(boxes):
        for j in range(i + 1, len(boxes)):
            b = boxes[j]
            ov = min(a.y1, b.y1) - max(a.y0, b.y0)
            if ov > min_overlap * min(a.h, b.h):
                parent[find(i)] = find(j)
    groups: dict[int, list[Box]] = {}
    for i, b in enumerate(boxes):
        groups.setdefault(find(i), []).append(b)
    for g in groups.values():
        votes: dict[int, int] = {}
        for b in g:
            votes[b.floor] = votes.get(b.floor, 0) + 1
        best = min(votes, key=lambda f: (-votes[f], f))
        for b in g:
            b.floor = best


STOREY_PROMPT = (
    "This photo shows a building with numbered red boxes drawn on its windows. "
    "For EVERY numbered box, say which storey of the building it is on: 1 = ground floor, 2 = the floor "
    "above, and so on. Windows in a gable, dormer or turret top count as the storey they sit at. "
    "Use the building's real floor levels, not the size of the window: a small window and a tall bay "
    "window on the same level are the same storey. The MAIN building is the one the photo is centred on. "
    "Any box on a different building, even one attached to it on the left or right, gets storey 0, and so "
    "does anything that is not a window (a car, a sign, a door). "
    'Boxes are numbered 0 to {last}. Return JSON only: {{"storeys": {{"0": 1, "1": 2, ...}}}}')

OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-5.4-mini")


def _openai_json(img: Image.Image, prompt: str, model=OPENAI_MODEL, timeout=90) -> str:
    r = requests.post(
        "https://api.openai.com/v1/chat/completions",
        headers={"Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"},
        json={"model": model,
              "reasoning_effort": "low",
              "response_format": {"type": "json_object"},
              "messages": [{"role": "user", "content": [
                  {"type": "text", "text": prompt},
                  {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{_jpeg_b64(img)}",
                                                      "detail": "high"}}]}]},
        timeout=timeout)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


def assign_floors_llm(img: Image.Image, boxes: list[Box], provider="openai") -> int:
    """Ask a vision model which storey each numbered box sits on. It classifies boxes the detector
    already found; it does not count anything, which is the job vision chatbots are bad at.
    Also drops anything it says is not a window on the main building (storey 0)."""
    if not boxes:
        return 0
    prompt = STOREY_PROMPT.format(last=len(boxes) - 1)
    picture = _numbered(img, boxes)
    text = _openai_json(picture, prompt) if provider == "openai" else _gemini_json(GEMINI_MODEL, picture, prompt)
    storeys = json.loads(text)["storeys"]
    for i, b in enumerate(boxes):
        b.floor = int(storeys.get(str(i), 0))
    boxes[:] = [b for b in boxes if b.floor > 0]
    _enforce_same_level(boxes)
    # renumber so there are no empty storeys in between (e.g. 1, 2, 4 -> 1, 2, 3)
    levels = sorted({b.floor for b in boxes})
    remap = {old: new for new, old in enumerate(levels, start=1)}
    for b in boxes:
        b.floor = remap[b.floor]
    return len(levels)


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
        fb = [b for b in boxes if b.floor == n and not b.edge]
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
        for b in [b for b in fb if b.edge]:  # not counted: dashed white
            od.rectangle([b.x0, b.y0, b.x1, b.y1], outline=(255, 255, 255, 200), width=lw)
        fb = [b for b in fb if not b.edge]
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
            n = assign_floors_geometric(boxes)
    elif floors == "geometric":
        n = assign_floors_geometric(boxes)
    else:
        n = assign_floors(boxes)
    # A window sliced by the left or right edge of the photo belongs to the neighbour nearly every time
    # (the photo is meant to frame the whole side). Kept and drawn, but not counted unless someone adds it back.
    W = img.width
    for b in boxes:
        b.edge = bool(b.x0 < 0.012 * W or b.x1 > 0.988 * W)
    res = SideResult(name, boxes)
    res.note = res_note
    res.floors = price(boxes, n, base, mult)
    return res
