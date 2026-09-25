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


def lattice_rows(boxes: list[Box]) -> tuple[list[list[Box]], bool]:
    """Floors the way a person counts them: go up a column of windows.

    Two local rules, each only between neighbouring windows:
      - the most level window beside you is on the SAME floor;
      - the nearest window directly above you is ONE floor up (if the gap is a normal floor's height).
    Following those links across the wall gives every window a floor. Because each rule only compares
    neighbours, sloping rows in a photo taken looking up (perspective) don't break it.
    Returns (rows bottom-first, is_grid). is_grid is True when most windows have a window above or below,
    i.e. a regular facade where this geometry can be trusted on its own."""
    n = len(boxes)
    if n == 0:
        return [], False
    cx = [(b.x0 + b.x1) / 2 for b in boxes]
    edges: list[tuple[int, int, int]] = []  # (i, j, level_j - level_i)

    # same floor: best neighbour on each side
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
                edges.append((i, best, 0))

    # one floor up: nearest window above that shares most of its width
    ups: list[tuple[int, int, float]] = []
    for i, a in enumerate(boxes):
        best, dist = -1, None
        for j, c in enumerate(boxes):
            if j == i or c.cy >= a.cy - 0.5 * min(a.h, c.h):
                continue
            xov = (min(a.x1, c.x1) - max(a.x0, c.x0)) / min(a.w, c.w)
            if xov < 0.4:
                continue
            d = a.cy - c.cy
            if dist is None or d < dist:
                best, dist = j, d
        if best >= 0:
            ups.append((i, best, dist))
    linked = set()
    pitch_all = float(np.median([d for _, _, d in ups])) if ups else float(np.median([b.h for b in boxes])) * 1.6
    if len(ups) >= 3:
        pitch = float(np.median([d for _, _, d in ups]))
        for i, j, d in ups:
            a, c = boxes[i], boxes[j]
            if 0.55 * pitch <= d <= max(1.6 * pitch, 0.95 * (a.h + c.h)):
                edges.append((i, j, 1))
                linked.update((i, j))
    is_grid = n >= 12 and len(linked) >= 0.6 * n

    # walk the links: each connected group gets relative floor numbers
    adj: dict[int, list[tuple[int, int]]] = {i: [] for i in range(n)}
    for i, j, d in edges:
        adj[i].append((j, d))
        adj[j].append((i, -d))
    level = [None] * n
    comps: list[list[int]] = []
    for root in sorted(range(n), key=lambda k: -len(adj[k])):
        if level[root] is not None:
            continue
        level[root] = 0
        comp, queue = [root], [root]
        while queue:
            u = queue.pop()
            for v, d in adj[u]:
                if level[v] is None:
                    level[v] = level[u] + d
                    comp.append(v)
                    queue.append(v)
        comps.append(comp)

    # line the groups up with each other: the biggest group is the reference; every other group is
    # shifted to agree with the placed windows it overlaps in height (or, failing that, the nearest one)
    comps.sort(key=len, reverse=True)
    placed = list(comps[0])
    for comp in comps[1:]:
        votes: dict[int, int] = {}
        for u in comp:
            a = boxes[u]
            same = [p for p in placed if (min(a.y1, boxes[p].y1) - max(a.y0, boxes[p].y0)) > 0.5 * min(a.h, boxes[p].h)]
            if same:  # level with a placed window: same floor as the nearest such one
                p = min(same, key=lambda p: abs(cx[p] - cx[u]))
                off = level[p] - level[u]
            else:     # not level with anything placed: count whole floors up or down from the nearest one
                p = min(placed, key=lambda p: abs(boxes[p].cy - a.cy) + 0.3 * abs(cx[p] - cx[u]))
                dy = boxes[p].cy - a.cy  # positive = u is higher up the photo
                steps = int(round(dy / pitch_all)) if pitch_all else 0
                if steps == 0:
                    steps = 1 if dy > 0 else -1
                off = level[p] + steps - level[u]
            votes[off] = votes.get(off, 0) + 1
        off = max(votes, key=lambda k: votes[k])
        for u in comp:
            level[u] += off
        placed.extend(comp)

    lo = min(level)
    rows: dict[int, list[Box]] = {}
    for i, b in enumerate(boxes):
        rows.setdefault(level[i] - lo, []).append(b)
    return [rows[k] for k in sorted(rows)], is_grid


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
    "3) TOTAL: how many storeys does that wall have, counting every level that has windows? "
    'Return JSON only: {{"side_from": 0, "side_to": 10, "storeys": {{"R1": 1, "R2": 2}}, "total_storeys": 2}}')

OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-5.4-mini")
VOTES = int(os.environ.get("VOTES", "3"))


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


def assign_floors_llm(img: Image.Image, boxes: list[Box], provider="openai") -> tuple[int, int | None]:
    """Geometry makes the rows; a vision model answers two small questions: which part of the photo is the
    one wall being quoted, and which storey each ROW is on. It never deletes or adds a window: windows off
    that wall are only marked other_side, and a click brings any of them back."""
    if not boxes:
        return 0, None
    rows, is_grid = lattice_rows(boxes)
    prompt = ROWS_PROMPT.format(rows=", ".join(f"R{k + 1}" for k in range(len(rows))))
    picture = _rows_picture(img, rows)
    # Ask 3 times in parallel and take the middle answer: one reading of "which wall" or "which storey"
    # varies from run to run, the median of three barely does. Same wall-clock time as one call.
    from concurrent.futures import ThreadPoolExecutor
    ask = (lambda: _openai_json(picture, prompt)) if provider == "openai" else (lambda: _gemini_json(GEMINI_MODEL, picture, prompt))
    with ThreadPoolExecutor(VOTES) as ex:
        futs = [ex.submit(ask) for _ in range(VOTES)]
    answers = []
    for f in futs:
        try:
            answers.append(json.loads(f.result()))
        except Exception:
            pass
    if not answers:
        raise RuntimeError("no usable answer from the vision model")
    med = lambda xs: float(np.median(xs))
    lo = min(max(med([float(a.get("side_from", 0)) for a in answers]), 0), 10) / 10 * img.width
    hi = min(max(med([float(a.get("side_to", 10)) for a in answers]), 0), 10) / 10 * img.width
    storey_votes = [[int(a.get("storeys", {}).get(f"R{k + 1}", 0) or 0) for a in answers] for k in range(len(rows))]
    ans = {"storeys": {f"R{k + 1}": int(round(med(v))) for k, v in enumerate(storey_votes)}}
    totals = [int(a["total_storeys"]) for a in answers if str(a.get("total_storeys", "")).isdigit()]
    seen_storeys = int(round(med(totals))) if totals else None
    if hi - lo < 0.15 * img.width:  # an implausibly narrow answer: ignore it rather than drop most windows
        lo, hi = 0, img.width
    for b in boxes:
        cx = (b.x0 + b.x1) / 2
        b.other_side = bool(cx < lo or cx > hi)
    if is_grid:  # a regular facade: the column count is more reliable than the model's reading
        n = _set_floors(rows, list(range(1, len(rows) + 1)))
    else:
        n = _set_floors(rows, [ans.get("storeys", {}).get(f"R{k + 1}", 0) for k in range(len(rows))])
    return n, seen_storeys  # seen_storeys: the model's own storey count, used as a cross-check


def assign_floors_rows(boxes: list[Box]) -> int:
    """No-AI fallback: every row is its own storey."""
    rows, _ = lattice_rows(boxes)
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
    floors_check: bool = False  # the photo reads as a different number of storeys than were found

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
    seen = None
    if floors == "ai":
        n = None
        for provider in ("openai", "gemini"):
            if not os.environ.get("OPENAI_API_KEY" if provider == "openai" else "GEMINI_API_KEY"):
                continue
            try:
                n, seen = assign_floors_llm(img, boxes, provider)
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
    if seen and n and seen != n:
        res.floors_check = True
        res.note = (res.note + " " if res.note else "") + (
            f"Floor split needs a look: the building reads as {seen} storeys but {n} floors were found. "
            "Check the colours on the photo; click a window to move it, or merge a floor into the one below.")
    res.floors = price(boxes, n, base, mult)
    return res
