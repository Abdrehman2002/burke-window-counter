"""Glass curtain-wall benchmark: synthetic walls with a known pane grid.

Counting 300 panes by eye is itself error-prone, so the pane-grid measurement (_core.glass_grid) is scored
on generated walls whose grid is exact: random column/row counts and pane sizes, a random camera angle
(perspective both ways), reflections (soft blobs and long straight reflected edges), dark, light,
spandrel and alternating-row frames, and lamp posts and trees in front in 40% of them. The outline passed
in is jittered like a SAM outline. No API calls, no network.

    python glass_bench.py          # 60 walls
    python glass_bench.py 200
"""
import os
import random
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "api"))
import _core as c  # noqa: E402


def _coeffs(src, dst):
    """PIL PERSPECTIVE coeffs mapping output(dst) -> input(src)."""
    M = []
    for (x, y), (u, v) in zip(dst, src):
        M.append([x, y, 1, 0, 0, 0, -u * x, -u * y]); M.append([0, 0, 0, x, y, 1, -v * x, -v * y])
    b = np.array(src).reshape(8)
    return np.linalg.solve(np.array(M, float), b).tolist()


def synth(seed):
    r = random.Random(seed)
    C, R = r.randint(5, 26), r.randint(3, 18)
    pw, ph = r.randint(40, 90), r.randint(40, 110)
    bar = r.randint(2, 6)
    style = r.choice(["dark", "dark", "light", "spandrel", "alt"])
    rh = [ph if (j % 2 == 0 or style != "alt") else max(18, int(ph * 0.4)) for j in range(R)]
    ry = np.concatenate([[0], np.cumsum(rh)]).astype(int)
    W, H = C * pw, int(ry[-1])
    base = np.array([r.randint(20, 90), r.randint(60, 140), r.randint(120, 220)], float)
    yy = np.linspace(0, 1, H)[:, None, None]
    xx = np.linspace(0, 1, W)[None, :, None]
    img = base + 60 * yy * np.array([0.3, 0.5, 0.4]) + 30 * xx * np.array([r.random() - .5, r.random() - .5, 0])
    img = np.broadcast_to(img, (H, W, 3)).copy()
    for i in range(C):
        for j in range(R):
            img[ry[j]:ry[j + 1], i * pw:(i + 1) * pw] += r.gauss(0, 14)
    # reflections: big soft blobs + a few long straight reflected edges (the nasty case)
    refl = Image.new("L", (W, H), 0); d = ImageDraw.Draw(refl)
    for _ in range(r.randint(2, 7)):
        cx, cy, rr = r.randint(0, W), r.randint(0, H), r.randint(30, 300)
        d.ellipse((cx - rr, cy - rr, cx + rr, cy + rr), fill=r.randint(20, 90))
    for _ in range(r.randint(0, 3)):
        d.line((r.randint(0, W), r.randint(0, H), r.randint(0, W), r.randint(0, H)), fill=r.randint(30, 80), width=r.randint(3, 12))
    refl = np.asarray(refl.filter(ImageFilter.GaussianBlur(6)), float)[..., None]
    img += refl
    if style == "spandrel":
        for j in range(R):
            if j % r.choice([2, 3]) == 0:
                img[ry[j]:ry[j] + rh[j] // 3] = base * 0.4
    col = (15, 20, 30) if style != "light" else (225, 230, 235)
    for i in range(C + 1):
        img[:, max(0, i * pw - bar // 2):i * pw + (bar + 1) // 2] = col
    for j in range(R + 1):
        img[max(0, ry[j] - bar // 2):ry[j] + (bar + 1) // 2] = col
    img += np.random.default_rng(seed).normal(0, 6, img.shape)
    wall = Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))
    # camera: random yaw (horizontal convergence) and pitch (looking up)
    OW, OH = 1400, 1050
    yaw, pitch = r.uniform(-0.45, 0.45), r.uniform(0, 0.3)
    sc = min(OW * 0.8 / W, OH * 0.8 / H)
    w2, h2 = W * sc, H * sc
    cx, cy = OW / 2 + r.uniform(-100, 100), OH / 2 + r.uniform(-60, 60)
    # near side taller than far side (yaw), top narrower than bottom (pitch)
    kl, kr = (1 + yaw, 1 - yaw) if yaw > 0 else (1 + yaw, 1 - yaw)
    tl = (cx - w2 / 2 * (1 - pitch * .5), cy - h2 / 2 * kl)
    tr = (cx + w2 / 2 * (1 - pitch * .5), cy - h2 / 2 * kr)
    bl = (cx - w2 / 2, cy + h2 / 2 * kl)
    br = (cx + w2 / 2, cy + h2 / 2 * kr)
    dst = [tl, tr, br, bl]
    src = [(0, 0), (W, 0), (W, H), (0, H)]
    bg = Image.fromarray(np.clip(np.random.default_rng(seed + 1).normal(150, 25, (OH, OW, 3)), 0, 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(2))
    warped = wall.transform((OW, OH), Image.PERSPECTIVE, _coeffs(src, dst), Image.BILINEAR)
    wm = Image.new("L", (OW, OH), 0); ImageDraw.Draw(wm).polygon(dst, fill=255)
    bg.paste(warped, (0, 0), wm)
    if r.random() < 0.4:  # occluders in front of the glass: lamp posts, a tree
        d2 = ImageDraw.Draw(bg)
        for _ in range(r.randint(1, 2)):
            px_ = r.uniform(tl[0], tr[0]); d2.line((px_, OH, px_ + r.uniform(-15, 15), r.uniform(OH * .3, OH * .7)), fill=(90, 90, 95), width=r.randint(5, 12))
        if r.random() < 0.5:
            tx, ty, tr_ = r.uniform(bl[0], br[0]), OH * 0.85, r.randint(60, 140)
            d2.ellipse((tx - tr_, ty - tr_, tx + tr_, ty + tr_), fill=(50, 80, 40))
    # the outline SAM would give: slightly off the true edge
    jit = lambda p: (p[0] + r.uniform(-6, 6), p[1] + r.uniform(-6, 6))
    return bg, [jit(p) for p in dst], C, R, style



if __name__ == "__main__":
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 60
    err, cols, rows = 0.0, 0, 0
    for s in range(N):
        im, poly, C, R, st = synth(s)
        panes = c.glass_grid(im, poly)
        xs = sorted({round(p[0]) for p in panes})
        n = len(panes)
        e = abs(n - C * R) / (C * R)
        err += e
        ok = "ok " if n == C * R else "   "
        print(f"{s:3d} {st:8s} truth {C:2d}x{R:2d} = {C * R:4d}   found {n:4d}  {ok}({e:4.0%})")
    print(f"mean pane-count error {err / N:.1%} over {N} walls")
