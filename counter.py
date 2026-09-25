"""Command line + Streamlit entry to the window counter. The logic lives in api/_core.py so the
Vercel function and local tools share one copy.

Usage:  python counter.py photos/06_apt_karlsruhe.jpg [--crop x0,y0,x1,y1] [--thresh 0.4] [--floors ai|geometric]
"""
import argparse
import os
import sys

from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "api"))
from _core import *  # noqa: F401,F403  (re-exported for app.py)
from _core import HERE, draw, load_env, run_side


def main():
    load_env()
    ap = argparse.ArgumentParser()
    ap.add_argument("photo")
    ap.add_argument("--crop", help="x0,y0,x1,y1 in pixels, to keep only one building")
    ap.add_argument("--thresh", type=float, default=0.4)
    ap.add_argument("--detector", choices=["sam3", "gemini"], default="sam3")
    ap.add_argument("--floors", choices=["ai", "geometric", "rows"], default="ai")
    a = ap.parse_args()
    img = Image.open(a.photo).convert("RGB")
    if a.crop:
        img = img.crop(tuple(int(v) for v in a.crop.split(",")))
    name = os.path.splitext(os.path.basename(a.photo))[0]
    res = run_side(img, name, a.detector, a.thresh, floors=a.floors)
    os.makedirs(os.path.join(HERE, "output"), exist_ok=True)
    path = os.path.join(HERE, "output", f"{name}_{a.detector}_{a.floors}.jpg")
    draw(img, res).save(path, quality=90)
    print(f"{name} [{a.detector} / floors={a.floors}]")
    for fl in res.floors:
        print(f"  floor {fl.floor}: {fl.windows:3d} windows x ${fl.price_each:g} = ${fl.subtotal:g}"
              + ("   <- check this" if fl.check else ""))
    print(f"  TOTAL: {res.windows} windows, ${res.total:g}")
    if res.note:
        print(f"  NOTE: {res.note}")
    print(f"  -> {path}")


if __name__ == "__main__":
    sys.exit(main())
