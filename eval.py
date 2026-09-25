"""Score the pipeline against hand counts. Usage: python eval.py [--runs 2]

Ground truth = windows per floor, bottom floor first, main wall only, counted by hand from the photo.
Each separate window frame is one window (a triple window is 3). SAM 3 detections are cached per photo
in test-photos/.cache/ so re-scoring after a change to the floor logic costs only the model calls.
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "api"))
import _core as c  # noqa: E402
from PIL import Image  # noqa: E402

TRUTH = {
    "06_apt_karlsruhe": [7, 8, 5],
    "03_2storey_buffalo": [5, 9, 6],
    "03b_rowhouses_park": [6, 9, 9, 9],
    "02b_colonial": [6, 10],
    "10_irregular_knox": [2, 7, 8, 8],
    "11_irregular_wien": [3, 4, 3, 3],
    "01_1storey_maracaibo": [2],
    "02a_colonial": [8, 6],
    "03a_rowhouses_fulton": ([4, 4, 11, 11], "all"),
    "07b_church_mtgregory": [4],
}

CACHE = os.path.join(HERE, "test-photos", ".cache")
os.makedirs(CACHE, exist_ok=True)
_real_detect = c.detect_sam3


def cached_detect(img, thresh=0.4, prompt="window"):
    tag = "-".join(p.replace(" ", "") for p in c.GLASS_PROMPTS) + ("_door" if c.DOOR_RULE else "")
    key = f"{cached_detect.name}_g{c.TILE_GRID}_t{thresh}_{tag}.json"
    path = os.path.join(CACHE, key)
    if os.path.exists(path):
        return [c.Box(*b[:5], kind=b[5] if len(b) > 5 else "window") for b in json.load(open(path))]
    boxes = _real_detect(img, thresh, prompt)
    json.dump([[b.x0, b.y0, b.x1, b.y1, b.conf, b.kind] for b in boxes], open(path, "w"))
    return boxes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=1)
    a = ap.parse_args()
    c.load_env()
    c.detect_sam3 = cached_detect
    tot_floor_ok = tot_abs = tot_true = n = 0
    for name, truth in TRUTH.items():
        select = None
        if isinstance(truth, tuple):
            truth, select = truth
        img = Image.open(os.path.join(HERE, "test-photos", name + ".jpg")).convert("RGB")
        for r in range(a.runs):
            cached_detect.name = name
            res = c.run_side(img, name)
            if select == "all":  # the viewer taps every building that has windows
                res.selected = [r["id"] for r in res.regions if r["windows"] > 0]
                res.floors = c.price(res.boxes, 0, selected=set(res.selected))
            got = [f.windows for f in res.floors]
            k = max(len(got), len(truth))
            g = got + [0] * (k - len(got))
            t = truth + [0] * (k - len(truth))
            err = sum(abs(x - y) for x, y in zip(g, t))
            ok = len(got) == len(truth)
            tot_floor_ok += ok
            tot_abs += err
            tot_true += sum(truth)
            n += 1
            flag = " FLAGGED" if res.floors_check else ""
            print(f"{name:24s} truth {truth!s:16s} got {got!s:24s} floors {'OK ' if ok else 'BAD'}  "
                  f"per-floor error {err:2d}{flag}")
    print(f"\nfloor count right: {tot_floor_ok}/{n}   per-floor error: {tot_abs} of {tot_true} windows "
          f"({100 * (1 - tot_abs / tot_true):.0f}% placed right)")


if __name__ == "__main__":
    main()
