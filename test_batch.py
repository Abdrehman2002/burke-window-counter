"""Run every photo in a folder through the pipeline and write marked-up results + a contact sheet.
Usage: python test_batch.py test-photos"""
import glob, os, sys, time
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "api"))
import _core as c
from PIL import Image, ImageDraw

c.load_env()
folder = sys.argv[1] if len(sys.argv) > 1 else "test-photos"
out = os.path.join(folder, "results"); os.makedirs(out, exist_ok=True)

def one(f):
    n = os.path.splitext(os.path.basename(f))[0]; img = Image.open(f).convert("RGB"); t = time.time()
    try:
        r = c.run_side(img, n)
        c.draw(img, r).save(os.path.join(out, n + ".jpg"), quality=85)
        fl = " / ".join(f"F{x.floor}:{x.windows}{'!' if x.check else ''}" for x in r.floors)
        return f"{n:28s} {r.windows:3d} win  ${r.total:>7,.0f}  {time.time()-t:4.1f}s  {fl}  {r.note}"
    except Exception as e:
        return f"{n:28s} FAILED {e}"

fs = sorted(f for f in glob.glob(os.path.join(folder, "*.jpg")) if not os.path.basename(f).startswith("_"))
with ThreadPoolExecutor(4) as ex:
    for line in ex.map(one, fs):
        print(line)
W, H = 480, 360
rs = sorted(glob.glob(os.path.join(out, "*.jpg")))
cols = 5
sheet = Image.new("RGB", (W * cols, H * ((len(rs) + cols - 1) // cols)), "white")
for i, f in enumerate(rs):
    im = Image.open(f); im.thumbnail((W - 6, H - 22)); x, y = (i % cols) * W, (i // cols) * H
    sheet.paste(im, (x + 3, y + 20)); ImageDraw.Draw(sheet).text((x + 4, y + 4), os.path.basename(f), fill="black")
sheet.save(os.path.join(folder, "_sheet.jpg"), quality=85)
