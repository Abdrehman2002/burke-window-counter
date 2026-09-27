# Window Quote from Photos

Photograph each side of a building. It finds every window, sorts them into floors, prices each floor
(floor n = $3 × 2^(n−1) per window by default, both numbers editable), and adds up each side and the
building. Built for window-cleaning quotes.

## Using it

- **Add a photo per side** (Front is enough), or load a sample building, then press **Count**.
- **Several buildings in one photo?** They're found, numbered and dimmed. Tap the ones you're quoting,
  on the photo or with the chips above it. Each building gets its own floors.
- **Corner shot or odd framing?** **Draw the area to count**: drag a box over the wall. Floors are
  worked out again for just those windows, in under a second, with no new detection.
- **Fix anything:**
  - Click a window to move it to another floor or stop counting it.
  - Use **+ / −** to change a floor's count.
  - **merge ↓** joins a floor to the one below.
  - **Add a floor on top / below** adds a missed floor.
  - **Undo** puts the side back to what was detected.
- **Keyboard:** Tab to the photo, then arrow keys to move between windows and Enter to change one.
  Arrow keys also switch tabs.
- **Amber boxes** are windows the detector was unsure about. Dashed boxes are not counted (cut off by
  the photo edge, too unsure, on another building, or removed by you). One click counts one.
- **Summary** has the floors × sides table, print or save as PDF, copy as text, and CSV.

## How it works

| Step | What | Where |
|---|---|---|
| Windows | SAM 3 (hosted on Roboflow) on the whole photo **and** 4 overlapping tiles in parallel, so small windows on big buildings aren't lost. Windows on cars are dropped. | `detect_sam3` |
| Clean-up | Merge panes into their window, split halves into one box, and near-duplicates. A box around two windows is dropped in favour of the two. | `clean` |
| Buildings | SAM 3 "building" outlines split the photo into buildings. A corner shot's two walls are split where the rows stop running toward one vanishing point. | `segment_buildings`, `find_regions` |
| Floors | Per building: each window is slid along its floor line toward the vanishing point, and windows that land together form a row. Tall regular facades trust this geometry. On houses, `gpt-5.4-mini` (3 votes, median) says which rows share a storey. Stray 1–2 window "floors" are merged. | `vp_rows`, `floors_for_wall` |
| Price | Windows × base × multiplier^(floor−1), in the browser, so price changes are instant. | `index.html` |

The AI never adds or deletes a window; it only labels rows. If it's unavailable, floors come from
geometry and the page says so.

## Run it locally

```bash
pip install -r requirements.txt
cp .env.example .env          # then fill in the keys
python dev_server.py 3000     # same page and API as Vercel; only reachable from this computer
```

Other tools:
- `python counter.py photo.jpg` counts one photo from the command line.
- `python test_batch.py test-photos` runs a folder and writes a contact sheet.
- `python eval.py --runs 2` scores against hand counts. Run it after every change to the counting logic.
- `python glass_bench.py` scores the glass-wall pane grid on synthetic walls with exact grids (no network). Run it after any change to `glass_grid`.

## Deploy (Vercel)

1. Import the repo in Vercel. `vercel.json` sets the "Other" preset and a 60 s limit.
2. Add the environment variables `ROBOFLOW_API_KEY`, `OPENAI_API_KEY`, optionally `GEMINI_API_KEY`
   (the fallback), and **`ACCESS_CODE`**.
3. Every push to `main` redeploys.

**Set `ACCESS_CODE`.** Without it, anyone who finds `/api/count` can run it on your keys. With it, the
page asks for the code once and remembers it on that device.

## Limits and safeguards

- **Budget:** 50 s per request (Vercel stops at 60 s). Outside calls get short timeouts inside that
  budget, and a failed tile is skipped with a note rather than failing the side.
- **Input limits:** photos over 40 MP are refused, photos are shrunk to 1600 px, and at most 400
  windows are kept per photo.
- **AI calls:** only the 3 biggest buildings in a photo get the AI storey check; the rest use geometry.
- **Rate limit:** a best-effort 40 requests per 10 minutes per address, per warm instance. The access
  code is the real gate.
- **Cost:** under 3 cents per photo side (6 Roboflow images plus a few small OpenAI calls). The
  Roboflow free plan covers roughly 5,000 sides a month.

**Known misses:**
- Windows behind balcony railings, French doors and glass doors aren't detected.
- A window in a door is counted.
- Some corner shots aren't split into walls; use Draw the area.

See `AUDIT.md`.

Test photos: Wikimedia Commons, with credits in `photos/CREDITS.md`.
