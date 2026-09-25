# Window Quote: audit (2026-09-26)

## Status: every code-review item is fixed (2026-09-26)

Each fix was tested: backend edge cases called directly, UI flows in a real browser (22 checks, all
passing), accuracy re-scored with `eval.py` (89%, unchanged).

| Area | Fixed |
|---|---|
| Security / cost | Access code (`ACCESS_CODE` env var; the page asks once and remembers it). Best-effort rate limit. Sensitivity clamped 0.2–0.8. AI storey check only for the 3 biggest buildings. At most 400 windows. Area mode validated and honours "geometry only". Images over 40 MP refused and shrunk before rotating. The key-listing GET removed. `esc()` escapes apostrophes. The dev server binds to localhost and never serves `.env` or source. `.env` is kept out of Vercel uploads. |
| Reliability | One 50 s budget per request. Outside calls get connect 5 s / read ≤20 s (Roboflow) and ≤40 s (OpenAI) timeouts inside it, Roboflow gets 2 tries, and Gemini is skipped late in the budget. A failed tile is noted and the side flagged. `VOTES` is at least 1. |
| Correctness | Every window has a floor, including in fallback paths, so a click-in is always priced. An empty drawn area never re-runs detection. Floor edits touch only the selected buildings. Added floors survive selecting buildings. Draw mode can't get stuck, Escape cancels it, and the drag's ending click is ignored. One bad AI answer no longer discards the other votes. The fallback note says why. The first tab is Front. Undo keeps a drawn area. CSV rounded to cents. |
| Performance | `_merge_stacked` is a single sweep. Building-mask distances are computed once. |
| Maintainability | About 280 lines of dead floor code deleted. The broken Streamlit app removed. README and API docstring rewritten. |
| Accessibility | Workspace no longer an `aria-live` region; toast and a screen-reader line are. Arrow keys move between windows on the photo and Enter opens the menu. Tab stays in the menu and Escape returns focus. Focus kept after +/−, merge and tab changes. Proper tab pattern with arrow keys. Ctrl+P prints the quote. Legend matches the drawing. |

**Still open** (not code-review items): the product and accuracy work in P1/P2 below. Drawing an area
with the keyboard alone isn't possible, but the building chips cover the same need.

**To do on Vercel:** set **`ACCESS_CODE`** in Project → Settings → Environment Variables, then redeploy.
Until then the API stays open.

Where it stands, what's wrong, and what to do next, ranked. Every number comes from a measured run:
`eval.py` (hand counts on 7 buildings), `test_batch.py` (25 photos), live calls to Vercel, and an
independent read of the code.

## Where it stands

| | Now |
|---|---|
| Windows on the right floor (hand-count benchmark) | **89–91%** (was 57% at the start of the accuracy work) |
| Floor count right | 12 of 14 runs; same answer run to run |
| Time per photo (live, Vercel) | **~7 s**; sides run in parallel |
| Cost per photo | 6 Roboflow images (~$0.003 each) + a few small OpenAI calls: **under 3 cents** |
| Free tier | Roboflow free plan ≈ 30,000 images/month ≈ **5,000 photo sides/month** |
| Buildings per photo | Separate buildings split and numbered; the user taps what to count |
| Corner shots | Split into walls on some photos (brick corner building); missed on others (office tower). "Draw the area to count" covers the misses |

## P0: fix before Burke or anyone else gets the link

1. **The API is public and spends your credits.** Anyone who finds `/api/count` can run unlimited
   Roboflow and OpenAI calls on your keys. It also runs the AI for every building in the photo, even
   ones nobody picked. **Fix:** an access code in the page (a shared secret checked by the API), a
   per-IP rate limit, the sensitivity setting clamped to 0.2–0.8, at most 4 regions, and the AI run
   only for the picked building. *About 1 hour.*
2. **A slow run can blow past Vercel's 60 s limit.** Worst case with retries is ~500 s. Vercel cuts
   it off, the user sees a generic error, and the calls still bill. **Fix:** one 50 s budget for the
   whole request, shorter timeouts (connect 5 s, read 20 s), 2 tries, skip the Gemini fallback late
   in the budget, and return what's done with a note. *About 1 hour.*
3. **Oversized images can exhaust memory.** A small PNG can expand to hundreds of MB before it is
   shrunk. **Fix:** check the pixel size first and shrink before rotating. *15 minutes.*
4. **Vercel's free plan is non-commercial.** Fine for the demo. If Burke pays, move to Vercel Pro
   ($20/mo) or Railway (~$5/mo).
5. ✅ **Done today:** the local dev server was serving `.env` (your API keys) to your whole network.
   It now binds to localhost and serves only the page and photos. `.env` is excluded from Vercel uploads.

## Accuracy experiments (2026-09-26)

The benchmark grew from 7 to 10 hand-counted buildings (colonial with French doors, Baltimore rowhouses,
country church). Counting rule: every window or glass door you'd clean counts; a solid door with a small
window doesn't. Baseline on the harder set: **83% of windows on the right floor, floor count right 8/10,
identical across runs.**

| Change tried | Result | Kept? |
|---|---|---|
| Add "door" to the SAM 3 request to drop door lights | 70%. The extra prompt changes what SAM 3 returns for "window" and created fake floors | No (off by default) |
| Add "glass door" / "french door" prompts | 82% | No |
| Add "glass" prompt | 59% | No |
| Low threshold (0.2) + GPT close-up verification of doubtful candidates | 73%. Found Wien's missing balcony floor but let in panes and porch lattice | No (off by default, kept as a switch) |
| Same, only for candidates not overlapping a confident window (0.2 / 0.3) | 69% / 69% | No |
| "Normal window size" measured from confident boxes only | Stabilises clean-up; baseline unchanged at 83% | Yes |

**Conclusion:** zero-shot SAM 3 plus prompt and threshold tuning has reached its ceiling here, about
83–89% depending on the buildings. The misses left (French and glass doors, glass behind balconies,
boarded openings, side walls) are "what counts as a window" judgments. A general detector can't learn
those from prompts. **A step change needs a detector trained on labelled facades under Burke's own
counting rules** (see #4 below).

## Commercial test (2026-09-26): 18 US commercial buildings

Run through the full pipeline; results in `commercial-photos/results/` and `_results.txt`. **It's not
ready for commercial quotes.** Roughly 5 of 18 are usable as-is (Cheese Store upper floor, medical
office, Harmony Hills school, Seneca 506 upper floor, Fredrich upper floor).

| # | Failure | Where | Cause (measured) | Fix |
|---|---|---|---|---|
| C1 | **Shop-window glass not counted** | c01a, c05b, c06a, c06b, c09a, c09b, c12 | SAM 3 scores storefront glass 0.41–0.54, so it lands under the 0.5 "counted" line, or it's 8× a typical window and the size filter drops it (15 boxes on the Cheese Store) | Treat storefront glass as its own type: keep large ground-floor glass (no 8× drop), price it per Burke's rule (pane, or per sq ft) |
| C2 | **Floors wrong on corner shots and tall buildings** | c07b hotel (50 windows on "floor 3" zig-zagging), c11 Doubletree (3 floors, should be ~16), c01b, c12, c09a | Two walls in one photo (the corner split didn't fire), so row-matching pairs windows across the corner; on the tower, steep perspective plus 389 overlapping boxes | Stronger corner split (the building outline's corner line); floors per wall; with 12+ floors, count floors by stepping up one column |
| C3 | **Glass curtain walls: 0 windows** | c03b (0), c03a (few), c08b | Reflective glass isn't "window" at 0.4 (57 candidates exist at 0.2); a curtain wall is panels, not windows | Curtain-wall mode: count panels (mullion grid) or price by area, per Burke |
| C4 | **Building too small in the frame: 0 windows** | c04a strip mall (far shot) | Zero detections even at 0.2; the storefront is a thin strip | "Stand closer" guidance, or auto-zoom on the building outline and detect again |
| C5 | **Divided-lite and transom panes counted as separate windows** | c05b transom row counted as 20 | Each pane detected | Depends on Burke's pane-vs-window answer; merge panes in one frame if he says "per window" |
| C6 | **Default building pick is sometimes wrong** | c06a picked a 4-window strip, not the main building | Default = most windows | Default = the building nearest the photo centre and largest in area |

### Fixes for C2, C3, C4, C5 (2026-09-26)

House benchmark unchanged at 83% after all of them.

| # | Fix | Before → after |
|---|---|---|
| C2 corners | A corner is the building outline's top point, confirmed by row slopes flipping sign across it (>16°). Floors are computed per wall | Otis hotel: 50 windows zig-zagging on "floor 3" → **wall B 13·13·13·13, wall A 8·7·7·9** (correct) |
| C2 towers | Count floors up each column (one window per floor per column), used only when row matching finds under half the floors | Doubletree: 3 floors → **16 levels of ~8–13** (tower ≈ 15 storeys + podium) |
| C3 glass walls | A separate SAM 3 "glass curtain wall" outline, accepted only when ≥4× a window's height and ≥38% of its building (shop fronts measured 1.2–3.2×). Panels counted from the frame grid, strip by strip. Each glass wall is its own item with its own rows, always flagged "check" | Rockville glass offices: 0 → **3 glass walls, e.g. 5 rows × 10 panels**. Hallidie: grid of ~20 per row. Shop fronts untouched |
| C5 panes | A wide, short band (≥3.5:1) whose lights cover ≥80% of its width is one divided window (transom/ribbon). `COUNT_PANES=1` counts every pane | Cheese Store transoms: 20 → **1 per band** (floor 1: 3); house triples unchanged |
| C4 far shots | Zoom-and-detect-again on small buildings. **Measured no gain** (the tiled pass already sees them), so off by default (`ZOOM_BELOW`). The strip mall's real problem is shop glass (C1) | Strip mall still 0 |

**C6 fixed:** the default is the building that takes up the most of the photo, weighted towards the middle,
together with any glass walls on it (window count only breaks ties). Seneca 426 now picks the main building;
Hallidie picks its glass facade; the medical office picks its glass wall and brick part together.

**Still open:** C1 shop-front glass (low detector confidence; needs Burke's pricing rule). Glass-wall panel *rows* are not always storeys (glass can have 2 rows per floor), so
glass walls are always flagged for a check.

## P1: accuracy (the product is only as good as this)

Ranked by how much they move the count per hour of work:

1. **Get Burke's answers. They change counts and prices more than any code does.**
   - One window or each pane? The detector counts frames; a 4-pane window may be 1 or 4 to him.
   - Does the price really double on every floor? Floor 11 at $3,072 per window made the office tower $49,000.
   - Are shopfronts, French doors and glass doors priced as windows?
   - Is a basement or garden level "floor 1"?
2. **Test on real photos from Burke.** The benchmark is 7 buildings from Wikimedia, mostly not
   American houses. Ask him for 10–20 photos of real jobs with his own counts. That becomes the
   benchmark that actually matters. *Zero code.*
3. **Detection misses:**
   - Glass behind balcony railings (the Wien house loses a whole floor).
   - French doors and glass doors (the colonial's ground floor).
   - Windows in doors counted as windows.

   **Fix:** extra SAM 3 prompts ("glass door", "balcony door", "storefront window") as their own
   types, priced by Burke's answers. *2–3 hours, measured with `eval.py`.*
4. **Fine-tune a detector on labelled facades (the biggest long-term gain).** 150–300 photos labelled
   in Roboflow, then train RF-DETR there. That gives a model that knows *his* definition of a window,
   runs faster, and costs less per photo. *1–2 days of labelling; do it once there's paying demand.*
5. **Sides of one building should agree.** Front and back usually have the same number of floors;
   flag a side that disagrees with the others. *1 hour.*
6. **Corner split misses some towers.** The draw-area tool covers it for now. A better split would
   use the building outline's corner line (where the edge bends) as well as the window rows. *Half a day.*
7. **The "Check the floors" flag still fires on some correct quotes** (the model sees a basement or a
   roof level). Tune it against the bigger benchmark once #2 exists.

## P1: correctness bugs found in the code review

| # | Bug | Effect | Fix |
|---|---|---|---|
| 1 | A window clicked back in (edge or low confidence) can have no floor in the fallback paths | It's counted but on no floor, so the price is wrong | Give it the nearest floor, as the main path already does |
| 2 | "Add a floor below" and "merge" shift windows in buildings that aren't selected | Selecting another building afterwards misprices it by a floor | Only shift the selected buildings |
| 3 | Selecting or deselecting a building drops floors added with "Add a floor on top" | Those floors quietly vanish from the price | Keep manually added floors |
| 4 | Drawing an area on a photo with no windows re-runs full detection | Bills again; the message is wrong | Treat an empty list as a list |
| 5 | Draw mode can get stuck if the page re-renders mid-drag; finishing a drag can also toggle a building | Dead clicks | Clear draw mode on re-render; ignore the click that ends a drag |
| 6 | One malformed AI answer throws away all three votes | Falls back to geometry for no reason | Validate each answer separately |
| 7 | A non-number sensitivity crashes the API | Generic 500 | Parse inside the error handler; return 400 |
| 8 | A tile that fails is skipped silently | Recall drops with no warning | Add a note on the side |
| 9 | The first tab is whichever side finished first; CSV prices aren't rounded; the legend describes the wrong "not counted" style; Ctrl+P prints a blank sheet | Polish | Small fixes |

*All of it: about 2 hours.*

## P2: product (what makes Burke use it every day)

1. **Built for the phone, on site.** Take the photo straight from the camera
   (`capture="environment"`), one side at a time with a "next side" prompt, big touch targets.
   Burke quotes standing in front of the building.
2. **Save quotes.** Nothing is stored now; closing the tab loses the work. Add a quote list with
   name, address, date, total and photos. It needs a small database (Supabase free tier).
3. **A customer-ready quote.** Branded PDF with his logo, the photo with floors marked, per-floor
   lines and the total, sent by email or text from the app.
4. **His real pricing rules**, beyond base × 2 per floor: a minimum charge, inside/outside, screens,
   hard-to-reach or large windows, travel. Collect these on the call.
5. **The accuracy story on screen:** "Checked by you: yes/no" per side before a quote can be sent,
   so a flagged floor can't go out unchecked.

## P3: engineering hygiene

- **Delete about 330 lines of dead code:** the old floor methods `assign_floors`, `build_rows`,
  `lattice_rows`, `assign_floors_llm`, `ROWS_PROMPT` and `_extend_wall`.
- **Fix or retire the old Streamlit `app.py`.** It's broken against the current data format.
- **Update the README and the `count.py` docstring.** They still describe the old app.
- **Run `eval.py` on every change**, as a GitHub Action with cached detections, so accuracy can't
  quietly regress (it did twice during this work).
- **Speed up two slow spots:** `_merge_stacked` and the per-box mask distance in `find_regions`.
  Cap the number of windows at ~400.
- **Accessibility:**
  - The workspace is an `aria-live` region that re-reads on every keystroke.
  - Canvas actions need a keyboard route (a list of windows per floor).
  - The tabs need proper tab roles.
  - Focus should return after menus.

## Suggested order

1. **P0 items 1–3 and the P1 bug table.** About half a day; the tool is then safe to send.
2. **Record the demo video** on straight-on houses and rowhouses. Show one building pick, one flagged
   floor and one fix. Send it with Burke's four questions (P1 accuracy #1).
3. **When he answers:** set the pane/door/basement rules, then run his own photos through `eval.py`.
4. **If he wants to pay:** P2 items 1–3, then the fine-tuned detector.
