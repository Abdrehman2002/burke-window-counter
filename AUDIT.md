# Window Quote: audit (2026-09-26)

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
