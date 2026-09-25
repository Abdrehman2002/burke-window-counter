# Window Quote from Photos

Upload a photo of each side of a building. It boxes every window, groups them into floors, prices each
floor, and adds up each side and the whole building.

Default pricing: **floor n = $3 × 2^(n−1)** per window ($3, $6, $12, $24 …). Both numbers can be changed in the sidebar.

## Deploy on Vercel

1. In Vercel: **Add New → Project → Import** `burke-window-counter` from GitHub. Leave Framework Preset as **Other**, with no build command.
2. **Settings → Environment Variables:** add `ROBOFLOW_API_KEY` and `OPENAI_API_KEY`, and optionally `GEMINI_API_KEY`.
3. Deploy. Every push to `main` redeploys automatically.

Check the keys are loaded: `GET /api/count` returns `{"keys": {...: true}}`.

| Part | File |
|---|---|
| The dashboard (static page, pricing and editing run in the browser) | `index.html` |
| Counting API: one call per side, sides run in parallel | `api/count.py` → `api/_core.py` |
| Function limits | `vercel.json` (60s max per side; a side usually takes 5 to 12s) |

Photos are shrunk to 1600px in the browser before upload, which keeps them under Vercel's 4.5MB request limit.

## Run it locally

```bash
pip install -r requirements.txt
cp .env.example .env        # then fill in the keys
python dev_server.py 3000   # same page and API as Vercel, at http://localhost:3000
```

Command line, one photo: `python counter.py photos/06_apt_karlsruhe.jpg`
There's also an older Streamlit version: `pip install -r requirements-local.txt && streamlit run app.py`

## Using the dashboard

- Add a photo for each side (Front is enough), or load a sample building.
- **Click a box on the photo to remove it** if it isn't a window. Click it again to bring it back.
- Use **+ / −** on a floor to fix its count, and **Add a floor it missed** if a whole floor was skipped.
- Amber boxes and **check** tags mark windows the detector was unsure about.
- The **Summary** tab has the floors-by-sides table, print / save as PDF, copy as text, and CSV.

## How it works

| Step | What | Tool |
|---|---|---|
| 1. Detect | One box per window. The count is the number of boxes. | **SAM 3** (Meta), hosted on Roboflow, text prompt `window` |
| 1b. Filter | Windows sitting on a car are dropped (`car`/`vehicle` go in the same call) | SAM 3 |
| 1c. Clean | Panes inside a window, duplicates, and a tall window split into two boxes are merged | Python |
| 2. Floors | Each box is numbered and a vision model says which storey it's on. Boxes on neighbouring buildings get storey 0 and are dropped. | OpenAI `gpt-5.4-mini` (Gemini as fallback) |
| 2b. Guard | Windows whose heights mostly overlap are forced onto the same floor, whatever the model said | Python |
| 3. Price | windows × base × multiplier^(floor−1), per floor, per side, total | Python |
| 4. Flag | A floor with a low-confidence box is marked **check** and the box is drawn in amber | Python |

Why not just ask an AI to count: chat models see windows fine but lose count across a grid of
identical ones. A detector gives one visible box per window, so the count can be checked. The
language model is only used to *sort* boxes into floors, which is a much easier job.

If every AI floor provider fails, it falls back to pure geometry, so a quote never fails completely.

## Limits

- Straight-on photos work best. Steep angles and windows behind balconies get missed. Those floors are flagged; check them.
- All-glass towers (no separate windows) are out of scope.
- Counting from photos is never 100%. The **Windows** column in the dashboard can be edited, and the price updates.

## Keys

`ROBOFLOW_API_KEY` (detector), `OPENAI_API_KEY` (floors), optional `GEMINI_API_KEY` (fallback).
Keep them in `.env`, which is git-ignored. On Streamlit Cloud, use Secrets.

Test photos: Wikimedia Commons, credits in `photos/CREDITS.md`.
