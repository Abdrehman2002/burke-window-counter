# Window Quote from Photos

Upload a photo of each side of a building. It boxes every window, groups them into floors, prices each
floor, and adds up each side and the whole building.

Default pricing: **floor n = $3 × 2^(n−1)** per window ($3, $6, $12, $24 …). Both numbers can be changed in the sidebar.

## Run it

```bash
pip install -r requirements.txt
cp .env.example .env        # then fill in the keys
streamlit run app.py
```

Command line, one photo:

```bash
python counter.py photos/06_apt_karlsruhe.jpg
```

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
