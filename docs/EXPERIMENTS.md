# Experiments

## Experiment Cards
### EXP-001: Framed photo makes checking the count faster
- Hypothesis: We believe the owner confirms a quote faster, with fewer missed errors, if the photo is framed on
  the chosen building, because the windows are big enough to check without zooming.
- Type: A-B on the demo recording, then Burke's first 5 real quotes
- Primary metric & threshold (pre-committed): median time from "Done" to Summary ≤ 60 s on a 2-3 storey house
- Guardrail metric: wrong-floor or missed windows found on review do not rise (eval.py stays ≥ 89%)
- Decision rule: persevere if the time is met and the guardrail holds; iterate (tap-to-zoom) if not
- Result & verdict: pending (needs Burke)

### EXP-002: Phone result lands at the top
- Hypothesis: We believe more on-site quotes get finished if, on a phone, the result appears at the top after
  Count, because nobody scrolls 1,000 px on a sidewalk.
- Type: Instrumented use with Burke
- Primary metric & threshold (pre-committed): ≥ 80% of phone counts reach the Summary tab
- Guardrail metric: the "Edit photos & prices" bar is found when needed (no "where are my photos?" questions)
- Decision rule: persevere at ≥ 80%; iterate the fold bar copy if he can't find the setup
- Result & verdict: pending

## Experiment Backlog
| Idea | ICE (impact/confidence/ease) | Status |
|---|---|---|
| Frame the photo on the chosen buildings (UX #1) | 9/8/6 | shipped 2026-09-26 (EXP-001) |
| Land on the result on phones, fold the setup (UX #2) | 8/8/8 | shipped 2026-09-26 (EXP-002) |
| Undo a recount instead of silently wiping fixes (UX #3) | 7/9/8 | shipped 2026-09-26 (metric: 0 reports of lost fixes) |
| 44 px touch targets (UX #4) | 6/9/9 | shipped 2026-09-26 (metric: mis-taps on chips/merge in the demo recording = 0) |
| "Things to check" becomes a clickable list (UX #5) | 7/7/7 | backlog |
| Floor row highlights its windows on the photo (UX #6) | 8/7/6 | backlog, candidate for Phase 5 |
| De-overlap badges, simpler chip labels (UX #7) | 4/8/7 | backlog |
| Detect glass doors, balcony glass, storefronts as their own types (Phase 1 accuracy decision) | 9/6/5 | backlog (code) |
| A near-miss tap never drops a building; highlight the window under the pointer (P3-1) | 8/9/7 | shipped 2026-09-26 (metric: 0 accidental building drops in the demo recording and Burke's first 10 quotes) |
| Price keeps its last valid value; inline error (P3-2) | 8/9/9 | shipped 2026-09-26 (metric: 0 quotes sent at $0 or with a blank price) |
| Removing a photo removes its side, with Undo (P3-3) | 7/9/8 | shipped 2026-09-26 (metric: quote total always equals the sum of sides that have photos) |
| Undo for sample loading and New quote (P3-4, P3-5) | 7/9/8 | shipped 2026-09-26 (metric: 0 reports of lost photos or quotes) |
| Last building can't be unselected (P3-6); multiplier minimum shown (P3-7); − disabled at 0 (P3-13) | 5/8/9 | backlog |
| "Settings changed: Count again to apply" (P3-8) | 6/8/8 | backlog |
| Per-action Undo on merge and Add floor below (P3-9) | 6/8/7 | backlog |
| Name on the Summary tab with a nudge before print or copy (P3-11) | 6/8/8 | backlog |
