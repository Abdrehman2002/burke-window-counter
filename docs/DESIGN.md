# Design System

## Design Direction
A field tool for a window-cleaning owner quoting on the sidewalk. The photo is the proof: every number
on screen must be traceable to a box on the building. Signature moment (to confirm in Phase 5): tapping
a floor and seeing exactly which windows it charges for.

## Typography
Bebas Neue for display (building name, totals, floor headings); Poppins for body and controls.
Tabular numerals on all money and counts.

## Tokens
Light: ground `#F8F9FC`, panel `#FFFFFF`, ink `#08142D`, accent `#154BFF`. Dark: ground `#07101F`,
accent `#3A7BFF`. Warning amber `#A35F00` / `#FFB938`. Floor colours: 8 fixed hues, one per floor.
(Spacing scale and a full palette audit: Phase 4.)

## Components
| Component | Decision | Status |
|---|---|---|
| Photo viewer | Frames the chosen buildings (zoomed, drawn at high resolution); "Show whole photo" toggle | shipped 2026-09-26 |
| Setup panel on phones | Folds into one "Edit photos & prices" bar once a real quote exists | shipped 2026-09-26 |
| Toast | Can carry one action button (Undo), stays 9 s when it does | shipped 2026-09-26 |

## UX Audit Findings
Phase 2 (ux-heuristics), 2026-09-26. Heuristic evaluation of the demo flow (add photo → Count → pick
building → check → fix → summary), desktop and phone. **Score before: 6/10.**

| # | Issue | Heuristic | Severity (0-4) | Fix | Status |
|---|---|---|---|---|---|
| 1 | Windows too small to check on wide photos (a building can fill a fifth of the frame), and no zoom | Visibility of status / match to the real task | 3 | Frame the photo on the chosen buildings at high resolution; "Show whole photo" toggle; clicks, keyboard and draw-area go through the framing | **fixed** 2026-09-26 (motel: view 742×265 of 1600×1200) |
| 2 | On a phone the result is ~1,060 px below the Count button | Visibility of status | 3 | Scroll to the result when counting finishes; fold the setup into one bar | **fixed** 2026-09-26 (result now at the top) |
| 3 | "Count" again silently discards all manual fixes | User control; error prevention | 3 | Button becomes "Count again"; warning line before; toast with Undo restores the earlier quote | **fixed** 2026-09-26 |
| 4 | Tap targets 19–32 px on phones (chips, merge, add-floor links, +/−, remove photo) | Error prevention (slips) | 3 | 44 px minimum on touch screens | **fixed** 2026-09-26 (0 targets under 44 px) |
| 5 | "N things to check" doesn't say what or where; the pill isn't actionable | Recognition over recall | 2 | Make it a button that lists each item and jumps to it | backlog |
| 6 | No link between a floor row and its windows on the photo | Recognition over recall | 2 | Touching or hovering a floor row highlights only that floor's windows | backlog (candidate signature moment, Phase 5) |
| 7 | Building badges overlap on the photo; chips read "1 · Building 1 · 10" | Aesthetic and minimalist | 2 | Nudge overlapping badges apart; chips read "Building 1 · 10 windows" | backlog |
| 8 | Example banner says "on the left", but photos are above on phones | Match to the real world | 1 | "Add your own photos and press Count" | backlog (Phase 6) |
| 9 | Tab dot's meaning is hover-only; "counted in 9.2s" is noise | Works without hover; minimalist | 1 | Say "check" in text; drop the timing | backlog |
| 10 | Wording: "merge", "boxed", "sensitivity", "buildings or walls" | Match to the real world | 2 | Plain-language rewrites | Phase 6 |

**Trunk Test:**
- Start screen: **pass**. The app name, the example label and the one primary button are all clear.
- Result screen: **partial**. Tabs show the side, but "what do I check next?" isn't answered (#5, #6).
- Summary: **pass**.

**Score after the fixes: 8/10.** To reach 10: #5 and #6 (what to check, and where), plus the Phase 6
wording.

## Microinteraction Inventory
| Interaction | Trigger/Rules/Feedback/Loops | Fix | Status |
|---|---|---|---|
