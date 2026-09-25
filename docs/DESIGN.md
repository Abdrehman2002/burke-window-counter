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
| Snapshot / restore | One helper captures photos, quote, name and tab before any reset; Undo restores all of it | shipped 2026-09-26 |
| Price input | Holds the last valid value; inline error with `aria-invalid` and `role=alert` | shipped 2026-09-26 |
| Pointer highlight | White outline on the window under the pointer or finger; pointer cursor only over windows | shipped 2026-09-26 |

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

### Phase 3 (design-everyday-things), 2026-09-26
Probed in the browser: blank or odd prices, a sample over real photos, removing a photo after
counting, a near-miss tap, deselecting every building, changing settings after a count, and New
quote. **Score before: 4/10** (error recovery and evaluation failing). Heuristic column = Norman gulf
or error type.

| # | Issue | Heuristic (gulf / error type) | Severity (0-4) | Fix | Status |
|---|---|---|---|---|---|
| P3-1 | Tapping just beside a window unselected the whole building ($204 → $141 on one mis-tap) | Execution; action slip | 3 | A counted building is never dropped by a photo tap (a hint points to its chip); dimmed buildings can still be tapped to add. White outline on the window under the pointer or finger | **fixed** 2026-09-26 |
| P3-2 | Blank, zero or negative first-floor price silently made the quote $0 | Constraint; mistake | 3 | Price only changes to a valid amount; inline "Enter a price above $0. Using $3 until you do."; leaving the field restores the last good value | **fixed** 2026-09-26 |
| P3-3 | Removing a photo after counting left that side in the total | Evaluation (system state wrong) | 3 | The side leaves the quote with it; toast with Undo restores both | **fixed** 2026-09-26 |
| P3-4 | Picking a sample wiped the user's photos with no undo | Destructive action | 3 | Toast with Undo restores photos, quote and name | **fixed** 2026-09-26 |
| P3-5 | New quote wiped everything instantly with no undo | Destructive action | 3 | Toast with Undo restores photos, quote and name | **fixed** 2026-09-26 |
| P3-6 | Every building unselected → that side quietly $0 | Constraint; mistake | 2 | The last selected building can't be unselected; its chip says why | backlog |
| P3-7 | Multiplier below 1 quietly treated as 1 while the field shows the typed value | Evaluation | 2 | Inline "Minimum ×1"; the field corrects itself when you leave it | backlog |
| P3-8 | Changing sensitivity or floor grouping after a count has no effect and no signal | Mode error | 2 | "Settings changed: Count again to apply" | backlog |
| P3-9 | Merge and Add floor below undo only via whole-side Undo | Recovery | 2 | Each action's toast carries its own Undo | backlog |
| P3-11 | Print or Copy with no name gives "Untitled building" on the customer's quote | Memory lapse | 2 | Name field on the Summary tab; nudge when empty | backlog |
| P3-12 | Detection errors read technical ("window detector unavailable (…)") | Error message (what/why/how) | 2 | "Couldn't reach the window finder. Check your connection and press Count again. Your photos are kept." | backlog (Phase 6) |
| P3-13 | − at 0 does nothing, silently | Constraint | 1 | Disable − at 0 | backlog |

**Rule set:** use a guard-rail where the mistake can be made impossible (P3-1, 2, 6, 7, 13); use Undo
instead of an "are you sure?" dialog wherever work is thrown away (P3-3, 4, 5, 9).
**Score after: 7/10.** Recovery and constraints fixed for the severity-3 paths; stale settings (P3-8)
and the smaller guard-rails remain.

## Microinteraction Inventory
| Interaction | Trigger/Rules/Feedback/Loops | Fix | Status |
|---|---|---|---|
