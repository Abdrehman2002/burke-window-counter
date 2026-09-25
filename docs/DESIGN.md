# Design System

## Design Direction
A field tool for a window-cleaning owner quoting on the sidewalk. The photo is the proof: every number
on screen must be traceable to a box on the building. Signature moment (to confirm in Phase 5): tapping
a floor and seeing exactly which windows it charges for.

## Typography
Bebas Neue for display (building name, totals, floor headings); Poppins for body and controls.
Tabular numerals on all money and counts.

## Tokens
- **Spacing scale (Phase 4):** 4 · 8 · 12 · 16 · 24 · 32 · 48 px for every padding, gap and margin.
  1–3 px hairline offsets on badges are allowed.
- **Type scale (Phase 4):** 12 · 13 · 15 · 18 · 24 · 36 · 52 · 64 px. Display (Bebas Neue) at 24 and up;
  Poppins below.
- **Light palette:** ground `#F8F9FC`, panel `#FFFFFF`, sunk `#F1F4FA`, ink `#08142D`, muted `#5A6480`
  (5.9:1), **faint `#626D88` (4.7:1; was `#8A93AA` at 2.9:1, failing)**, line `#E2E7F1`, accent `#154BFF`
  (6.0:1), warn `#A35F00` (5.0:1).
- **Dark palette:** ground `#07101F`, panel `#0D1A33`, ink `#F3F5FA`, muted `#A4AEC6`, **faint `#7E88A1`
  (4.9:1)**, accent `#3A7BFF`, warn `#FFB938`.
- **Colour roles:** ink carries the one number that matters (the quote total); blue is for things you
  can tap; amber is only "check this"; the 8 floor hues are data, one per floor.
- **Shadows:** panels `0 1px 2px + 0 8px 24px` (raised); the window menu `0 10px 30px` (floating).

## Components
| Component | Decision | Status |
|---|---|---|
| Photo viewer | Frames the chosen buildings (zoomed, drawn at high resolution); "Show whole photo" toggle | shipped 2026-09-26 |
| Setup panel on phones | Folds into one "Edit photos & prices" bar once a real quote exists | shipped 2026-09-26 |
| Toast | Can carry one action button (Undo), stays 9 s when it does | shipped 2026-09-26 |
| Snapshot / restore | One helper captures photos, quote, name and tab before any reset; Undo restores all of it | shipped 2026-09-26 |
| Price input | Holds the last valid value; inline error with `aria-invalid` and `role=alert` | shipped 2026-09-26 |
| Pointer highlight | White outline on the window under the pointer or finger; pointer cursor only over windows | shipped 2026-09-26 |
| Quote total | Ink, 64 px display: the single "all three levers" element on the screen. Building name drops to 24 px | shipped 2026-09-26 |
| Count button | Filled only when there is something new to count; outlined once the quote matches the photos | shipped 2026-09-26 |
| Floor row | Window count 18 px semibold ink (the data being checked); per-floor money 15 px regular muted | shipped 2026-09-26 |
| Side footer | "Front total · 23 windows  $156" at 18 px, no longer a second display-size total | shipped 2026-09-26 |

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

### Phase 4 (refactoring-ui), 2026-09-26
Grayscale and blur tests on the result and summary screens, a CSS value census, and WCAG contrast.
**Score before: 5/10** (4 of 8 diagnostic rows failing).

| # | Issue | Heuristic | Severity (0-4) | Fix | Status |
|---|---|---|---|---|---|
| P4-1 | Blur test: the building name and the dark Count button outweighed the $156 total, which turns mid-grey in grayscale | Hierarchy (blur / grayscale) | 3 | Total in ink at 64 px; name down to 24 px; blue kept for things you can tap | **fixed** 2026-09-26 |
| P4-2 | Two primary actions on the result screen: "Count again" stayed filled after a count | One primary per screen | 2 | Outlined once the quote matches the photos; filled again when a photo changes | **fixed** 2026-09-26 |
| P4-3 | Per-floor window counts (the data being checked) at 14.5 px while the money beside them was bold | Labels vs values | 2 | Counts 18 px semibold ink; money regular muted | **fixed** 2026-09-26 |
| P4-4 | Two display-size totals (header and side footer) | Hierarchy | 2 | Side footer is a quiet "Front total · N windows" subtotal | **fixed** 2026-09-26 |
| P4-5 | 17 padding values, 14 gaps, 17 font sizes | Spacing / type scale | 2 | Snapped to 4/8/12/16/24/32/48 and 12/13/15/18/24/36/52/64 (padding 17 → 7 values, fonts 17 → 8) | **fixed** 2026-09-26 |
| P4-6 | Faint grey `#8A93AA` at 2.9:1 on small text ("Optional", tab amounts, stair labels) | Contrast (WCAG AA 4.5:1) | 3 | `#626D88` (4.7:1) light, `#7E88A1` (4.9:1) dark | **fixed** 2026-09-26 |
| P4-7 | "counted in 7.0s" in the stats line | Minimalism | 1 | Removed | **fixed** 2026-09-26 |
| P4-8 | The legend under the photo is three lines of small text | White space | 1 | Shorten in Phase 6 copy pass | backlog (Phase 6) |

**Score after: 9/10.** The blur test now puts the total first, grayscale reads, spacing and type are on
scale, and contrast passes. Remaining: P4-8.

## Microinteraction Inventory
| Interaction | Trigger/Rules/Feedback/Loops | Fix | Status |
|---|---|---|---|
