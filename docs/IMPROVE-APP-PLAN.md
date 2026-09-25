# Improve App Plan

## Context
Started 2026-09-26. App: **Window Quote**, a web app (phone and desktop browser, Vercel) that turns
building photos into a per-floor window count and a price, for window-cleaning quotes.

Intake answers:
- **Job (Burke Allen, US window-cleaning owner):** quote on the spot. "When I'm standing in front of
  a building, I want an accurate per-floor window count and price in a minute, so I can give the
  customer a quote before I leave."
- **Roughest area:** trusting the count. It isn't clear the windows and floors are right without
  checking each one, and the flags and fixes take thought.
- **Evidence:** none yet from real users, only our own testing and the hand-count benchmark
  (`eval.py`, 89%). Optional continuous-discovery step added: Burke's demo reaction is the first
  real evidence.
- **Platform:** web; phones on site are the main context. Phase 8 applies.
- **Upsell surfaces:** none (no paywall, trial or badges). Phase 7 skipped.
- **Existing docs:** none. Phases create CUSTOMER, DESIGN, EXPERIMENTS, POSITIONING and PRODUCT.
- **Code-level fixes:** every item from the code review was fixed before this journey (`05a94f5`,
  see AUDIT.md).
- **Pace:** phase by phase, with approval at each phase.

## Phase Status
| Phase | Skill | Status | Artifact | Date |
|---|---|---|---|---|
| 1 | jobs-to-be-done | done | CUSTOMER.md | 2026-09-26 |
| 2 | ux-heuristics | done | DESIGN.md, EXPERIMENTS.md | 2026-09-26 |
| 3 | design-everyday-things | done | DESIGN.md, EXPERIMENTS.md | 2026-09-26 |
| 4 | refactoring-ui | pending | DESIGN.md, EXPERIMENTS.md | |
| 5 | microinteractions | pending | DESIGN.md, EXPERIMENTS.md | |
| 6 | made-to-stick | pending | POSITIONING.md, EXPERIMENTS.md | |
| 7 | influence-psychology | skipped: no in-app upsell, paywall or social-proof surfaces | POSITIONING.md, EXPERIMENTS.md | 2026-09-26 |
| 8 | high-perf-browser | pending | DESIGN.md, EXPERIMENTS.md | |
| 9 | steve-jobs-design-review | pending | PRODUCT.md, DESIGN.md, EXPERIMENTS.md | |
| opt | continuous-discovery | pending (after the Burke demo) | CUSTOMER.md | |
Statuses: pending · in-progress · awaiting-evidence · done · deferred: <reason> · skipped: <reason>

## Key Decisions
| Date | Phase | Decision | Rationale |
|---|---|---|---|
| 2026-09-26 | Intake | Job = quote on the spot, with speed plus trust in the number | Burke's own words: count by floor, price per floor, "whether it would work for the types of buildings we estimate" |
| 2026-09-26 | Intake | Front-load "trusting the count" | Roughest area named by Abdur |
| 2026-09-26 | Intake | Skip phase 7 | No upsell surfaces exist |
| 2026-09-26 | Intake | Run phase by phase with approval | Abdur's choice |
| 2026-09-26 | 1 | Job statement approved as written | Reads without the product; matches Burke's own words |
| 2026-09-26 | 1 | Worst dimension = **functional accuracy** | Abdur: count accuracy before trust or polish; phases 2-9 aim at correct counts that are visibly correct |
| 2026-09-26 | 1 | Most likely leak = **Big Hire (the demo)** | If the demo doesn't convince on his building types he never starts; demo-critical flows go first |
| 2026-09-26 | 3 | Fix all five severity-3 issues now (near-miss tap + pointer highlight, no silent $0 price, removing a photo removes its side, undo for sample and New quote); severity 1-2 to backlog | Abdur's choice. Rule: guard-rails where the mistake can be made impossible, Undo instead of confirm dialogs. Score 4/10 → 7/10 |
| 2026-09-26 | 2 | Fix all four severity-3 issues now (framed photo, land on result on phones, undo a recount, 44 px targets); severity 1-2 to backlog | Abdur's choice; the four are the demo-critical ones. Score 6/10 → 8/10 |

## Next Actions
- [x] Phase 1: job statement, three dimensions, alternatives (done 2026-09-26)
- [x] Phase 2: heuristic evaluation; 4 severity-3 fixes shipped and browser-tested (2026-09-26)
- [x] Phase 3: 5 severity-3 error-proofing fixes shipped and browser-tested (2026-09-26)
- [ ] Phase 4: refactoring-ui visual hierarchy pass (Claude, next)
- [ ] Backlog P3-6..P3-13 guard-rails, stale settings, per-action undo (Claude)
- [ ] Backlog UX #5 (clickable "things to check") and #6 (floor row highlights its windows) (Claude)
- [ ] Accuracy work routed to code (from Phase 1 decision): glass-door, balcony and storefront detection as their own types; see AUDIT.md P1 #3 (Claude)
- [ ] Interview Burke on his timeline (first quote, what he tried, what scared him) to evidence the emotional and social dimensions (Abdur, on the first call)
- [ ] Set `ACCESS_CODE` in Vercel before the link goes to Burke (Abdur)
- [ ] After the demo: record Burke's reaction as the first evidence (continuous-discovery)
