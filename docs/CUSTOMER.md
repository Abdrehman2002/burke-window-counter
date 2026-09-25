# Customer

Primary customer: **Burke Allen**, owner of a US window-cleaning business. He came in from LinkedIn and
asked for a demo video before any call. He had already tried to do this with AI himself and failed.

## Job Statement
> **When I'm quoting a building I haven't cleaned before, I want to know how many windows are on each
> floor and what that comes to at my per-floor rates, so I can give the customer a firm price on the
> spot, without underpricing the high floors.**

His words: *"take pictures of the building and have it count the windows by each floor, and then have a
different price per window for each floor... Go around to four sides of the building and then add
everything up."* · *"It has the wrong number of windows even though I taught it what a window is like."*
· *"we have different types all the time, each and every day."*

## Job Dimensions
| Dimension | What he needs | Where we underdeliver today | Priority |
|---|---|---|---|
| **Functional** | A correct count per floor and per side, priced, in about a minute, on any building type | 89% of windows on the right floor on a 7-building benchmark (Wikimedia, not his buildings). Misses glass behind balconies, French doors and glass doors; counts a window in a door; some corner shots need a drawn area. **Errors cost most at the top**: with ×2 per floor, one missed floor-8 window is $384 | **#1: the worst gap (decided 2026-09-26)** |
| **Emotional** | Confidence the number is right; no fear of under-quoting | He starts distrustful ("wrong number of windows even though I taught it"). Amber boxes and "check" flags hand the doubt back to him, and nothing summarises how sure the count is | #2 |
| **Social** | Look quick and professional in front of the customer | No branded, customer-ready quote; quotes aren't saved; the phone flow isn't tuned for the sidewalk | #3 |

**Evidence status:** functional numbers are measured (`eval.py`, `test_batch.py`). The emotional and
social notes are inferred from his messages until he is interviewed.

## Forces of Progress
| Force | Burke |
|---|---|
| Push | Counting by hand is slow; his own AI attempts gave wrong counts |
| Pull | A count by floor, priced his way, automatically, per side and total |
| Anxiety | "Is this count right?" The doubling price makes top-floor errors expensive. He has already been burned by AI |
| Habit | Counting by eye with a clipboard; pricing on experience |

## Big Hire / Little Hire
- **Big Hire = the demo video.** His test: *"That would give me a good idea of how accurate it is and
  whether it would work for the types of buildings we estimate."* **Decided 2026-09-26: this is where
  we are most likely to lose him.** The demo must show correct counts on building types he recognises,
  and be honest where it's unsure.
- **Little Hire = each quote on site.** It wins only if checking the count is faster than counting by
  eye. Not yet measurable: no usage data. Metrics to add: time per quote, and edits per quote.

## Competing Alternatives
| Alternative | Why hired | Weakness |
|---|---|---|
| Count by eye + clipboard + calculator | Free, trusted, no new tool | Slow on big buildings; easy to miss windows on upper floors |
| Square-footage or flat-rate estimate | Fast | Inaccurate on unusual buildings; under- or over-prices |
| ChatGPT-style AI on a photo | Already at hand; he tried it | Wrong counts on repeated windows; no floors; he no longer trusts it |
| QuoteIQ AI Estimator (photo-based, window-cleaning CRM) | Built for window cleaners; bundled with a CRM | Opaque count; not his per-floor pricing |
| Satellite/aerial measuring (e.g. MapMeasure) | Quote without a visit | Can't see facades or windows per floor |
| Ask the customer for a window count | No effort | Unreliable |
| **Non-consumption:** don't quote big buildings, or quote after a second visit | Avoids the risk | Lost jobs and wasted trips |

## Diagnostic Score
**7/10.** Missing: a timeline interview with Burke (emotional and social dimensions are assumptions),
and Little Hire metrics (time and edits per quote).
