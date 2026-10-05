# Notes for the written report

Facts that were checked on the full data and must appear in the report.

## Data cleaning
- **Relevant sections.** D ONE's rule is "at least 20 weighings over at least 5 days". It was re-checked on all 58,720 sections and holds for 100% of them. Excluded sections have a median of 9 weighings over 4.8 days, too short for a weekly KPI.
- **Duplicates.** A duplicate is the same scale with the same timestamp to the millisecond. There are 5,499 groups.
  - 4,199 groups are identical copies; only one copy is kept.
  - 1,300 groups carry two different labels. One copy is kept for the weight chain, but it is left out of the KPI (rule B0).
- **Invalid weights.** The 5 rows deleted for impossible weights (144 kg to 50 t) are exactly the rows D ONE put into the `invalid weight` bin.
- **Trigger configuration.**
  - Rows with an explicitly non-standard min-wait or delay are kept but left out of the KPI. These are 57k rows (0.8%), for example min-wait 60 min.
  - Rows where min-wait is empty are treated as standard. At sensitivity 8 they behave like standard rows: TOO_MANY fraction 0.79 vs 0.83 (20 kg) and 0.69 vs 0.69 (6 kg).

## How TOO_FEW is created (limits of the KPI)
- TOO_FEW can only come from a MANUAL weighing.
  - 85% of manual weighings come at least 24 h after the previous weighing (the "24 h no activity" check). The rest are triggered by people.
  - These checks cluster at fixed hours (about 15–16 h and 20–21 h in the data), so they look like a scheduled job. It only weighs scales that were idle for 24 h.
- A pick the sensor missed is **not** counted as TOO_FEW if the next weighing is automatic. That automatic weighing sees the accumulated change and is labelled OK. TOO_FEW is therefore a **lower bound** of the misses.
- With a long min-wait (for example 60 min), picks during the lock-out are caught by the next automatic weighing. This lowers TOO_MANY and hides the delay; it does not raise TOO_FEW.

## Methods: what is final and what is future work
- Parameter importance: the main result is the spread between the best and worst group. Shapley R² (LMG) is mentioned as a refinement. It splits the explained variance fairly between correlated parameters, but it measures "where today's differences come from", not "how strong a knob is".
