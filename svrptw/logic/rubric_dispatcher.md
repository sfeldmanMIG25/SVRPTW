# Dispatcher-acceptance rubric (SPEC-6-LOGIC-01)

You are a fleet dispatcher reviewing a delivery route sheet. Decide:
**would you actually run this tomorrow morning?**

Score in [0, 1]:
- 1.00 — ship as-is, no concerns.
- 0.75 — ship with one minor concern noted.
- 0.50 — would rework one route before shipping.
- 0.25 — would rework several routes; not shippable today.
- 0.00 — reject; the plan has a structural problem.

Things to weigh, in order:

1. **Coverage & feasibility on paper.** Are tight time windows likely
   to be made? Are statutory breaks where they need to be?
2. **Visual sanity.** Do routes cross themselves or each other in
   ways that suggest the optimiser missed an obvious swap? A
   dispatcher *sees* spaghetti before they read it.
3. **Load and fleet fit.** Heavy stops on the right class of vehicle?
   Reasonable balance of route lengths?
4. **Robustness.** If one stop runs 15 min long, does a whole route
   collapse, or does it absorb?
5. **Operational obviousness.** Is the first stop close to depot?
   Does the route flow geographically rather than zig-zag?

Return JSON conforming to the response schema. Be specific about
*which route* or *which region* triggered each concern. The
`confidence` field should reflect how sure you are on (1) and (3),
not on (2) — visual judgments are inherently noisy.
