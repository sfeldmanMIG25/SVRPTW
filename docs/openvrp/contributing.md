# Adding a constraint (contributor guide)

The research meta-recipe as an on-ramp:

1. **Opt-in schema field** — extend `openvrp/schema/input.py` with a
   new field on `Constraints` or `VehicleClass`. Default value MUST
   keep existing behaviour bit-identical (no surprise regressions).

2. **Gated objective/evaluator term** — extend
   `openvrp/engine/objective.py::compose_objective` (or the adapter's
   penalties dict) with the new term. Gate it on the schema field
   being non-default so the cost stays bit-identical when the term
   isn't opted into.

3. **Unit tests** — add to `tests_openvrp/unit/test_<your_field>.py`
   covering: default-off ⇒ identical cost; on ⇒ penalty applied;
   round-trips through JSON.

4. **Conformance row** — add a test in
   `tests_openvrp/conformance/test_conformance_extra.py` named
   `test_E<n>_<your_constraint>` that exercises the constraint
   end-to-end and asserts the corresponding diagnostics field.

5. **Conditional operator registration** — if the search needs a new
   operator (e.g. peak-hour requires a `shift_start` move), extend
   `openvrp/diagnostics.py::active_operator_pool` so the operator is
   registered only when the axis is active. Off-axis operators
   measurably degrade result magnitude (the research record's
   "iter-5x peak_hour boundary" finding).

6. **mypy strict** — the schema modules are under `--strict`; the new
   field must carry a complete type + docstring.

The contributor is dogfooded once per release: a maintainer follows
this recipe on a toy constraint and the conformance gate must pass.
