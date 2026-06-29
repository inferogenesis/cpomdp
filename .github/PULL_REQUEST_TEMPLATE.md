<!--
Thanks for contributing to cpomdp. Keep this description focused; delete any
section that doesn't apply. For anything beyond a small fix, please open an
issue or discussion first so the design can be agreed before the code.
-->

# Summary

<!-- What does this change do, and why? One or two paragraphs is plenty. -->

## Type of change

<!-- Tick all that apply. The PR title should follow Conventional Commits
(feat:, fix:, docs:, test:, chore:, ...) — see CONTRIBUTING.md. -->

- [ ] `fix` — bug fix (no API change)
- [ ] `feat` — new behaviour or API
- [ ] `docs` — documentation only
- [ ] `test` — tests only
- [ ] `refactor` / `chore` — no behaviour change
- [ ] Breaking change (existing API or numerical results change)

## Related issues and design notes

<!-- Link the issue this closes and any decision record it implements or
revises. Non-trivial behaviour should trace back to an ADR or RFC. -->

- Closes #
- Relevant ADR (`DECISIONS.md`):
- Relevant RFC (`rfcs/`):
- Build-plan item (`BUILD_PLAN.md`):

## Approach

<!-- The reasoning behind the implementation: alternatives weighed, trade-offs
made, anything a reviewer would otherwise have to reconstruct from the diff. -->

## Validation

<!-- How do we know this is correct? The linear-Gaussian regime has exact
oracles (per-step Kalman, RxInfer, brute-force EFE) — say which one this is
checked against, or why no oracle applies. -->

- Oracle / test checked against:
- New or changed tests:

```text
# Paste the relevant pytest output, e.g.
# uv run --no-sync pytest -m "not rxinfer"
```

## Energy / hot-path impact

<!-- The project's driving objective is per-cycle energy efficiency (RFC-001).
Note any change to the per-decision cost, and keep front-loading intact where
the regime allows (solve once at construction, cheap loop body). If this adds
per-cycle compute, confirm it's measurable and attributable, not buried in the
loop. Write "none" if the hot path is untouched. -->

- Per-cycle cost impact:
