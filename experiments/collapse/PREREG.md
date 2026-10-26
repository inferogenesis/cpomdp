# Pre-registration: does H\* collapse onto a dimensionless group?

<!-- cspell:words prag parameterisation -->

**Date:** 2026-10-26
**Issue:** #58
**Status:** registered before any crossover horizon was computed. This file is committed on its own, ahead of `sweep.py`, `results.csv`, `collapse.png` and `FINDINGS.md`. Anything that changes after this commit goes into `FINDINGS.md` as a named deviation. This file is not edited.

## The question

`H*` is the shortest planning horizon at which an `R(x)` agent's epistemic value outweighs its pragmatic gradient. It has been measured at particular integers on particular models. Those integers belong to a parameterisation. The question is whether `H*` is a function of one dimensionless group across a family of models. If it is, the threshold is structural. If it is not, each measured `H*` is a fact about its model and is reported as one.

## What was seen before this was written

The design pass recorded in issue #58 included a grid-scoping run. That run printed one number per grid: the fraction of cells whose margin crosses zero exactly once inside `T <= 40`. It did not compute a crossover horizon, run a regression or draw a plot. The grid below is the best of the grids scoped (305 of 432 cells crossing cleanly, 71%). Choosing ranges so that the experiment measures something is not tuning for collapse. It is disclosed here because it happened.

The analytic results below were derived from the definitions, not read from data.

## Model

Scalar, conditionally linear-Gaussian, one sensor family held fixed across the sweep:

```text
x_{k+1} = a*x_k + b*u_k + w_k        w_k ~ N(0, q)
o_k     = c*x_k + v_k                v_k ~ N(0, R(x_k))
R(x)    = r0 * exp(-(x - x_s) / lam)
```

The model is built from the library unchanged: `LinearGaussianModel` with a `CallableSensor`, scored with `cpomdp.efe.policy_efe_trace` (full EFE, ADR-005: full pragmatic including `1/2 tr(Lambda S)`, minus the state information gain).

Fixed for every cell: `a = c = r0 = u_max = 1`, `x_s = mu_0^- = 0`.

- `a = 1` and the prior covariance is `Sigma_0 = Sigma_0^- - q`, so the first predicted covariance the rollout forms is exactly the swept `Sigma_0^-` and `mu_0^- = mu_0 = 0`.
- The goal is on the blind side: `x* = mu_0^- - d = -d`. `R` falls as `x` rises, so the goal and the sharp sensing pull in opposite directions.
- The preference is observation-space, target `o* = c*x* = -d`, precision `Lambda = 1/sigma_g^2`.

## Definitions (evaluated at the planning start)

```text
ell_R   = |d ln R / dx|^-1  at mu_0^-   =  lam
delta_u = |b| * u_max
kappa   = delta_u / ell_R
s       = c^2 * Sigma_0^- / R(mu_0^-)
g_epi   = 1/2 * kappa * s/(1+s)
c_prag  = delta_u * |mu_0^- - x*| / sigma_g^2
Pi      = g_epi / c_prag                       # the hypothesised group
```

With the fixed values above, `s = Sigma_0^-` and `|mu_0^- - x*| = d`.

## The two policies and the margin

Actions lie in `[-u_max, u_max]`. Both policies are constant open-loop sequences.

- `pi_prag` minimises the goal-mismatch mean term `1/2 (o+ - o*)^2 Lambda` alone at `T = 1`. That is the term `c_prag` is the derivative of. It is *not* the library's full pragmatic, whose `1/2 tr(Lambda S)` pulls toward sharper sensing under a monotone `R(x)`. The `T = 1` minimiser is `clip((x* - a*mu_0)/b, -u_max, u_max) = clip(-d/b, -1, 1)`. The grid has `min(d) = 3 >= max(b) * u_max = 2`, so it is `-u_max` in every cell.
- `pi_epi` maximises the epistemic term alone at `T = 1`. `R` is monotone decreasing in `x`, so the maximiser is `+u_max` in every cell.

`sweep.py` checks both claims numerically on a 201-point action grid in every cell and stops if either fails.

Both policies are scored under the full EFE. The margin is

```text
Delta(T) = G(pi_epi, T) - G(pi_prag, T),   T = 1 .. 40
```

`G(pi, T)` for every `T` comes from one `policy_efe_trace` call at `T = 40` per policy per cell, followed by a prefix sum. The rollout is a causal `lax.scan`, so the first `T` steps of the 40-step trace are the `T`-step rollout.

- `H*_int` is the smallest `T` with `Delta(T) < 0`.
- `H*_cts` interpolates `Delta` linearly through zero between `H*_int - 1` and `H*_int`:
  `H*_cts = (H*_int - 1) + Delta(H*_int - 1) / (Delta(H*_int - 1) - Delta(H*_int))`.
  The analysis uses `H*_cts`, because quantisation at small `H*` would otherwise dominate the residuals.

This is a two-policy comparison, not policy optimisation. It bounds the crossover of the optimal agent from above and is not that crossover. The findings carry this caveat in full.

## Grid

A full factorial, 4 x 4 x 3 x 3 x 3 = 432 cells, `T` from 1 to 40:

| Parameter | Values |
| --- | --- |
| `lam` | `0.5 * 10^(i/3)`, `i = 0..3` (0.5, 1.07722, 2.32079, 5.0) |
| `b` | `0.2 * 10^(i/3)`, `i = 0..3` (0.2, 0.430887, 0.928318, 2.0) |
| `sigma_g` | 0.5, 1.0, 2.0 |
| `d` | 3.0, 6.0, 12.0 |
| `(q, Sigma_0^-)` | (0.01, 0.02), (0.02, 0.10), (0.05, 0.50) |

## Cell categories

Every cell is written to `results.csv` with a category, whichever it gets. Precedence runs top to bottom: a cell takes the first category that applies.

1. `nonfinite`: any `Delta(T)` is NaN or infinite.
2. `degenerate`: any `Delta(T)` is exactly zero.
3. `multi_cross`: `Delta` changes sign more than once over `T = 1..40`.
4. `cross_at_1`: `Delta(1) < 0`. There is no positive margin to interpolate from.
5. `no_cross`: `Delta(T) > 0` for every `T`.
6. `valid`: exactly one sign change, from positive to negative, at some `H*_int >= 2`.

Only `valid` cells enter the analysis. The count in each category is reported. Because validity depends on the outcome, the valid set is a selected sample, and the findings say which regions of the grid it under-covers.

## Analysis and acceptance

All regressions are ordinary least squares with an intercept on the valid cells, with `ln H*_cts` as the response. `R^2` is the usual coefficient of determination.

Collapse onto `Pi` is accepted only if all three criteria pass.

1. **Fit.** The `R^2` of `ln H*_cts` on `ln Pi` alone exceeds 0.90.
2. **Beats every raw parameter.** That `R^2` exceeds, by at least 0.15, the `R^2` of `ln H*_cts` on `ln p` for each single raw parameter `p` in `lam`, `b`, `sigma_g`, `d`, `Sigma_0^-` and `q`, one at a time. Every one, not the average.
3. **Agreement across the split.** Split the valid cells at the grid median of `lam` (`{0.5, 1.077}` against `{2.321, 5.0}`) and of `b` (`{0.2, 0.431}` against `{0.928, 2.0}`), giving four disjoint quadrants. The issue calls these octants. Two binary splits make four, and the criterion is applied to the four. A pair of valid cells in different quadrants *overlaps in `Pi`* when `|ln Pi_i - ln Pi_j| <= 0.05`. Every overlapping pair must satisfy `|ln H*_cts,i - ln H*_cts,j| <= ln 1.15`, which means they agree to within 15%. All pairs must agree. The fraction that does is reported as well.

The outcome vocabulary follows ADR-029. Each criterion reports `PASS`, `FAIL` or `NOT APPLICABLE`.

- Criterion 3 is `NOT APPLICABLE` if no overlapping cross-quadrant pair exists.
- The whole test is `NOT APPLICABLE` (inconclusive) if fewer than 100 cells are valid.
- A `NOT APPLICABLE` criterion is evidence for nothing, so it never counts toward acceptance.

`Pi` does not depend on `b`, because `delta_u` cancels (see below). Cells that differ only in `b` therefore share `Pi` exactly, and criterion 3 consists largely of those pairs. That makes it the sharpest test in the grid: if `H*_cts` moves with `b`, `Pi` is not the whole story.

Failure is a reportable result.

## If `Pi` fails: the alternatives, registered now

These are fitted only if collapse onto `Pi` is not accepted. Each group named in 1 to 5 is put through criteria 1 to 3 with that group in place of `Pi`. Every group tried is reported with its `R^2`, including the ones that did worse than `Pi`.

1. `kappa` alone. It sets how fast the two policies separate in `ln R`, so it carries the sub-leading structure that `Pi` cannot see.
2. `Pi_inf = Pi * (1 + s) / (2s) = sigma_g^2 / (4 * ell_R * |mu_0^- - x*|)`. This is `Pi` with the SNR factor stripped out.
3. `s` alone. In this grid `s = Sigma_0^-` identically, because `r0 = c = 1` and `mu_0^- = x_s`. It is a raw parameter here and cannot beat itself by 0.15, so criterion 2 is `NOT APPLICABLE` for it and it cannot be accepted as a collapse. It is reported for its `R^2` only.
4. `N_sat = ln(1/s) / kappa`. This is the number of full-authority steps needed to drive the SNR from `s` to order one, and it already has horizon dimensions.
5. `Pi_amb = r0 / (2 * c^2 * ell_R * |mu_0^- - x*|)`. This is the ambiguity channel `c_prag` omits. The pragmatic policy walks into the blind region, where `1/2 tr(Lambda S)` grows like `r0 * exp(k*kappa)`, and `Pi_amb` is that term's initial rate over the mean-cost rate.
6. The two-group fit `ln H*_cts ~ beta * ln Pi + gamma * ln kappa`. It is reported as a two-group fit with its `R^2`, `beta` and `gamma`. It is not a collapse.

If more than one of the alternatives 1 to 5 passes, all of them are reported. None is singled out as the answer, and the number of groups tried is stated next to any that passed. No group beyond these six is introduced after the data are seen. If a new one suggests itself, it is named in `FINDINGS.md` as post hoc, and it is not tested on this grid.

## Analytic expectation (stated before the data)

**`Pi` is independent of `delta_u`.** Both `g_epi` and `c_prag` are proportional to `delta_u`, so `b` cancels:

```text
Pi = sigma_g^2 * s / (2 * ell_R * |mu_0^- - x*| * (1 + s))
```

**Leading order predicts no finite crossover beyond `T = 1`.** Under the two constant policies the predicted means separate linearly, `mu_k = +/- k*delta_u`. The mean pragmatic difference at step `k` is exactly `2k * c_prag`. The epistemic difference at leading order in the log-`R` separation is `2k * g_epi`. Summing over steps:

```text
Delta(T) ~ T(T+1) * c_prag * (1 - Pi)
```

The sign does not depend on `T`, so leading order gives `H* = 1` when `Pi > 1` and no crossing when `Pi < 1`. Any finite `H* > 1` lives in what that expansion drops. That is the saturation of the epistemic term as the operating SNR moves, and the ambiguity term `1/2 tr(Lambda S)`. **The hypothesis is a claim about sub-leading structure, and `Pi` alone is expected to be a weak candidate.** That expectation is registered here so it cannot be presented later as a discovery.

**Deep-saturation ratio.** Once `pi_epi` has driven `R` far down, its per-step gain tends to `kappa/2` rather than `2*g_epi`, giving `Pi_inf = sigma_g^2 / (4 * ell_R * d)`. A finite crossover needs roughly `Pi < 1 < Pi_inf`, which requires `s < 1`. Every grid value has `s <= 0.5`.

## Out of scope

- No change to `src/cpomdp`, check gates, demos, the warrant ledger or the ledger files.
- No new CI assertions.
- No biological framing, and no claim about "curiosity" as a general phenomenon.
- No model constant is tuned to improve the appearance of collapse. The grid and constants above are final.

## Deliverables

`sweep.py` writes `results.csv` (every cell, with its parameters, groups, category, `H*_int` and `H*_cts`) and `collapse.png` (`ln H*_cts` against `ln Pi`, coloured by quadrant). `FINDINGS.md` reports the category counts, each criterion's outcome, the alternatives if they were triggered, every deviation from this file, and the two-policy caveat in full.
