"""Post hoc, not registered: which EFE term carries the crossover in this family.

PREREG.md registers groups, not this. FINDINGS.md reports it as a post hoc diagnostic
of mechanism. It tests no group and changes no registered outcome.

For every cell ``sweep.py`` found valid, the margin is recomputed with one term
removed from both policies' per-step G, and the crossover is re-read with the same
rule:

- ``no_epistemic``: G = pragmatic, the full pragmatic with the ambiguity term in.
- ``no_ambiguity``: G = goal-mismatch mean term - epistemic. ``1/2 tr(Lambda S)`` is
  dropped.

Run from the repository root::

    uv run --extra examples python experiments/collapse/ablation.py
"""

import numpy as np
from jax import numpy as jnp
from sweep import (
    ACTION_BOUND,
    HORIZON,
    OBSERVATION_GAIN,
    _trace,
    categorise,
    grid,
)


def term_margins(cell) -> dict[str, np.ndarray]:
    """The registered margin and the two ablated ones, for T = 1..HORIZON."""
    model, prior, preference = cell.model(), cell.prior(), cell.preference()
    precision = 1.0 / cell.goal_width**2  # Lambda
    target = OBSERVATION_GAIN * cell.goal()  # o*
    parts = {}
    for name, action in (("epi", +ACTION_BOUND), ("prag", -ACTION_BOUND)):
        trace = _trace(model, prior, jnp.full((HORIZON, 1), action), preference)
        predicted_obs = OBSERVATION_GAIN * np.asarray(trace.predicted_mean[:, 0])
        mean_term = 0.5 * precision * (predicted_obs - target) ** 2
        parts[name] = {
            "full": np.asarray(trace.g),
            "no_epistemic": np.asarray(trace.pragmatic),
            "no_ambiguity": mean_term - np.asarray(trace.epistemic),
        }
    return {
        key: np.cumsum(parts["epi"][key]) - np.cumsum(parts["prag"][key])
        for key in ("full", "no_epistemic", "no_ambiguity")
    }


def main() -> None:
    """Print, per ablation, how the registered crossover moves."""
    summary = {"no_epistemic": [], "no_ambiguity": []}
    categories = {"no_epistemic": {}, "no_ambiguity": {}}
    for cell in grid():
        margins = term_margins(cell)
        category, _, h_full = categorise(margins["full"])
        if category != "valid":
            continue
        for key in summary:
            ablated_category, _, h = categorise(margins[key])
            categories[key][ablated_category] = (
                categories[key].get(ablated_category, 0) + 1
            )
            if ablated_category == "valid":
                summary[key].append(abs(np.log(h / h_full)))
    for key, gaps in summary.items():
        print(f"{key}: categories {dict(sorted(categories[key].items()))}")
        if gaps:
            gaps = np.array(gaps)
            print(
                f"  |ln H_ablated - ln H_full| median {np.median(gaps):.4f}, "
                f"max {gaps.max():.4f}, within 5%: "
                f"{np.mean(gaps <= np.log(1.05)):.3f}"
            )


if __name__ == "__main__":
    main()
