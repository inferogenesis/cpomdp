"""Why the constant-action family cannot express the detour-then-exploit crossover.

The interesting multi-step behaviour on the coupled-tree cue task is a *two-phase*
sequence: drive to the cue, sense the hidden context, then commit to the arm the cue
revealed. That policy varies across the horizon. The constant-action family holds one
action for the whole horizon, so a constant "walk" toward the cue *overshoots* it — the
agent senses at the first step and then moves straight past, so the information is
acquired once and never exploited.

This sweep makes that concrete. It runs the crossover statistic for the constant
reach/walk pair over the horizon and shows two things:

- the **epistemic pull plateaus** instead of accumulating — a flat Δε is the fingerprint
  of information acquired once and wasted;
- the **pragmatic gradient never yields**, so ΔG stays positive and the reach wins at
  every horizon — there is no crossover.

That null is a *search-family artefact*, not a property of the objective. It is the
concrete reason the exhaustive varying-sequence search (``EnumeratedEFESearch``) is
necessary rather than optional: the crossover horizon lives there, where the best policy
is free to become a two-phase walk. Read this demo as the motivation for that search.

``--check`` asserts the null with no plotting deps; the bare command prints the table::

    uv run --no-sync python examples/ffg/crossover_sweep.py --check
    uv run --no-sync python examples/ffg/crossover_sweep.py
"""

from __future__ import annotations

import sys

import epistemic_dissociation_figure as demo
import jax.numpy as jnp
import numpy as np

from cpomdp.crossover import crossover_horizon, crossover_statistic
from cpomdp.selection import Preference

# The feasible sweep budget for the constant pair. The null is robust well past this;
# the crossover is a varying-sequence phenomenon, so the constant pair's behaviour at
# large H is beside the point — it is not where the crossover can live.
CROSSOVER_MAX_H = 6
# Δε varies by less than this across the sweep. A flat epistemic pull is the mechanism:
# the constant walk overshoots the cue, so sensing does not compound.
EPS_PLATEAU = 0.2


def _crossover_setup():
    """``(backend, belief, preference, info_node)`` for the coupled-tree model."""
    backend = demo.build_backend(epistemic_alive=True, cue_x=demo.CUE_DETOUR_X)
    belief = demo.start_belief()
    preference = Preference(
        target=[0.0, 0.0],
        precision=[[demo.GOAL_PRECISION, 0.0], [0.0, demo.INFO_PRECISION]],
    )
    info_node = demo.CONTEXT
    return backend, belief, preference, info_node


def _reach_walk_actions():
    """The reach (``argmin G``, prior-ward) and walk (``argmax ε``, cue-ward)."""
    scan = demo._boundary_scan(alive=True, cue_x=demo.CUE_DETOUR_X)
    walk = float(scan["grid"][int(np.argmax(scan["epistemic"]))])
    reach = float(scan["grid"][int(np.argmin(scan["total"]))])
    return reach, walk


def _constant(action, horizon):
    return jnp.full((horizon, 1), action)


def sweep(max_horizon: int = CROSSOVER_MAX_H):
    """Per-horizon crossover statistic ``(H, Δε, Δc, ΔG)`` for the constant pair."""
    backend, belief, preference, info_node = _crossover_setup()
    reach, walk = _reach_walk_actions()
    rows = []
    for horizon in range(1, max_horizon + 1):
        stat = crossover_statistic(
            backend,
            belief,
            _constant(walk, horizon),
            _constant(reach, horizon),
            preference,
            info_node=info_node,
        )
        rows.append(
            (
                horizon,
                float(stat.delta_epsilon),
                float(stat.delta_c),
                float(stat.delta_g),
            )
        )
    return rows


def check() -> None:
    """Assert the constant-family null: no crossover, and a flat epistemic pull."""
    backend, belief, preference, info_node = _crossover_setup()
    reach, walk = _reach_walk_actions()
    rows = sweep()

    print(f"Constant reach (u={reach:+.1f}) vs walk (u={walk:+.1f}) over the horizon:")
    print(f"  {'H':>2} {'Δε (pull)':>11} {'Δc (grad)':>11} {'ΔG':>11}   outcome")
    for horizon, de, dc, dg in rows:
        verdict = "walk wins" if dg < 0 else "reach wins"
        print(f"  {horizon:>2} {de:>11.4f} {dc:>11.4f} {dg:>11.4f}   {verdict}")

    epistemic_pulls = [de for _, de, _, _ in rows]
    crossovers = [dg for _, _, _, dg in rows]

    # H=1 is the opening: moving to the cue buys information (Δε > 0) but costs more
    # goal-distance (Δc > Δε), so the reach wins (ΔG > 0). This is the anchored order.
    _, de1, dc1, dg1 = rows[0]
    assert de1 > 0
    assert dc1 > de1
    assert dg1 > 0

    # The null: the constant pair never crosses over — the reach wins at every horizon,
    # and the horizon finder returns None rather than an H*.
    assert all(dg > 0 for dg in crossovers)
    h_star = crossover_horizon(
        backend,
        belief,
        lambda h: _constant(walk, h),
        lambda h: _constant(reach, h),
        preference,
        info_node=info_node,
        max_horizon=CROSSOVER_MAX_H,
    )
    assert h_star is None

    # The mechanism: the epistemic pull plateaus rather than accumulating.
    pull_range = max(epistemic_pulls) - min(epistemic_pulls)
    assert pull_range < EPS_PLATEAU

    print()
    print(
        f"  No crossover at H <= {CROSSOVER_MAX_H}, and Δε is flat "
        f"(range {pull_range:.3f} < {EPS_PLATEAU}):"
    )
    print("  the constant walk overshoots the cue: it senses once and never exploits.")
    print("  The crossover is a varying sequence — it lives in the exhaustive")
    print("  EnumeratedEFESearch, not in the constant-action family. -- PASS")


def main():
    """``--check`` asserts the null; the bare command prints the sweep table."""
    if "--check" in sys.argv:
        check()
        return
    for horizon, de, dc, dg in sweep():
        print(f"H={horizon}  Δε={de:.4f}  Δc={dc:.4f}  ΔG={dg:.4f}")


if __name__ == "__main__":
    main()
