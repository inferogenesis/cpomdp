"""The epistemic/pragmatic crossover statistic over an EFE horizon.

The per-step epistemic value is fixed by the one-step EFE, but its aggregation over a
horizon is not — this module fixes it. The crossover contrasts two policies, a *walk*
(detour to sense, then exploit) and a *reach* (head for the goal), by the difference in
their summed EFE components:

    Δε(H) = Σ_k [ε_k(walk) − ε_k(reach)]      the accumulated epistemic pull
    Δc(H) = Σ_k [c_k(walk) − c_k(reach)]      the accumulated pragmatic gradient
    ΔG(H) = Δc(H) − Δε(H) = G(walk) − G(reach)

Both sides contrast the *same* two policies, summed over the horizon. That symmetry is
what the H=1 anchors force. ``ΔG(H) < 0`` is the crossover, and because ΔG is the
difference in the minimised objective, its sign flip is the argmin flip.
``H* = min{H : ΔG(H) < 0}``.

The statistic is defined before the sweep measures H* (see warrant_numbers.md and
``tests/test_crossover.py``). This module gives the statistic and the H* definition;
the H-sweep harness does the measuring.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

import jax.numpy as jnp
from jaxtyping import Array, Float64

from cpomdp.efe import policy_efe_ffg
from cpomdp.types import Belief

if TYPE_CHECKING:
    from cpomdp.backends.base import EfeBackend
    from cpomdp.selection import Preference

__all__ = ["CrossoverStatistic", "crossover_horizon", "crossover_statistic"]


@dataclass(frozen=True)
class CrossoverStatistic:
    """The horizon-H crossover contrast between a walk policy and a reach policy.

    A host-side value (a plain frozen dataclass, not a pytree): computed from the
    rollout and then inspected, never threaded through ``jit``.
    """

    horizon: int
    delta_epsilon: Float64[Array, ""]  # Δε, a value (higher = walk senses more)
    delta_c: Float64[Array, ""]  # Δc, a cost
    delta_g: Float64[Array, ""]  # ΔG = Δc − Δε, minimised; < 0 is the crossover

    @property
    def walk_wins(self) -> bool:
        """Whether the walk has overtaken the reach at this horizon (ΔG < 0)."""
        return bool(self.delta_g < 0)


def crossover_statistic(
    backend: "EfeBackend",
    belief: Belief,
    walk: Float64[Array, "H p"],
    reach: Float64[Array, "H p"],
    preference: "Preference",
    *,
    info_block,
) -> CrossoverStatistic:
    """The crossover contrast between two equal-length policies at their horizon.

    Runs the FFG rollout (``policy_efe_ffg``) for ``walk`` and ``reach``, then
    contrasts the summed components: ``Δε = ε(walk) − ε(reach)``,
    ``Δc = c(walk) − c(reach)``, ``ΔG = Δc − Δε``. The epistemic reading follows
    ``info_block`` - a node's block (via ``backend.block``) for the node-restricted
    pull,
    or the whole state.

    Raises:
        ValueError: If ``walk`` and ``reach`` have different horizons.
    """
    walk = jnp.asarray(walk, dtype=float)
    reach = jnp.asarray(reach, dtype=float)
    if walk.shape[0] != reach.shape[0]:
        raise ValueError(
            "walk and reach must have the same horizon; got "
            f"{walk.shape[0]} and {reach.shape[0]}"
        )

    _, walk_parts = policy_efe_ffg(
        backend, belief, walk, preference, info_block=info_block
    )
    _, reach_parts = policy_efe_ffg(
        backend, belief, reach, preference, info_block=info_block
    )

    delta_epsilon = walk_parts["epistemic"] - reach_parts["epistemic"]
    delta_c = walk_parts["pragmatic"] - reach_parts["pragmatic"]
    delta_g = delta_c - delta_epsilon

    return CrossoverStatistic(int(walk.shape[0]), delta_epsilon, delta_c, delta_g)


def crossover_horizon(
    backend: "EfeBackend",
    belief: Belief,
    walk_of: Callable[[int], Float64[Array, "H p"]],
    reach_of: Callable[[int], Float64[Array, "H p"]],
    preference: "Preference",
    *,
    info_block,
    max_horizon: int,
) -> int | None:
    """H* — the first horizon at which the walk overtakes the reach.

    ``H* = min{H : ΔG(H) < 0}`` over ``1..max_horizon``; ``walk_of`` and ``reach_of``
    map a horizon to its H-step policy (the anchor action tiled H times for the pair).
    Returns ``None`` if the walk never overtakes within the budget — a registered D3
    falsifier (no crossover at any feasible H), not a silent pass. This defines H*; the
    H-sweep harness measures it with the cost and conditioning table.
    """
    for horizon in range(1, max_horizon + 1):
        stat = crossover_statistic(
            backend,
            belief,
            walk_of(horizon),
            reach_of(horizon),
            preference,
            info_block=info_block,
        )
        if bool(stat.delta_g < 0):
            return horizon
    return None
