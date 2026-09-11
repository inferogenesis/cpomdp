"""The averaged inference gap, ``E_{y∼p*}[ KL(q ‖ p(x|y)) ]``.

What a Gaussian filter loses by being Gaussian, in nats, averaged over the readings
the true model actually produces. Zero when the approximation is exact, which under a
fixed ``R`` it is, and positive under a state-dependent ``R(x)`` for reasons no
resolution removes.

Three conventions are pinned, and they are the same three
``research.checks.gap_kernel`` pins. They belong to the quantity rather than to either
implementation, and a fork between the two would be two different numbers under one
name:

- **Direction is reverse**, ``KL(q ‖ p)``, the agent against the exact posterior. The
  forward direction is a different number and no declared figure refers to it.
- **Where the approximation reads its noise is the rule's business**, not this
  module's. The plug-in rung freezes ``R̂ = R(μ)`` at the prior mean across the
  update. That decision lives in whatever is passed as ``approximate_posterior``.
- **The average is under the true ``p*(y)``**, never the agent's own predictive.
  ``p*`` falls out of the same product the exact posterior comes from, so it costs
  nothing extra and cannot drift from it.

That last point is why the observation grid needs the same care as the state grid.
``p*`` under an unbounded ``R(x)`` is a scale mixture, so its tails are exponential
and a half-width of "k standard deviations" sizes a Gaussian tail against one that is
not (``research.checks.predictive_truncation`` measures where a given ``k`` fails).
The box is declared by the caller and what fell outside it is reported.

Two boxes can be wrong here and they fail differently. A narrow observation box loses
predictive mass, which ``predictive_mass`` reports. A narrow *state* box is worse,
because it goes unreported by that number: an extreme reading drags the exact
posterior toward the edge, and past it the joint integrates to far less than ``p*(y)``
while the predictive mass, dominated by the core, still reads one.
``worst_edge_ratio`` is the second reading, and it is the one that catches a state box
sized from the prior while the observation box was sized for the tails.

A rule may decline a reading. An iterating rung that spends its budget without
converging returns a mean and a covariance that are both wrong by an unmeasured
amount, and near its pole a truncated run returns the prediction looking converged
(ADR-056). So it answers ``Void`` instead of a belief, and the sweep records the node
as unmeasured: its divergence is NaN, its predictive weight is added up in
``voided_mass``, and ``value`` is the expectation over the readings that were
measured, normalised to their mass. That is the same conditional reading
``predictive_mass`` already gives a clipped box, and it is reported on the same
terms. A caller that wants the strict figure asserts ``voided_mass`` is zero.

Two engines compute this quantity today. ADR-052 records why, and issue #102 tracks
cutting it back to one once the ``p*`` work settles where that code lives.
"""

from collections.abc import Callable
from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float64
from numpy.typing import ArrayLike

from cpomdp.reference.likelihood import ObservationLikelihood
from cpomdp.reference.quadrature import GridDensity, QuadratureGrid

__all__ = ["InferenceGap", "Void", "averaged_inference_gap"]


@dataclass(frozen=True)
class Void:
    """A rule's answer when a step did not converge: no belief, and why.

    Returned in place of a ``GridDensity``. The sweep records the reading as
    unmeasured rather than averaging over a number the rule could not stand behind.

    Attributes:
        iterations: how many the rule spent before it stopped.
        detail: one line on why, for the report.
    """

    iterations: int
    detail: str


# What an approximate filter does with one reading: prior in, belief out, on the
# prior's own grid, or ``Void`` where it has none to give. The rule ladder's rungs are
# the implementations of this.
ApproximatePosterior = Callable[[GridDensity, ArrayLike], GridDensity | Void]


def _boundary_mask(grid: QuadratureGrid) -> Float64[Array, "N"]:
    """Which nodes sit on the box's surface, where a truncated density piles up.

    Exact comparison is safe: the nodes come from ``linspace``, which reproduces both
    endpoints exactly, so a boundary node equals its corner value bit for bit.
    """
    nodes = grid.nodes
    return jnp.any(
        (nodes == jnp.asarray(grid.lower)) | (nodes == jnp.asarray(grid.upper)),
        axis=-1,
    )


@jax.jit
def _weigh_one_observation(
    prior: GridDensity,
    approximation: GridDensity | None,
    log_likelihood: Float64[Array, "N"],
    boundary: Float64[Array, "N"],
) -> tuple[Float64[Array, ""], Float64[Array, ""], Float64[Array, ""]]:
    """``log p*(y)``, ``KL(q ‖ p(x|y))`` and the edge ratio for one reading.

    The first two come off the same unnormalised product, so the predictive and the
    posterior cannot disagree about what the model says. ``approximation`` is
    ``None`` for a reading the rule declined. The divergence is then NaN, and the
    other two are still measured, since neither needs the rule.

    The third is the posterior's largest density on the box's surface against its
    largest anywhere. Near zero when the density has decayed before the edge. Near one
    when the mode itself is at or beyond the edge, which is the state box being too
    small for this reading and is invisible in the predictive mass.

    Compiled because the sweep calls it once per observation node and every operation
    in it is a small array op. Unjitted, the dispatch dominates: the divergence alone
    is two passes over the state grid, and the loop spends its time launching them
    rather than computing. The shapes and the lattice are constant across the sweep,
    so this traces once per case. ``None`` is a pytree with no leaves, so the declined
    case is a second trace that builds no divergence, and still one dispatch.
    """
    joint = GridDensity(prior.grid, prior.log_density + log_likelihood)
    log_density = joint.log_density
    edge = jnp.max(jnp.where(boundary, log_density, -jnp.inf)) - jnp.max(log_density)
    if approximation is None:
        divergence = jnp.asarray(jnp.nan)
    else:
        divergence = approximation.kl_to(joint)
    return joint.log_mass, divergence, jnp.exp(edge)


@dataclass(frozen=True)
class InferenceGap:
    """A measured gap, with what a reader needs to judge it.

    Attributes:
        value: the averaged gap in nats, the expectation under the measured part of
            ``p*``: the readings inside the box that the rule answered. Normalised
            to their mass, so it is a conditional expectation rather than an
            undercount. NaN when the rule answered none of them.
        predictive_mass: what ``p*(y)`` integrated to over the declared observation
            box. One when the box caught everything. Below one when it did not, and
            then ``value`` speaks for a conditional the caller did not ask for.
        voided_mass: the share of ``predictive_mass`` at the readings the rule
            declined. Zero for a rule that always answers. Anything else is weight
            ``value`` does not cover, on the same terms as a clipped box.
        voided_nodes: which observation nodes the rule declined, in the grid's node
            order. ``divergences`` is NaN there.
        worst_edge_ratio: across the sweep, the largest share any exact posterior put
            at the state box's surface, as its boundary density against its peak.
            Near zero on a state box wide enough for every reading. Approaching one
            means some reading pushed the posterior to the edge, where the joint
            integrates to less than ``p*(y)`` and this ``value`` is built on a
            corrupted integrand. ``predictive_mass`` does not see it, since the
            readings that cause it are the ones carrying least weight.
        divergences: ``KL(q ‖ p(x|y))`` at each observation node, in the grid's node
            order. Where the gap concentrates in ``y`` is a property of the gap, not
            a by-product, and it is what a tail-dominated result looks like.
        observation_grid: the grid the average was taken on, carrying the box and the
            resolution that produced these numbers.
    """

    value: float
    predictive_mass: float
    voided_mass: float
    voided_nodes: Bool[Array, "K"]
    worst_edge_ratio: float
    divergences: Float64[Array, "K"]
    observation_grid: QuadratureGrid


def averaged_inference_gap(
    prior: GridDensity,
    likelihood: ObservationLikelihood,
    approximate_posterior: ApproximatePosterior,
    observation_grid: QuadratureGrid,
) -> InferenceGap:
    """Measure ``E_{y∼p*}[ KL(q ‖ p(x|y)) ]`` over a declared observation box.

    One pass per observation node. Each builds the unnormalised product
    ``p(x)·p(y|x)`` once and reads both things off it: what it integrates to is
    ``p*(y)``, and normalising it gives the exact posterior. The predictive and the
    posterior therefore come from the same object and cannot disagree.

    Costs ``K`` likelihood evaluations over ``N`` nodes, which is linear in both. It
    is cheap beside a single prediction, which is quadratic in ``N``.

    Args:
        prior: the true state density before the reading. Normalised internally, so
            ``p*`` is a density rather than a scaled one.
        likelihood: the true observation likelihood. The same one produces ``p*`` and
            the exact posterior, since the gap is about the approximation only.
        approximate_posterior: the rule under test, called as
            ``rule(prior, observation)`` and returning its belief on the prior's
            grid, or ``Void`` for a reading it could not converge on.
        observation_grid: the box and resolution the average is taken over. Its
            dimension is the observation's, not the state's.

    Returns:
        The gap and its diagnostics.

    Raises:
        ValueError: if a rule returns a belief on a lattice other than the prior's.
            Two densities on different lattices have no divergence between them.
    """
    prior = prior.normalise()
    states = prior.grid.nodes
    boundary = _boundary_mask(prior.grid)

    log_predictive = []
    divergences = []
    voided = []
    edge_ratios = []
    for observation in observation_grid.nodes:
        belief = approximate_posterior(prior, observation)
        approximation = None if isinstance(belief, Void) else belief
        if approximation is not None and not approximation.grid.same_lattice_as(
            prior.grid
        ):
            raise ValueError(
                "approximate_posterior must return a belief on the prior's lattice, "
                f"got {approximation.grid!r} against {prior.grid!r}"
            )
        log_mass, divergence, edge_ratio = _weigh_one_observation(
            prior,
            approximation,
            likelihood.log_likelihood(observation, states),
            boundary,
        )
        log_predictive.append(log_mass)
        divergences.append(divergence)
        voided.append(approximation is None)
        edge_ratios.append(edge_ratio)

    voided_nodes = jnp.asarray(voided)
    predictive = GridDensity(observation_grid, jnp.stack(log_predictive))
    # A declined node leaves the average by weight, not by index, so the normaliser
    # is the measured mass and the NaN it carries never touches the sum.
    measured = GridDensity(
        observation_grid, jnp.where(voided_nodes, -jnp.inf, predictive.log_density)
    )
    declined = GridDensity(
        observation_grid, jnp.where(voided_nodes, predictive.log_density, -jnp.inf)
    )
    stacked = jnp.stack(divergences)
    return InferenceGap(
        value=float(measured.expectation(jnp.where(voided_nodes, 0.0, stacked))),
        predictive_mass=float(jnp.exp(predictive.log_mass)),
        voided_mass=float(jnp.exp(declined.log_mass)),
        voided_nodes=voided_nodes,
        worst_edge_ratio=float(jnp.max(jnp.stack(edge_ratios))),
        divergences=stacked,
        observation_grid=observation_grid,
    )
