"""Exact Bayesian filtering on a quadrature grid.

Bayes as it is actually written: move the belief forward through the dynamics, then
multiply by the likelihood and divide by what that integrates to. On a grid every
step is arithmetic over node values, and the normalising integral that has no closed
form in general is a weighted sum.

The intractability of exact Bayes is a statement about dimension. A tensor grid costs
``k^d`` nodes, which is hopeless by ``d = 10`` and free at ``d = 1``. This module is
an instrument rather than a deployable filter: it runs on a low-dimensional latent to
measure what a Gaussian filter gets wrong on the same model, and it is never on an
agent's per-cycle path.

The two steps do not cost the same. ``condition`` touches each node once. ``predict``
contracts over an ``N x N`` matrix of pairwise transition densities, so it is
quadratic in the node count and it is where the dimension limit actually bites: at
``N = 10⁴`` that matrix alone is 800 MB. The matrix is an argument rather than
something built inside the prediction, so a trajectory reusing one kernel builds it
once and the cost sits where a reader can see it. If it ever needs to stop being
materialised, the chunked enumerator's pattern applies — accumulate over blocks of
destination rows and keep residency at ``O(chunk · N)``.

One grid carries the whole trajectory. Predicting onto a second grid would be a
different object with a different truncation, and nothing here needs one.
"""

from collections.abc import Sequence

import jax.numpy as jnp
from jax.scipy.special import logsumexp
from numpy.typing import ArrayLike

from cpomdp._lab.reference.likelihood import ObservationLikelihood
from cpomdp._lab.reference.quadrature import GridDensity
from cpomdp._lab.reference.transition import TransitionKernel

__all__ = ["condition", "filter_sequence", "predict"]


def condition(
    prior: GridDensity,
    likelihood: ObservationLikelihood,
    observation: ArrayLike,
) -> GridDensity:
    """The measurement update: ``p(x | y) ∝ p(x) · p(y | x)``, normalised.

    Exact up to the grid. The only error is the quadrature's, which is discretisation
    and truncation, both of which shrink under refinement. No step of this assumes
    the posterior has any particular shape, which is the difference from a Gaussian
    filter, whose error under a state-dependent ``R(x)`` is structural and shrinks
    under nothing.

    The result is normalised, since an unnormalised posterior fed back in as the next
    step's prior drifts out of floating-point range within a few steps. Nothing is
    lost by it: ``posterior.log_normaliser - prior.log_normaliser`` is the log of
    what the product integrated to, which is the log evidence ``log p(y)`` whenever
    the prior itself integrates to one.

    Args:
        prior: the belief before the observation, on the grid the posterior will use.
        likelihood: evaluated once at every node of ``prior.grid``.
        observation: the reading to condition on.

    Returns:
        The posterior on the same grid, integrating to one.
    """
    log_likelihood = likelihood.log_likelihood(observation, prior.grid.nodes)
    return GridDensity(
        prior.grid,
        prior.log_density + log_likelihood,
        prior.log_normaliser,
    ).normalise()


def predict(prior: GridDensity, log_transition: ArrayLike) -> GridDensity:
    """The time update: ``p(x') = ∫ p(x' | x) p(x) dx``, as a quadrature.

    The integral is a contraction of the transition matrix against the belief and the
    quadrature weights, done in logs so a tail many orders below the mode still
    contributes what it should.

    Unnormalised on purpose. Exact prediction moves mass without creating or
    destroying it, so whatever the result integrates to below one is the box being
    too small for the density after the dynamics have spread it. Normalising here
    would divide that signal away, and prediction is where it first appears, since
    the transition widens the support the prior was sized for.

    Args:
        prior: the belief before the step.
        log_transition: ``log p(x' | x, u)`` between every pair of the prior's nodes,
            indexed ``[destination, origin]``. Square, because one grid carries the
            whole trajectory.

    Returns:
        The predicted belief on the same grid, carrying the prior's normaliser
        forward unchanged.

    Raises:
        ValueError: if the matrix is not ``N x N`` for the prior's node count.
    """
    log_transition = jnp.asarray(log_transition, dtype=float)
    size = prior.grid.size
    if log_transition.shape != (size, size):
        raise ValueError(
            f"log_transition must be {size}x{size} to match the grid, indexed "
            f"[destination, origin], got shape {log_transition.shape}"
        )
    log_mass_at_origin = jnp.log(prior.grid.weights) + prior.log_density
    return GridDensity(
        prior.grid,
        logsumexp(log_transition + log_mass_at_origin[None, :], axis=1),
        prior.log_normaliser,
    )


def filter_sequence(
    prior: GridDensity,
    kernel: TransitionKernel,
    likelihood: ObservationLikelihood,
    observations: Sequence[ArrayLike],
    actions: Sequence[ArrayLike] | None = None,
) -> tuple[GridDensity, ...]:
    """Run predict-then-condition over a sequence, returning every posterior.

    The transition matrix is built once and reused when there are no actions, since
    a control-free kernel gives the same matrix at every step and it is the most
    expensive object in the run. With actions it is rebuilt per step, because the
    action shifts every destination mean.

    Args:
        prior: the belief before the first step.
        kernel: supplies ``log p(x' | x, u)`` over the prior's grid.
        likelihood: supplies ``log p(y | x)`` at the prior's nodes.
        observations: one reading per step.
        actions: one action per step, or ``None`` for a control-free kernel.

    Returns:
        One posterior per observation, in order. Each is normalised, and each carries
        the accumulated ``log_normaliser`` so the run's log evidence is the last
        one's minus the prior's.

    Raises:
        ValueError: if ``actions`` is given and does not have one entry per
            observation.
    """
    if actions is not None and len(actions) != len(observations):
        raise ValueError(
            f"actions must have one entry per observation, got {len(actions)} "
            f"actions and {len(observations)} observations"
        )

    nodes = prior.grid.nodes
    if actions is None:
        shared = kernel.log_transition(nodes, nodes)
        transitions = (shared for _ in observations)
    else:
        # Lazy, so a driven run holds one matrix at a time rather than one per step.
        transitions = (
            kernel.log_transition(nodes, nodes, action) for action in actions
        )

    posteriors = []
    belief = prior
    for observation, log_transition in zip(observations, transitions, strict=True):
        belief = condition(predict(belief, log_transition), likelihood, observation)
        posteriors.append(belief)
    return tuple(posteriors)
