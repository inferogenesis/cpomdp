"""The rule ladder: declared ways of approximating the posterior, one seam each.

A rung is a factory. It takes the likelihood a reading is conditioned on and returns
an ``ApproximatePosterior``, the callable ``averaged_inference_gap`` measures. What a
rung reads from the model is closed over at construction, so the seam passes only
``(prior, observation)``. The per-reading cost is then the update itself plus the
render onto the prior's lattice, and every rung pays the render alike.

ADR-056 declares five rungs, all here. The plug-in rung is the Kalman update with the
noise read once at the prior mean. The single-step and iterated rungs are the
Spinello–Stilwell scheme with the modification of ADR-057, at a budget of one and at
the budget of ADR-058. The belief-smoothed rung is the Kalman update with the noise
averaged under the prior. The exact reference is the top. The versioned ladder is
declared once all five exist.

The Kalman algebra in the plug-in rung is written here and not imported from
``cpomdp.backends.kalman``. The reference is evidence about that filter, and a rung
built out of it would agree with it for reasons that are not evidence.
``tests/test_module_boundary.py`` keeps that true.
"""

import functools
from dataclasses import dataclass
from enum import Enum

import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float64, Int
from numpy.typing import ArrayLike

from cpomdp._validation import validate_declared
from cpomdp.reference.filtering import condition
from cpomdp.reference.gap import ApproximatePosterior, Void
from cpomdp.reference.likelihood import GaussianChannel, ObservationLikelihood
from cpomdp.reference.quadrature import GridDensity, gaussian_on

__all__ = [
    "BELIEF_SMOOTHED_RUNG",
    "CONVERGENCE_TOLERANCE",
    "EXACT_RUNG",
    "ITERATED_RUNG",
    "ITERATION_BUDGET",
    "LADDER",
    "PLUG_IN_RUNG",
    "SINGLE_STEP_RUNG",
    "IteratedUpdate",
    "RuleLadder",
    "Rung",
    "RungKind",
    "iterated_update",
]

ITERATION_BUDGET = 64
"""Steps the iterated rung may spend on one reading before it answers ``Void``.

ADR-058, sized off the counts the modified scheme needs in the bulk of the predictive
mass. A convergent run stops at its tolerance, so the budget is a cap and not a cost.
"""

CONVERGENCE_TOLERANCE = 1e-12
"""A step shorter than this, in prior standard deviations, ends the iteration.

ADR-058. Only reachable with an exact derivative of the noise, which is why the rungs
differentiate the channel and never difference it.
"""


class RungKind(Enum):
    """What a rung does with one reading.

    In ladder order. ``PLUG_IN`` is the Kalman update with the observation noise read
    once, at the prior mean. It is the first rung of ADR-056's ladder and what a
    Gaussian filter does under a state-dependent sensor. ``SINGLE_STEP`` is one
    modified-metric Newton step on the posterior's objective from the prior mean, and
    ``ITERATED`` runs that step to the declared tolerance within the declared budget:
    the Spinello–Stilwell scheme with the modification of ADR-057. ``BELIEF_SMOOTHED``
    is the Kalman update with the noise averaged under the prior, the ``E[R(x)]`` rule
    of the paper's Remark 3. ``EXACT`` conditions on the grid and is the top, the
    reference every other rung's gap is measured against.
    """

    PLUG_IN = "PlugIn"
    SINGLE_STEP = "SingleStep"
    ITERATED = "Iterated"
    BELIEF_SMOOTHED = "BeliefSmoothed"
    EXACT = "Exact"


@jax.jit
def _kalman_update(
    observation_matrix: Float64[Array, "m n"],
    observation_noise: Float64[Array, "m m"],
    prior_mean: Float64[Array, "n"],
    prior_cov: Float64[Array, "n n"],
    observation: Float64[Array, "m"],
) -> tuple[Float64[Array, "n"], Float64[Array, "n n"]]:
    """One measurement update, ``(mean, cov)`` from the prior's and the reading.

    The gain is a solve against the innovation covariance rather than an inverse.
    """
    innovation_cov = (
        observation_matrix @ prior_cov @ observation_matrix.T + observation_noise
    )  # S
    gain = jnp.linalg.solve(innovation_cov, observation_matrix @ prior_cov).T  # K
    mean = prior_mean + gain @ (observation - observation_matrix @ prior_mean)
    cov = (jnp.eye(prior_mean.shape[0]) - gain @ observation_matrix) @ prior_cov
    return mean, cov


def _as_observation(observation: ArrayLike, length: int) -> Float64[Array, "m"]:
    """The reading as a vector of the channel's length, checked once.

    Closes the same silent-broadcast trap the likelihood closes: a reading of the
    wrong length against the update's algebra would broadcast into a confident
    belief about the wrong thing.
    """
    observation = jnp.asarray(observation, dtype=float)
    if observation.shape != (length,):
        raise ValueError(
            f"observation must be a 1-D vector of length {length} (the observation "
            f"dimension), got shape {observation.shape}"
        )
    return observation


def _plug_in(channel: GaussianChannel) -> ApproximatePosterior:
    """Rung one: ``R(μ⁻)`` plugged into the Kalman update."""
    observation_matrix = channel.observation_matrix

    def rule(prior: GridDensity, observation: ArrayLike) -> GridDensity:
        prior_mean, prior_cov = prior.moments
        noise = channel.observation_noise_at(prior_mean[None, :])[0]
        mean, cov = _kalman_update(
            observation_matrix,
            noise,
            prior_mean,
            prior_cov,
            _as_observation(observation, observation_matrix.shape[0]),
        )
        return gaussian_on(prior.grid, mean, cov)

    return rule


def _belief_smoothed(channel: GaussianChannel) -> ApproximatePosterior:
    """Rung four: ``E[R(x)]`` under the prior, plugged into the Kalman update.

    The average is taken under the density the rung is handed, on its own grid and
    with its own weights, and not under the Gaussian with that density's moments.
    The two agree when the prior is Gaussian, which is the paper's case, and only the
    first stays defined when it is not. Per reading this rung evaluates the noise at
    every node of the prior where the plug-in rung evaluates it once.
    """
    observation_matrix = channel.observation_matrix

    def rule(prior: GridDensity, observation: ArrayLike) -> GridDensity:
        prior_mean, prior_cov = prior.moments
        noise = prior.expectation(channel.observation_noise_at(prior.grid.nodes))
        mean, cov = _kalman_update(
            observation_matrix,
            noise,
            prior_mean,
            prior_cov,
            _as_observation(observation, observation_matrix.shape[0]),
        )
        return gaussian_on(prior.grid, mean, cov)

    return rule


@dataclass(frozen=True)
class IteratedUpdate:
    """What one run of the modified-metric Newton scheme produced, and what it cost.

    Attributes:
        mean: the last iterate.
        cov: the inverse of the prior precision plus the Fisher information at
            ``mean``, equation (35e) of the paper.
        iterations: steps taken.
        converged: whether the last step fell under the tolerance. Reported alongside
            the count so a run that stopped on its budget is not read as settled.
    """

    mean: Float64[Array, "n"]
    cov: Float64[Array, "n n"]
    iterations: int
    converged: bool


def _single_channel(channel: GaussianChannel) -> Float64[Array, "n"]:
    """The one row of the observation matrix, or a refusal.

    The scheme here is the paper's scalar-observation case, with a vector state. Its
    form for several channels is not written in this tree, so a channel with more
    than one is refused rather than run through algebra that does not cover it.
    """
    observation_matrix = channel.observation_matrix
    if observation_matrix.shape[0] != 1:
        raise ValueError(
            "the single-step and iterated rungs are written for one observation "
            f"channel, got {observation_matrix.shape[0]}"
        )
    return observation_matrix[0]


def _newton_terms(
    channel: GaussianChannel,
    mean_slope: Float64[Array, "n"],
    state: Float64[Array, "n"],
    observation: Float64[Array, ""],
) -> tuple[Float64[Array, "n"], Float64[Array, "n n"], Float64[Array, "n n"]]:
    """The measurement's score, its modified curvature, and its Fisher information.

    Equations (35c), (35d) without its fourth term, and (35e) of the paper, at one
    state. The noise and its derivative come from one forward-mode pass through the
    channel, so the derivative is exact.
    """

    def noise_at(x: Float64[Array, "n"]) -> Float64[Array, ""]:
        return channel.observation_noise_at(x[None, :])[0, 0, 0]

    noise, noise_slope = jax.value_and_grad(noise_at)(state)
    residual = observation - mean_slope @ state
    score = (
        -(residual / noise) * mean_slope
        + ((1.0 - residual**2 / noise) / (2.0 * noise)) * noise_slope
    )
    steered = mean_slope + (residual / (2.0 * noise)) * noise_slope  # b
    curvature = jnp.outer(steered, steered) / noise
    fisher = jnp.outer(mean_slope, mean_slope) / noise + jnp.outer(
        noise_slope, noise_slope
    ) / (2.0 * noise**2)
    return score, curvature, fisher


@functools.partial(jax.jit, static_argnames=("budget",))
def _iterate(
    channel: GaussianChannel,
    prior_mean: Float64[Array, "n"],
    prior_cov: Float64[Array, "n n"],
    observation: Float64[Array, "1"],
    budget: int,
    tolerance: Float64[Array, ""],
) -> tuple[Float64[Array, "n"], Float64[Array, "n n"], Int[Array, ""], Bool[Array, ""]]:
    """The scheme's loop, compiled once per channel.

    Each step solves the prior precision plus the modified curvature against the
    full gradient. The step's length is measured in the prior's metric, which for one
    state dimension is the step over the prior standard deviation.
    """
    mean_slope = channel.observation_matrix[0]
    prior_precision = jnp.linalg.inv(prior_cov)
    reading = observation[0]

    def step(carry):
        state, _, taken = carry
        score, curvature, _ = _newton_terms(channel, mean_slope, state, reading)
        gradient = prior_precision @ (state - prior_mean) + score
        step = jnp.linalg.solve(prior_precision + curvature, gradient)
        state = state - step
        return state, jnp.sqrt(step @ prior_precision @ step), taken + 1

    def unfinished(carry):
        _, length, taken = carry
        return (taken < budget) & (length >= tolerance)

    state, length, taken = jax.lax.while_loop(
        unfinished,
        step,
        (prior_mean, jnp.asarray(jnp.inf, dtype=float), jnp.asarray(0)),
    )
    _, _, fisher = _newton_terms(channel, mean_slope, state, reading)
    return state, jnp.linalg.inv(prior_precision + fisher), taken, length < tolerance


def iterated_update(
    channel: GaussianChannel,
    prior_mean: ArrayLike,
    prior_cov: ArrayLike,
    observation: ArrayLike,
    *,
    budget: int,
    tolerance: float,
) -> IteratedUpdate:
    """The modified-metric Newton scheme on one reading, with its cost on the result.

    Starts at the prior mean and stops when a step is shorter than ``tolerance``
    prior standard deviations or when ``budget`` steps have been taken, whichever is
    first. The rungs call this with the declared constants; it takes them as
    arguments so the count a budget has to cover can be measured at some other one.

    Args:
        channel: a one-channel likelihood, differentiated for the noise's slope.
        prior_mean: the predicted mean.
        prior_cov: the predicted covariance.
        observation: the reading, a vector of length one.
        budget: most steps to take.
        tolerance: the step length, in prior standard deviations, that ends the run.

    Returns:
        The last iterate, the covariance at it, and how the run ended.

    Raises:
        ValueError: if the channel has more than one observation dimension, or the
            reading is not a vector of length one.
    """
    _single_channel(channel)
    mean, cov, taken, converged = _iterate(
        channel,
        jnp.asarray(prior_mean, dtype=float),
        jnp.asarray(prior_cov, dtype=float),
        _as_observation(observation, 1),
        budget,
        jnp.asarray(tolerance, dtype=float),
    )
    return IteratedUpdate(mean, cov, int(taken), bool(converged))


def _single_step(channel: GaussianChannel) -> ApproximatePosterior:
    """Rung two: one modified-metric Newton step from the prior mean.

    The paper's single-step filter (36) under ADR-057's modification. One step is
    the rung's definition, so it never answers ``Void``.
    """
    _single_channel(channel)

    def rule(prior: GridDensity, observation: ArrayLike) -> GridDensity:
        prior_mean, prior_cov = prior.moments
        update = iterated_update(
            channel, prior_mean, prior_cov, observation, budget=1, tolerance=0.0
        )
        return gaussian_on(prior.grid, update.mean, update.cov)

    return rule


def _iterated(channel: GaussianChannel) -> ApproximatePosterior:
    """Rung three: the scheme run to the declared tolerance within the declared budget.

    The paper's iterated filter (35) under ADR-057's modification. A run that spends
    the budget answers ``Void`` with the count, per ADR-056: the covariance is read
    at the last iterate, so a truncated run is wrong in both moments.
    """
    _single_channel(channel)

    def rule(prior: GridDensity, observation: ArrayLike) -> GridDensity | Void:
        prior_mean, prior_cov = prior.moments
        update = iterated_update(
            channel,
            prior_mean,
            prior_cov,
            observation,
            budget=ITERATION_BUDGET,
            tolerance=CONVERGENCE_TOLERANCE,
        )
        if not update.converged:
            return Void(
                iterations=update.iterations,
                detail=f"budget of {ITERATION_BUDGET} spent above the tolerance",
            )
        return gaussian_on(prior.grid, update.mean, update.cov)

    return rule


def _exact(likelihood: ObservationLikelihood) -> ApproximatePosterior:
    """The top rung: the exact posterior on the prior's grid."""

    def rule(prior: GridDensity, observation: ArrayLike) -> GridDensity:
        return condition(prior, likelihood, observation)

    return rule


@dataclass(frozen=True)
class Rung:
    """A named rung, built into an ``ApproximatePosterior`` over a channel on demand.

    Args:
        name: the rung's label, non-empty and unique within a ladder.
        kind: which of the declared kinds this rung is.
    """

    name: str
    kind: RungKind

    def __post_init__(self) -> None:
        """Reject a nameless rung."""
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("a Rung needs a name, so a row of the ladder can be read")

    def build(self, channel: GaussianChannel) -> ApproximatePosterior:
        """The rule this rung runs over ``channel``.

        Args:
            channel: the likelihood the reading is conditioned on. The plug-in rung
                reads its matrix and its noise at the prior mean, the belief-smoothed
                rung its matrix and its noise at every node of the prior. The
                single-step and iterated rungs read its matrix and differentiate its
                noise at each iterate, and take one observation channel only. The
                exact rung reads its density and nothing else.

        Returns:
            ``rule(prior, observation)``, the callable ``averaged_inference_gap``
            takes as the rule under test.

        Raises:
            ValueError: if the kind names no way of building a rule.
        """
        if self.kind is RungKind.PLUG_IN:
            return _plug_in(channel)
        if self.kind is RungKind.SINGLE_STEP:
            return _single_step(channel)
        if self.kind is RungKind.ITERATED:
            return _iterated(channel)
        if self.kind is RungKind.BELIEF_SMOOTHED:
            return _belief_smoothed(channel)
        if self.kind is RungKind.EXACT:
            return _exact(channel)
        # Not a fallthrough: a kind added without a branch here would otherwise be
        # built as whichever branch came last, and report a rung it never ran.
        raise ValueError(f"{self.kind} names no way of building a rule")


PLUG_IN_RUNG = Rung(name="plug-in", kind=RungKind.PLUG_IN)
"""The first rung, ``R`` read once at the prior mean."""

SINGLE_STEP_RUNG = Rung(name="modified-single-step", kind=RungKind.SINGLE_STEP)
"""The second rung, one step of the modified scheme. Named as modified (ADR-057)."""

ITERATED_RUNG = Rung(name="modified-iterated", kind=RungKind.ITERATED)
"""The third rung, the modified scheme run to tolerance. Named as modified (ADR-057)."""

BELIEF_SMOOTHED_RUNG = Rung(name="belief-smoothed", kind=RungKind.BELIEF_SMOOTHED)
"""The fourth rung, ``R`` averaged under the prior (ADR-061)."""

EXACT_RUNG = Rung(name="exact", kind=RungKind.EXACT)
"""The top of the ladder, against which every other rung's gap is measured."""


@dataclass(frozen=True)
class RuleLadder:
    """A declared, versioned set of rungs, to run one channel through.

    Names are unique. At least one rung is exact, so the ladder always has its top.

    Args:
        rungs: the declared members, in the order they are reported.
        version: a non-empty tag, so a rung added later shows up in the diff.
    """

    rungs: tuple[Rung, ...]
    version: str

    def __post_init__(self) -> None:
        """Reject an unversioned, empty, duplicated or topless ladder."""
        validate_declared(
            self.rungs,
            self.version,
            subject="ladder",
            contains=lambda rung: rung.kind is RungKind.EXACT,
            requirement=(
                "no rung of this ladder is the exact reference, so nothing is the "
                "top the others are measured against; include EXACT_RUNG"
            ),
        )

    @property
    def names(self) -> tuple[str, ...]:
        """The declared names, in declaration order."""
        return tuple(rung.name for rung in self.rungs)

    @property
    def size(self) -> int:
        """The rung count."""
        return len(self.rungs)

    def build_all(
        self, channel: GaussianChannel
    ) -> tuple[tuple[str, ApproximatePosterior], ...]:
        """Every rung built over ``channel``, paired with the name that produced it.

        Args:
            channel: the likelihood each rung conditions on.

        Returns:
            One ``(name, rule)`` pair per rung, in declaration order.
        """
        return tuple((rung.name, rung.build(channel)) for rung in self.rungs)


LADDER = RuleLadder(
    rungs=(
        PLUG_IN_RUNG,
        SINGLE_STEP_RUNG,
        ITERATED_RUNG,
        BELIEF_SMOOTHED_RUNG,
        EXACT_RUNG,
    ),
    version="v1",
)
"""The five rungs of ADR-056, in the order the battery's D1 leg is registered over.

Declared once every rung existed, so no version names a ladder with a rung missing.
A rung added after an ordering is seen changes this constant and its version.
"""
