"""The rule ladder: declared ways of approximating the posterior, one seam each.

A rung is a factory. It takes the likelihood a reading is conditioned on and returns
an ``ApproximatePosterior``, the callable ``averaged_inference_gap`` measures. What a
rung reads from the model is closed over at construction, so the seam passes only
``(prior, observation)``. The per-reading cost is then the update itself plus the
render onto the prior's lattice, and every rung pays the render alike.

ADR-056 declares five rungs. Three are here: the plug-in rung, the Kalman update with
the noise read once at the prior mean; the belief-smoothed rung, the same update with
the noise averaged under the prior; and the exact reference at the top. The two
Spinello–Stilwell rungs arrive behind the same ``Rung`` type as they are built, and
the versioned ladder is declared once all five exist.

The Kalman algebra in the plug-in rung is written here and not imported from
``cpomdp.backends.kalman``. The reference is evidence about that filter, and a rung
built out of it would agree with it for reasons that are not evidence.
``tests/test_module_boundary.py`` keeps that true.
"""

from dataclasses import dataclass
from enum import Enum

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float64
from numpy.typing import ArrayLike

from cpomdp._validation import validate_declared
from cpomdp.reference.filtering import condition
from cpomdp.reference.gap import ApproximatePosterior
from cpomdp.reference.likelihood import GaussianChannel, ObservationLikelihood
from cpomdp.reference.quadrature import GridDensity, gaussian_on

__all__ = ["EXACT_RUNG", "RuleLadder", "Rung", "RungKind"]


class RungKind(Enum):
    """What a rung does with one reading.

    ``PLUG_IN`` is the Kalman update with the observation noise read once, at the
    prior mean. It is the first rung of ADR-056's ladder and what a Gaussian filter
    does under a state-dependent sensor. ``BELIEF_SMOOTHED`` is the same update with
    the noise averaged under the prior, the ``E[R(x)]`` rule of the paper's Remark 3.
    ``EXACT`` conditions on the grid and is the top, the reference every other rung's
    gap is measured against.
    """

    PLUG_IN = "PlugIn"
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
                rung its matrix and its noise at every node of the prior. The exact
                rung reads its density and nothing else.

        Returns:
            ``rule(prior, observation)``, the callable ``averaged_inference_gap``
            takes as the rule under test.

        Raises:
            ValueError: if the kind names no way of building a rule.
        """
        if self.kind is RungKind.PLUG_IN:
            return _plug_in(channel)
        if self.kind is RungKind.BELIEF_SMOOTHED:
            return _belief_smoothed(channel)
        if self.kind is RungKind.EXACT:
            return _exact(channel)
        # Not a fallthrough: a kind added without a branch here would otherwise be
        # built as whichever branch came last, and report a rung it never ran.
        raise ValueError(f"{self.kind} names no way of building a rule")


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
