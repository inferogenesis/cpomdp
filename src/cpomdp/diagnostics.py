"""Checkable conditions on a model's sensor, over the states an action can reach.

Whether a state-dependent ``R(x)`` actually buys an agent anything is not a property of
the noise function alone — it depends on where the control can put the predicted mean.
A noise that varies only in a corner no policy reaches is, for every purpose the filter
and the objective have, a constant. The conditions here are the ones that separate the
two cases, and each is reported against a *sampled reachable set*: the predicted means
of a set of candidate actions from a shared belief.

Four conditions, in the order they bite:

``observation_matrix_full_row_rank``
    ``C`` carries no redundant channels, so ``C Cᵀ`` is invertible. Matching a
    posterior covariance then identifies the noise uniquely; without it, only the
    noise's action on the row space of ``C`` is identified, and covariances off that
    subspace are free.

``observation_noise_definite``
    ``R(x) ≻ 0`` at every sampled mean. The epistemic term inverts ``R`` and takes its
    log-determinant, so anywhere this fails the term is not defined — the kernel returns
    NaN there rather than a plausible-looking number.

``observation_noise_varies``
    ``R`` takes different values across the sampled means. This is what makes the
    posterior covariance, and so the Kalman gain, depend on the action.

``epistemic_varies``
    The per-step epistemic value differs across those means. Stronger than
    ``observation_noise_varies``: the epistemic term is a log-determinant, a single
    scalar summary, and a covariance can move along a contour of it without the scalar
    noticing. With a fixed ``Q`` and a scalar observation the two coincide, since the
    map is then strictly monotone in the variance. ``observation_noise_loewner_ordered``
    reports the cheap sufficient condition — noise covariances ordered by ``≺`` always
    separate the determinant.

A state-dependent ``Q(x)`` moves the predicted covariance with the action too, so the
epistemic value can vary with ``R`` held still. ``dynamics_noise_varies`` reports it,
and ``flattens`` needs neither noise to move. A ``Q(x)`` that is not a covariance at a
sampled mean raises instead of being reported. That is a broken model, not a condition
on the sensor.

Sampling is what it is: these answer "over the actions I asked about", not "over every
reachable state". A negative is evidence, not proof.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, cast

import jax
import numpy as np
from jaxtyping import Float64
from numpy.typing import ArrayLike

from cpomdp._validation import validate_covariance

if TYPE_CHECKING:
    # Annotation-only: every diagnostic here reads its argument's fields by duck typing,
    # so diagnostics depends on none of these at runtime (they already depend on nothing
    # here). Keeps the dependency one-way.
    from cpomdp.efe import PolicyEfeTrace
    from cpomdp.types import Belief, LinearGaussianModel


class ProbeBackend(Protocol):
    """The three members [`probe_model`][cpomdp.probe_model] reads off a graph backend.

    Structural rather than a name, because three members is what the function requires.
    Any backend growing them works, and the diagnostic keeps its one-way dependency on
    the backends package. [`CouplingGraphBackend`][cpomdp.CouplingGraphBackend]
    satisfies it.
    """

    @property
    def observation_model(self) -> tuple[jax.Array, jax.Array]:
        """``(C, R)`` over the joint state."""
        ...

    def predicted_belief(
        self, prior: "Belief", action: ArrayLike | None = None
    ) -> "Belief":
        """The belief one prediction step ahead under ``action``."""
        ...

    def observation_noise_at(self, mean: ArrayLike) -> Float64[jax.Array, "m m"]:
        """The stacked ``R`` linearized at a predicted mean."""
        ...


__all__ = [
    "ProbeBackend",
    "ProbeReport",
    "RolloutConditioning",
    "epistemic_value",
    "is_positive_definite",
    "loewner_order",
    "probe_model",
    "rollout_conditioning",
]


def is_positive_definite(matrix: ArrayLike) -> bool:
    """Whether ``matrix`` is symmetric positive definite.

    A Cholesky factorisation, which succeeds exactly on the positive-definite
    matrices — rather than a determinant sign, which passes anything with an even
    number of negative eigenvalues.
    """
    m = np.asarray(matrix, dtype=float)
    if m.ndim != 2 or m.shape[0] != m.shape[1]:
        return False
    if not np.allclose(m, m.T, atol=1e-12):
        return False
    try:
        np.linalg.cholesky(m)
    except np.linalg.LinAlgError:
        return False
    return True


def _host_logdet_pd(matrix: ArrayLike) -> float:
    """``ln det`` of a symmetric positive-definite matrix, NaN otherwise.

    ``slogdet`` splits its answer into a sign and a log-magnitude. Read the magnitude
    alone and ``diag(-1, -2)`` comes back as ``ln 2``. That matrix has determinant +2
    and two negative eigenvalues, so its log-determinant does not exist. Reading the
    sign catches that case but not the general one: any even count of negative
    eigenvalues leaves the sign positive. So the argument goes through
    ``is_positive_definite``, a Cholesky, and the magnitude is read only if it succeeds.

    Host counterpart to the objective's ``cpomdp.efe._logdet_pd``. The two are separate
    implementations on purpose. The kernel reads its log-determinant off a Cholesky
    factor under ``jit``. This one calls ``slogdet`` behind a Cholesky guard. Anything
    checking one against the other needs them to stay distinct.

    Args:
        matrix: the matrix to take a log-determinant of.

    Returns:
        ``ln det matrix`` in nats, or NaN if it is not symmetric positive definite.
    """
    if not is_positive_definite(matrix):
        return float("nan")
    return float(np.linalg.slogdet(np.asarray(matrix, dtype=float))[1])


def epistemic_value(
    predicted_cov: ArrayLike,
    observation_matrix: ArrayLike,
    observation_noise: ArrayLike,
) -> float:
    """The per-step epistemic value ``½(ln det S − ln det R)``, in nats.

    The mutual information between state and observation under the predictive joint,
    with ``S = C Σ⁻ Cᵀ + R`` the innovation covariance. Computed here on the host for
    reporting; the objective's own copy lives in ``cpomdp.efe``.

    Both log-determinants go through ``_host_logdet_pd``, so a non-positive-definite
    ``R`` or ``S`` returns NaN and an undefined value stays visibly undefined.
    """
    cov = np.asarray(predicted_cov, dtype=float)
    c = np.asarray(observation_matrix, dtype=float)
    r = np.asarray(observation_noise, dtype=float)
    s = c @ cov @ c.T + r
    return 0.5 * (_host_logdet_pd(s) - _host_logdet_pd(r))


def loewner_order(a: ArrayLike, b: ArrayLike) -> str:
    """Compare two covariances in the Loewner order.

    Returns ``"a<b"`` when ``b − a`` is positive definite, ``"b<a"`` for the reverse,
    ``"equal"`` when they agree, and ``"incomparable"`` when neither difference is
    definite. A strictly comparable pair is enough to separate the epistemic value, so
    only incomparable pairs need the determinant consulted.
    """
    x = np.asarray(a, dtype=float)
    y = np.asarray(b, dtype=float)
    if np.allclose(x, y, atol=1e-12):
        return "equal"
    if is_positive_definite(y - x):
        return "a<b"
    if is_positive_definite(x - y):
        return "b<a"
    return "incomparable"


@dataclass(frozen=True)
class ProbeReport:
    """What [`probe_model`][cpomdp.probe_model] found over the sampled reachable set.

    Each field is named for the object it describes: the observation matrix ``C``, the
    observation noise ``R``, the dynamics noise ``Q`` or the epistemic value.

    Attributes:
        n_samples: how many predicted means were probed.
        n_observations: the number of observation channels, ``C``'s row count.
        observation_matrix_rank: the rank of ``C``.
        observation_matrix_full_row_rank: whether ``C`` has no redundant channels.
        observation_noise_definite: whether ``R`` was positive definite at every
            sampled mean.
        observation_noise_indefinite_at: the sampled means where it was not.
        observation_noise_varies: whether ``R`` took more than one value across the
            samples.
        observation_noise_spread: the largest pairwise distance between the sampled
            ``R``.
        observation_noise_loewner_ordered: whether the extreme pair of sampled ``R`` is
            strictly ordered, the cheap sufficient condition for the epistemic value to
            move.
        dynamics_noise_varies: whether ``Q`` took more than one value across the
            samples. Always ``False`` for a fixed ``Q``.
        dynamics_noise_spread: the largest pairwise distance between the sampled ``Q``.
        epistemic_varies: whether the epistemic value differed across the samples.
        epistemic_range: its smallest and largest sampled values, in nats.
    """

    n_samples: int
    n_observations: int
    observation_matrix_rank: int
    observation_matrix_full_row_rank: bool
    observation_noise_definite: bool
    observation_noise_indefinite_at: tuple[tuple[float, ...], ...]
    observation_noise_varies: bool
    observation_noise_spread: float
    observation_noise_loewner_ordered: bool
    dynamics_noise_varies: bool
    dynamics_noise_spread: float
    epistemic_varies: bool
    epistemic_range: tuple[float, float]

    @property
    def flattens(self) -> bool:
        """Whether the sampled evidence says the noise is a constant in disguise.

        True when neither ``R`` nor ``Q`` moved across the sampled means: the covariance
        recursion then never consults the action, so a fixed noise schedule reproduces
        the agent and the epistemic term cannot tell two policies apart. A ``Q(x)`` that
        moves keeps it ``False`` with ``R`` fixed, since the action then reaches the
        recursion through ``Q``.
        """
        return not (self.observation_noise_varies or self.dynamics_noise_varies)

    def summary(self) -> str:
        """A few lines a human can read, one per condition."""
        lo, hi = self.epistemic_range
        mark = {True: "yes", False: "NO "}
        return "\n".join(
            [
                f"sampled {self.n_samples} predicted mean(s)",
                f"  full row rank C   {mark[self.observation_matrix_full_row_rank]}  "
                f"(rank {self.observation_matrix_rank} of {self.n_observations})",
                f"  R positive def.   {mark[self.observation_noise_definite]}  "
                f"({len(self.observation_noise_indefinite_at)} failing sample(s))",
                f"  R varies          {mark[self.observation_noise_varies]}  "
                f"(spread {self.observation_noise_spread:.3e})",
                f"  Q varies          {mark[self.dynamics_noise_varies]}  "
                f"(spread {self.dynamics_noise_spread:.3e})",
                f"  epistemic varies  {mark[self.epistemic_varies]}  "
                f"({lo:.6f} .. {hi:.6f} nats)",
                f"  R Loewner-ordered {mark[self.observation_noise_loewner_ordered]}",
            ]
        )

    def __str__(self) -> str:
        """The same text ``summary`` returns."""
        return self.summary()


def _linearizations(
    observation, observation_matrix: np.ndarray, means: np.ndarray
) -> list[np.ndarray]:
    """Each mean's noise covariance, read through whatever the caller supplied."""
    if observation is None:  # a fixed sensor carries no per-state noise
        return []
    return [np.asarray(observation.linearize(mu)[1], dtype=float) for mu in means]


def _dynamics_noises(
    process, dynamics_noise: ArrayLike, means: np.ndarray
) -> list[np.ndarray]:
    """Each mean's ``Q``, taken at that mean when the model carries a ``Q(x)``.

    The EFE kernel adds ``Q`` at the predicted mean, so the probe adds the same one. The
    model checks ``Q(x)`` at the origin only, so each is checked here as well, as
    ``harness.World`` checks the ``Q`` at every state it reaches.

    Raises:
        ValueError: If a sampled ``Q`` is not a covariance.
    """
    if process is None:
        return [np.asarray(dynamics_noise, dtype=float)] * len(means)
    noises = []
    for mu in means:
        noise = process.noise_at(mu)
        at = tuple(float(v) for v in mu)
        validate_covariance(
            noise, f"dynamics_noise_model.noise_at(state) at the sampled mean {at}"
        )
        noises.append(np.asarray(noise, dtype=float))
    return noises


def _spread(matrices: list[np.ndarray]) -> float:
    """The largest distance, entry by entry, between any two of ``matrices``."""
    spread = 0.0
    for i in range(len(matrices)):
        for j in range(i + 1, len(matrices)):
            spread = max(spread, float(np.max(np.abs(matrices[i] - matrices[j]))))
    return spread


def probe_model(
    model: "LinearGaussianModel | ProbeBackend",
    belief: "Belief",
    actions: Sequence[ArrayLike],
    *,
    tol: float = 1e-12,
) -> ProbeReport:
    """Probe a model's sensor over the predicted means a set of actions reaches.

    Each action is pushed through the prediction step to get ``(μ⁻, Σ⁻)``; the sensor is
    linearized at each ``μ⁻`` and the four conditions are evaluated over the resulting
    sample. This is the reachable set the conditions are about — vary ``actions`` to
    widen it. A state-dependent ``Q(x)`` enters ``Σ⁻`` at each ``μ⁻``, where the EFE
    kernel evaluates it.

    Args:
        model: a [`LinearGaussianModel`][cpomdp.LinearGaussianModel], or a backend
            exposing ``predicted_belief`` / ``observation_noise_at`` /
            ``observation_model`` ([`CouplingGraphBackend`][cpomdp.CouplingGraphBackend]
            does).
        belief: the belief to predict from — the shared prior every action starts at.
        actions: the candidate actions to sample.
        tol: below this, two noise covariances or two epistemic values count as equal.

    Returns:
        A [`ProbeReport`][cpomdp.ProbeReport].

    Raises:
        ValueError: If ``actions`` is empty, or if a state-dependent ``Q(x)`` is not a
            covariance at one of the sampled means.
    """
    if len(actions) == 0:
        raise ValueError("probe_model needs at least one action to predict under.")

    if hasattr(model, "predicted_belief"):  # a graph backend
        backend = cast("ProbeBackend", model)
        observation_matrix = np.asarray(backend.observation_model[0], dtype=float)
        predicted = [backend.predicted_belief(belief, np.asarray(a)) for a in actions]
        means = np.asarray([np.asarray(p.mean, dtype=float) for p in predicted])
        covs = [np.asarray(p.cov, dtype=float) for p in predicted]
        noises = [
            np.asarray(backend.observation_noise_at(mu), dtype=float) for mu in means
        ]
        dynamics_spread = 0.0  # the graph's transitions carry a fixed Q
    else:  # a flat LinearGaussianModel
        observation_matrix = np.asarray(model.observation_matrix, dtype=float)
        dynamics_matrix = np.asarray(model.dynamics_matrix, dtype=float)
        prior_mean = np.asarray(belief.mean, dtype=float)
        prior_cov = np.asarray(belief.cov, dtype=float)
        # A control-free model goes nowhere under any action, which is itself an answer:
        # every sample lands on the same predicted mean and the noise cannot move.
        if model.control_matrix is None:
            means = np.asarray([dynamics_matrix @ prior_mean for _ in actions])
        else:
            control_matrix = np.asarray(model.control_matrix, dtype=float)
            means = np.asarray(
                [
                    dynamics_matrix @ prior_mean
                    + control_matrix @ np.asarray(a, dtype=float).ravel()
                    for a in actions
                ]
            )
        propagated = dynamics_matrix @ prior_cov @ dynamics_matrix.T
        dynamics_noises = _dynamics_noises(
            model.dynamics_noise_model, model.dynamics_noise, means
        )
        covs = [propagated + q for q in dynamics_noises]
        dynamics_spread = _spread(dynamics_noises)
        fixed = np.asarray(model.observation_noise, dtype=float)
        noises = _linearizations(
            model.observation_model, observation_matrix, means
        ) or [fixed] * len(actions)

    rank = int(np.linalg.matrix_rank(observation_matrix))
    n_obs = int(observation_matrix.shape[0])

    indefinite = tuple(
        tuple(float(v) for v in mu)
        for mu, r in zip(means, noises, strict=True)
        if not is_positive_definite(r)
    )

    spread = _spread(noises)

    epistemics = [
        epistemic_value(cov, observation_matrix, r)
        for cov, r in zip(covs, noises, strict=True)
    ]
    finite = [e for e in epistemics if np.isfinite(e)]
    lo, hi = (min(finite), max(finite)) if finite else (float("nan"), float("nan"))

    # The extreme pair is the one most likely to be ordered; a strictly ordered pair is
    # enough on its own, so there is no need to walk every combination.
    comparable = False
    if finite and len(noises) > 1:
        i_lo = int(np.argmin([e if np.isfinite(e) else np.inf for e in epistemics]))
        i_hi = int(np.argmax([e if np.isfinite(e) else -np.inf for e in epistemics]))
        comparable = loewner_order(noises[i_lo], noises[i_hi]) in ("a<b", "b<a")

    return ProbeReport(
        n_samples=len(actions),
        n_observations=n_obs,
        observation_matrix_rank=rank,
        observation_matrix_full_row_rank=rank == n_obs,
        observation_noise_definite=not indefinite,
        observation_noise_indefinite_at=indefinite,
        observation_noise_varies=spread > tol,
        observation_noise_spread=spread,
        observation_noise_loewner_ordered=comparable,
        dynamics_noise_varies=dynamics_spread > tol,
        dynamics_noise_spread=dynamics_spread,
        epistemic_varies=bool(finite) and (hi - lo) > tol,
        epistemic_range=(lo, hi),
    )


@dataclass(frozen=True)
class RolloutConditioning:
    """Per-step conditioning of an H-step rollout, computed on the host.

    Every array field is length-H, indexed by rollout step. The condition numbers and
    the smallest eigenvalue are what a monotonically contracting ``Σ_post`` degrades
    first; ``all_positive_definite`` folds a per-step positive-definiteness check over
    ``Σ⁺``, ``S`` and ``Σ_post`` into one flag, so a matrix that has gone non-PD
    anywhere in the horizon is visible without inspecting every step.

    Attributes:
        cond_sigma_pred: 2-norm condition number of each predicted covariance ``Σ⁺``.
        cond_s: condition number of each innovation covariance ``S``.
        cond_sigma_post: condition number of each contracted covariance ``Σ_post``.
        min_eig_sigma_post: smallest eigenvalue of each ``Σ_post``.
        all_positive_definite: whether every ``Σ⁺``, ``S`` and ``Σ_post`` is PD.
    """

    cond_sigma_pred: np.ndarray
    cond_s: np.ndarray
    cond_sigma_post: np.ndarray
    min_eig_sigma_post: np.ndarray
    all_positive_definite: bool


def _condition_numbers(matrices: ArrayLike) -> np.ndarray:
    """The 2-norm condition number of each matrix in a stack, on the host.

    One implementation for every place a conditioning discipline is applied, so the
    rollout and the scoring harness read the same number for the same matrix. Pure
    NumPy: a condition number is asserted against on the host, outside any ``jit``.

    Args:
        matrices: a stack of square matrices, shape ``(k, n, n)``.

    Returns:
        A length-``k`` array of condition numbers, ``inf`` where a matrix is singular.
    """
    stack = np.asarray(matrices, dtype=float)
    if stack.ndim != 3 or stack.shape[1] != stack.shape[2]:
        raise ValueError(
            f"_condition_numbers reads a stack of square matrices, shape (k, n, n), "
            f"got shape {stack.shape}"
        )
    return np.array([np.linalg.cond(m) for m in stack])


def rollout_conditioning(trace: "PolicyEfeTrace") -> RolloutConditioning:
    """Per-step condition numbers and PD flags of a rollout trace, on the host.

    Reads the per-step matrices ``policy_efe_trace`` already returns and applies the
    condition-number discipline to the rollout. The assertions a caller builds on top —
    a ceiling on the condition numbers, a floor under ``min_eig_sigma_post`` — belong
    here on the host, outside any ``jit``/``vmap``, which is the only place they can
    raise. Pure NumPy.

    Args:
        trace: a ``PolicyEfeTrace`` — read for its ``sigma_pred``
            (``Σ⁺``), ``s`` (``S``) and ``sigma_post`` (``Σ_post``) columns.

    Returns:
        A ``RolloutConditioning``.
    """
    sigma_pred = np.asarray(trace.sigma_pred, dtype=float)
    s = np.asarray(trace.s, dtype=float)
    sigma_post = np.asarray(trace.sigma_post, dtype=float)

    return RolloutConditioning(
        cond_sigma_pred=_condition_numbers(sigma_pred),
        cond_s=_condition_numbers(s),
        cond_sigma_post=_condition_numbers(sigma_post),
        min_eig_sigma_post=np.array([np.linalg.eigvalsh(m).min() for m in sigma_post]),
        all_positive_definite=all(
            is_positive_definite(m)
            for column in (sigma_pred, s, sigma_post)
            for m in column
        ),
    )
