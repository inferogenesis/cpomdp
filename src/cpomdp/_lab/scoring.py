"""Scoring a declared cross of models against a declared cross of ways to infer.

The two axes live in ``cpomdp._lab.constructors``: one for what a model gets
wrong about the process, one for what its filter gets wrong about the model. A run
crosses them. Every separation reported from such a run is a comparison between two of
its cells, which is a decision only if the cross was visited in full.

So the cross carries a certificate. `|model axis| x |inference axis|` is a product, and
a product needs its factors the way a policy tree needs its base and exponent: 12 is
``3 x 4`` and ``2 x 6``, so a bare count names neither which axes were crossed nor which
version of each was declared. `ProductCompletenessCertificate` carries both, which is
what makes two runs over different crosses tellable apart (standing prohibition 9).
"""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from cpomdp._lab.constructors import (
    ConstructorSet,
    InferenceKind,
    InferenceSet,
    ModelSpec,
)
from cpomdp._lab.harness import DrivenRun, ExogenousActionSequence
from cpomdp._validation import require_fixed_noise
from cpomdp.backends.base import InferenceBackend
from cpomdp.backends.kalman import KalmanBackend
from cpomdp.diagnostics import condition_numbers
from cpomdp.enumeration import IncompleteEnumerationError
from cpomdp.types import Belief, LinearGaussianModel
from warrantlib import AxisDeclaration, ProductCompletenessCertificate, Warrant

__all__ = [
    "INFERENCE_AXIS",
    "MODEL_AXIS",
    "CellScore",
    "ConstructorCross",
    "CrossCell",
    "CrossScore",
    "Decomposition",
    "RunConditioning",
    "Separation",
    "ThreeTermEvaluator",
    "build_cross",
    "gaussian_kl",
    "inference_gap_step",
    "misspecification_step",
    "observation_predictive",
]

# The closed forms here are Gaussian ones. Under R(x) the exact posterior and the
# predictive are not Gaussian, and scoring them needs the reference filter rather than
# a formula.
_FIXED_NOISE_ONLY = "the closed-form terms here are for fixed noise only"

#: One filter step with its belief and action fixed: the reading in, the posterior out.
StepUpdate = Callable[[np.ndarray], Belief]

# How far a probed update may sit from the affine map read off it, relative to the
# map's own size. Rounding across three probes sits near 1e-15; a rule that is not
# affine misses by orders.
_AFFINE_TOLERANCE = 1e-9

#: What each axis is called on the certificate. Declared here rather than written at the
#: construction site, so the two runs a ledger joins cannot disagree about the names.
MODEL_AXIS = "model"
INFERENCE_AXIS = "inference"


@dataclass(frozen=True)
class Decomposition:
    """The two divergences of the three-term accounting, floor excluded.

    ``E[F] = H(p*) + misspecification + inference gap``. The floor is a property of
    ``p*`` and no business of a scored cell, so there is no entropy field, no estimator
    slot and no total. A term reached by subtracting ``H(p*)`` cannot be represented
    here (standing prohibition 1).

    Args:
        misspecification: ``D_KL[p*(y|u) ‖ p(y|u)]`` in nats, zero iff the cell's model
            is ``p*``.
        inference_gap: ``E_{y∼p*}[ D_KL[q ‖ p(x|y)] ]`` in nats, zero iff the cell
            infers exactly under its own model.
    """

    misspecification: float
    inference_gap: float


@dataclass(frozen=True)
class RunConditioning:
    """The conditioning of every matrix a cell's score factored or inverted, per step.

    A divergence is read off inverses and log-determinants, and an ill-conditioned
    matrix can manufacture or destroy the twelve orders a separation claims
    (``research/warrant_ledger.md`` section 4). So the numbers travel with the score
    rather than being looked up afterwards. Every field is length-``k``, indexed by
    step, computed on the host by ``cpomdp.diagnostics.condition_numbers``.

    Attributes:
        cond_s_true: of the true predictive covariance ``S*``, which weights the
            gap's average and is the divergence's first argument for the
            misspecification term.
        cond_s_model: of the cell model's predictive covariance ``S``, inverted for
            the misspecification term.
        cond_sigma_exact: of the exact posterior covariance under the cell's model,
            inverted for the inference gap.
        cond_sigma_agent: of the cell filter's posterior covariance, the gap's first
            argument.
    """

    cond_s_true: np.ndarray
    cond_s_model: np.ndarray
    cond_sigma_exact: np.ndarray
    cond_sigma_agent: np.ndarray

    @property
    def worst_s(self) -> float:
        """The largest condition number among the predictive covariances."""
        return float(max(self.cond_s_true.max(), self.cond_s_model.max()))

    @property
    def worst_sigma(self) -> float:
        """The largest condition number among the posterior covariances."""
        return float(max(self.cond_sigma_exact.max(), self.cond_sigma_agent.max()))


@dataclass(frozen=True)
class Separation:
    """One off-diagonal cell's claim: one term moved and the other stayed put.

    The claim is the ratio, not the small number. "Below ``1e-12``" is empty if the
    pinned term is naturally ``O(1e-13)`` in this configuration, so the moving term's
    magnitude is printed beside it and the ratio between them is what carries the
    argument (``research/warrant_ledger.md`` section 4). A pinned term of exactly
    ``0.0`` gives a ratio of ``inf``, and the pinned value beside it says why.

    Args:
        pinned: Which term the cell's axes leave untouched, ``"misspecification"``
            or ``"inference_gap"``.
        pinned_value: What that term read, in nats.
        moving_value: What the other term read, in nats.
    """

    pinned: str
    pinned_value: float
    moving_value: float

    @property
    def moving(self) -> str:
        """The term the cell's axes move."""
        return (
            "inference_gap" if self.pinned == "misspecification" else "misspecification"
        )

    @property
    def ratio(self) -> float:
        """``moving_value / pinned_value``, ``inf`` where the pinned term is zero."""
        if self.pinned_value == 0.0:
            return float("inf")
        return self.moving_value / self.pinned_value


@dataclass(frozen=True)
class CellScore:
    """One cell's two divergences over a run, with the conditioning behind them.

    Args:
        model_name: The cell's row, as the model axis declared it.
        inference_name: The cell's column, as the inference axis declared it.
        model_is_correct: Whether the row is the unperturbed model.
        inference_is_exact: Whether the column is the exact rule.
        decomposition: The two terms, each summed over the run's steps.
        conditioning: Of every matrix the terms factored or inverted, per step.
        steps: How many steps the sums run over.
    """

    model_name: str
    inference_name: str
    model_is_correct: bool
    inference_is_exact: bool
    decomposition: Decomposition
    conditioning: RunConditioning
    steps: int

    @property
    def separation(self) -> Separation | None:
        """The cell's separation, if exactly one of its axes is at the reference.

        ``None`` on the calibration cell, where both terms are pinned, and on a cell
        where both axes moved, where neither is.
        """
        if self.model_is_correct == self.inference_is_exact:
            return None
        terms = self.decomposition
        if self.model_is_correct:
            return Separation(
                pinned="misspecification",
                pinned_value=terms.misspecification,
                moving_value=terms.inference_gap,
            )
        return Separation(
            pinned="inference_gap",
            pinned_value=terms.inference_gap,
            moving_value=terms.misspecification,
        )


@dataclass(frozen=True)
class CrossScore:
    """Every cell of a cross scored over one run, with the certificate it rests on.

    Args:
        cells: The scores, in the cross's own cell order.
        certificate: That the cross was visited in full, carried over from the cross.
        action_sequence_version: The sequence the run was driven by.
    """

    cells: tuple[CellScore, ...]
    certificate: ProductCompletenessCertificate
    action_sequence_version: str

    @property
    def both_positive(self) -> tuple[CellScore, ...]:
        """The cells where both axes moved and both terms read above zero.

        Off-diagonal separations are read against these. A cross with none has
        not shown that both terms can move at all, and its separations say nothing.
        """
        return tuple(
            cell
            for cell in self.cells
            if not cell.model_is_correct
            and not cell.inference_is_exact
            and cell.decomposition.misspecification > 0.0
            and cell.decomposition.inference_gap > 0.0
        )

    @property
    def separations(self) -> tuple[tuple[CellScore, Separation], ...]:
        """Each off-diagonal cell with its separation, in cell order.

        Raises:
            ValueError: If no cell moved both terms. A separation is a contrast, and
                without a cell where both terms move the contrast has no reference.
        """
        if not self.both_positive:
            raise ValueError(
                "off-diagonal separations are read against a cell where both terms "
                "move, and no cell of this cross moved both; declare a perturbed model "
                "under a degraded rule, or read no separation"
            )
        return tuple(
            (cell, cell.separation)
            for cell in self.cells
            if cell.separation is not None
        )

    def render(self) -> str:
        """The scores as text, one row per cell, with what each row's claim needs.

        A separation row prints its pinned term, its moving term, their ratio, and the
        worst conditioning of the posterior and predictive covariances behind them,
        on that one line. A row that asserts a separation cannot be printed without them
        (standing prohibition 5). The both-positive cells are labelled as the
        reference the separations are read against.
        """
        lines = [
            f"cross {self.certificate} over sequence {self.action_sequence_version!r}",
            "model / rule | misspecification | inference gap | claim | "
            "cond(Σ) | cond(S)",
        ]
        for cell in self.cells:
            terms = cell.decomposition
            separation = cell.separation
            if separation is not None:
                claim = (
                    f"{separation.moving} moved {separation.moving_value:.6e} against "
                    f"{separation.pinned} {separation.pinned_value:.6e}, ratio "
                    f"{separation.ratio:.3e}"
                )
            elif cell in self.both_positive:
                claim = "both moved: the reference for the separations"
            elif cell.model_is_correct and cell.inference_is_exact:
                claim = "calibration"
            else:
                claim = "both axes moved and a term read zero"
            lines.append(
                f"{cell.model_name} / {cell.inference_name} | "
                f"{terms.misspecification:.6e} | {terms.inference_gap:.6e} | {claim} | "
                f"{cell.conditioning.worst_sigma:.3e} | {cell.conditioning.worst_s:.3e}"
            )
        return "\n".join(lines)


@dataclass(frozen=True)
class CrossCell:
    """One cell: a model built one way, inferred another.

    ``model`` is the thing the model-axis parts name, so it carries no ``model_``
    prefix of its own. The inference axis has no such centre, so both its parts take
    the prefix.

    Args:
        model_name: The perturbation that built ``model``, as the model axis
            declared it.
        inference_name: The rule that built ``inference_backend``.
        model: The model this cell is scored under.
        inference_backend: The filter this cell infers with. It reports ``model`` as
            its own, degraded or not.
        model_is_correct: Whether the row perturbs nothing, so ``model`` is the spec
            as declared.
        inference_is_exact: Whether the column is the exact rule, so the filter is
            the exact step under ``model``.
    """

    model_name: str
    inference_name: str
    model: LinearGaussianModel
    inference_backend: InferenceBackend
    model_is_correct: bool
    inference_is_exact: bool


@dataclass(frozen=True)
class ConstructorCross:
    """Every cell of the two axes, and the evidence that all of them were built.

    Args:
        cells: The cells, model-major in declaration order, so a row is one model
            inferred every declared way and a column is one way of inferring across
            every declared model.
        certificate: That the product of the declared axis sizes was reached.
    """

    cells: tuple[CrossCell, ...]
    certificate: ProductCompletenessCertificate


def build_cross(
    spec: ModelSpec, models: ConstructorSet, rules: InferenceSet
) -> ConstructorCross:
    """Build every cell of the two axes, and certify that none was missed.

    One model per row, shared across that row's rules. The axes are independent: a
    perturbation is one model inferred several ways, and rebuilding it per cell would
    put a difference on the inference axis that belongs to neither.

    Args:
        spec: The parameters the model axis perturbs.
        models: The declared model axis.
        rules: The declared inference axis.

    Returns:
        The cells and their certificate.

    Raises:
        IncompleteEnumerationError: if fewer cells were built than the two axes
            declare. The count is carried through the loop rather than read off the
            result, so a cell lost on the way out is a failure here instead of a
            shorter run in which every remaining cell still passes.
        ValueError: if a cell cannot be built. A rule may refuse the model its row
            supplies, as a frozen gain refuses state-dependent noise, and a
            perturbation may leave no valid covariance. The cell that refused is named,
            since the message the axes raise names only what it was asked to build.
        RuntimeError: if a frozen gain's steady-state recursion does not converge, from
            the same place and named the same way.
    """
    expected = models.size * rules.size
    correct = {p.name: p.parameter is None for p in models.perturbations}
    exact = {r.name: r.kind is InferenceKind.EXACT for r in rules.rules}
    cells: list[CrossCell] = []
    visited = 0
    for model_name, model in models.build_all(spec):
        for inference_name, backend in _built_over(model, rules, model_name):
            cells.append(
                CrossCell(
                    model_name=model_name,
                    inference_name=inference_name,
                    model=model,
                    inference_backend=backend,
                    model_is_correct=correct[model_name],
                    inference_is_exact=exact[inference_name],
                )
            )
            visited += 1

    if visited != expected:  # ADR-030: the certificate is asserted, not assumed
        # Rendered by a certificate rather than by hand, so the failure and the
        # evidence describe the cross in one voice. CORROBORATED because a shortfall
        # is what it says it is; a PROVED one would refuse to construct here.
        sampled = _certificate(models, rules, expected=expected, visited=visited)
        raise IncompleteEnumerationError(
            f"the cross visited {visited} of an expected {expected} cells over "
            f"{sampled.set_description}; the run would be a sample, not a decision."
        )
    return ConstructorCross(
        cells=tuple(cells),
        certificate=_certificate(
            models, rules, expected=expected, visited=visited, warrant=Warrant.PROVED
        ),
    )


def _built_over(
    model: LinearGaussianModel, rules: InferenceSet, model_name: str
) -> tuple[tuple[str, InferenceBackend], ...]:
    """One row's backends, with the cell named on anything the row refuses.

    `InferenceSet.build_all` names the rule that refused and cannot name the row, since
    it was handed a model and not the perturbation that produced one. On a cross that
    is half the address.
    """
    try:
        return rules.build_all(model)
    except (ValueError, RuntimeError) as refusal:
        raise type(refusal)(
            f"the cross could not build a cell on row {model_name!r}: {refusal}"
        ) from refusal


def _certificate(
    models: ConstructorSet,
    rules: InferenceSet,
    *,
    expected: int,
    visited: int,
    warrant: Warrant = Warrant.CORROBORATED,
) -> ProductCompletenessCertificate:
    """The cross's certificate, with both axes declared as the sets declared them."""
    return ProductCompletenessCertificate(
        expected=expected,
        visited=visited,
        warrant=warrant,
        axes=(
            AxisDeclaration(name=MODEL_AXIS, size=models.size, version=models.version),
            AxisDeclaration(
                name=INFERENCE_AXIS, size=rules.size, version=rules.version
            ),
        ),
    )


def gaussian_kl(
    mean_a: ArrayLike, cov_a: ArrayLike, mean_b: ArrayLike, cov_b: ArrayLike
) -> float:
    """``D_KL[N(mean_a, cov_a) ‖ N(mean_b, cov_b)]`` in nats, as non-negative parts.

    The textbook form subtracts ``n`` from a trace and one log-determinant from
    another. Near equality those are differences of like-sized numbers, so a
    divergence that is truly zero reads as rounding noise of either sign, and a
    reading below ``1e-12`` then says nothing about which of two Gaussians moved
    (``research/warrant_ledger.md`` section 4). Here every part is a square or a
    ``λ − 1 − ln λ`` over the eigenvalues of ``cov_b⁻¹ · cov_a``, each of which is
    non-negative on its own. Nothing is subtracted, so a small result is a small
    divergence rather than a cancellation.

    Args:
        mean_a: The divergence's first argument's mean, shape ``(n,)``.
        cov_a: Its covariance, shape ``(n, n)``, positive definite.
        mean_b: The second argument's mean, shape ``(n,)``.
        cov_b: Its covariance, shape ``(n, n)``, positive definite.

    Returns:
        The divergence, ``0.0`` exactly when the two Gaussians are identical.

    Raises:
        ValueError: On a shape mismatch, or if either covariance is not positive
            definite. A degenerate Gaussian has no density for the divergence to read.
    """
    mean_a, mean_b = np.asarray(mean_a, dtype=float), np.asarray(mean_b, dtype=float)
    cov_a, cov_b = np.asarray(cov_a, dtype=float), np.asarray(cov_b, dtype=float)
    if mean_a.ndim != 1 or mean_a.shape != mean_b.shape:
        raise ValueError(
            f"the two means must be vectors of one length, got shapes "
            f"{mean_a.shape} and {mean_b.shape}"
        )
    expected = (mean_a.shape[0], mean_a.shape[0])
    if cov_a.shape != expected or cov_b.shape != expected:
        raise ValueError(
            f"the two covariances must be {expected} to match the means, got shapes "
            f"{cov_a.shape} and {cov_b.shape}"
        )
    factor_a = _cholesky_or_refuse(cov_a, "cov_a")
    factor_b = _cholesky_or_refuse(cov_b, "cov_b")

    # Singular values of L_b⁻¹ L_a squared are the eigenvalues of cov_b⁻¹ cov_a.
    ratios = np.linalg.svd(_forward_substitute(factor_b, factor_a), compute_uv=False)
    shape_term = _excess_over_log(ratios**2 - 1.0)
    shift = _forward_substitute(factor_b, mean_a - mean_b)
    return float(0.5 * (shape_term.sum() + shift @ shift))


def _forward_substitute(lower: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    """``lower⁻¹ · rhs`` for a lower-triangular ``lower``, row by row.

    A general solve pivots and leaves a ``1e-16`` residue where the factor is solved
    against itself, which the series then squares into a divergence of ``1e-32``
    between identical Gaussians. Substitution in row order returns the identity
    exactly there, so identical inputs read ``0.0`` rather than something small.
    """
    solution = np.zeros_like(rhs, dtype=float)
    for row in range(lower.shape[0]):
        solution[row] = (rhs[row] - lower[row, :row] @ solution[:row]) / lower[row, row]
    return solution


# Below this, ``x − log1p(x)`` is itself a cancellation: the two agree to nearly every
# digit, and the series has no such subtraction.
_SERIES_BELOW = 1e-4


def _excess_over_log(excess: np.ndarray) -> np.ndarray:
    """``x − ln(1 + x)`` elementwise, non-negative for ``x > −1``, without cancelling.

    Near zero the direct form subtracts two numbers that agree to almost every digit.
    The series ``x²/2 − x³/3 + x⁴/4 − x⁵/5 + x⁶/6`` evaluates the same function there
    with a truncation error below ``x⁷/7``, which at the switch is ``1e-29``.
    """
    small = np.abs(excess) < _SERIES_BELOW
    x = np.where(small, excess, 0.0)
    series = x**2 / 2 - x**3 / 3 + x**4 / 4 - x**5 / 5 + x**6 / 6
    return np.where(small, series, excess - np.log1p(excess))


def _cholesky_or_refuse(cov: np.ndarray, name: str) -> np.ndarray:
    """The lower Cholesky factor, or a ``ValueError`` naming the argument."""
    if not np.allclose(cov, cov.T, atol=1e-12):
        raise ValueError(f"{name} must be symmetric")
    try:
        return np.linalg.cholesky(cov)
    except np.linalg.LinAlgError as failure:
        raise ValueError(
            f"{name} must be positive definite; a degenerate Gaussian has no density "
            f"for a divergence to read"
        ) from failure


def observation_predictive(
    model: LinearGaussianModel, belief: Belief, action: ArrayLike | None
) -> tuple[np.ndarray, np.ndarray]:
    """The mean and covariance of ``p(y | u)`` one step ahead under ``model``.

    The belief is pushed through the dynamics and then through the sensor::

        mean = C · (A · μ + B · u)
        cov  = C · (A · Σ · Aᵀ + Q) · Cᵀ + R

    Gaussian because everything in the model is, which is why this asks for fixed
    noise. Under a state-dependent ``R(x)`` or ``Q(x)`` the predictive is a scale
    mixture and has no covariance that describes it.

    Args:
        model: The model doing the predicting.
        belief: The state belief before the step, over the model's ``n`` states.
        action: The action applied over the step, shape ``(p,)``. ``None`` for a model
            without a control matrix.

    Returns:
        ``(mean, cov)`` of the predicted observation, shapes ``(m,)`` and ``(m, m)``.

    Raises:
        ValueError: If the model carries state-dependent noise, if the model has a
            control matrix and ``action`` is ``None``, or if the belief is not over
            the model's state.
    """
    predicted = predicted_belief(model, belief, action)
    observation_matrix = np.asarray(model.observation_matrix, dtype=float)  # C
    cov_pred = np.asarray(predicted.cov, dtype=float)
    return (
        observation_matrix @ np.asarray(predicted.mean, dtype=float),
        observation_matrix @ cov_pred @ observation_matrix.T
        + np.asarray(model.observation_noise, dtype=float),
    )


def predicted_belief(
    model: LinearGaussianModel, belief: Belief, action: ArrayLike | None
) -> Belief:
    """``p(x | u)`` one step ahead under ``model`` from ``belief``, the predict step.

    ``μ⁻ = A · μ + B · u`` and ``Σ⁻ = A · Σ · Aᵀ + Q``. This is the ``p(x)`` the
    model's joint ``p(x, y) = p(y | x) · p(x)`` is built on for the step, and so the
    prior the exact posterior conditions.

    Args:
        model: The model doing the predicting, with fixed noise.
        belief: The state belief before the step, over the model's ``n`` states.
        action: The action applied over the step, shape ``(p,)``. ``None`` for a model
            without a control matrix.

    Returns:
        The predicted belief.

    Raises:
        ValueError: If the model carries state-dependent noise, if the model has a
            control matrix and ``action`` is ``None``, or if the belief is not over
            the model's state.
    """
    require_fixed_noise(model, _FIXED_NOISE_ONLY)
    mean = np.asarray(belief.mean, dtype=float)
    cov = np.asarray(belief.cov, dtype=float)
    if mean.shape != (model.n_states,):
        raise ValueError(
            f"the belief is over {mean.shape[0]} states and the model has "
            f"{model.n_states}"
        )
    dynamics_matrix = np.asarray(model.dynamics_matrix, dtype=float)  # A
    mean_pred = dynamics_matrix @ mean
    if model.control_matrix is not None:
        if action is None:
            raise ValueError(
                "this model has a control matrix; the step needs an action"
            )
        mean_pred = mean_pred + np.asarray(
            model.control_matrix, dtype=float
        ) @ np.asarray(action, dtype=float)
    cov_pred = dynamics_matrix @ cov @ dynamics_matrix.T + np.asarray(
        model.dynamics_noise, dtype=float
    )
    return Belief(mean=mean_pred, cov=cov_pred)


def misspecification_step(
    truth: LinearGaussianModel,
    truth_belief: Belief,
    model: LinearGaussianModel,
    model_belief: Belief,
    action: ArrayLike | None,
) -> float:
    """``D_KL[p*(y|u) ‖ p(y|u)]`` for one step, in nats.

    Each side predicts from its own exact belief, so the term reads what the two
    models say about the next reading and nothing about how either is being filtered.
    A cell that infers badly under the true model scores zero here at every step,
    which is what keeps the two axes separable (battery C2, C3).

    Args:
        truth: ``p*``, the process the observations come from.
        truth_belief: The exact filter's belief under ``truth`` before this step.
        model: ``p``, the cell's model.
        model_belief: The exact filter's belief under ``model`` before this step.
        action: The action applied over the step, common to both.

    Returns:
        The divergence. ``0.0`` exactly when the two models agree in value and their
        beliefs do.
    """
    true_mean, true_cov = observation_predictive(truth, truth_belief, action)
    model_mean, model_cov = observation_predictive(model, model_belief, action)
    return gaussian_kl(true_mean, true_cov, model_mean, model_cov)


def inference_gap_step(
    agent_update: StepUpdate,
    exact_update: StepUpdate,
    true_mean: ArrayLike,
    true_cov: ArrayLike,
) -> float:
    """``E_{y∼p*}[ D_KL[q(y) ‖ p(x|y)] ]`` for one step, in nats, in closed form.

    ``q(y)`` is what the cell's filter believes after reading ``y`` and ``p(x|y)`` is
    the exact posterior under the cell's own model after the same reading. Both are
    read off their updates as affine maps in ``y``, which every Gaussian filter step
    is: the mean is a gain times the innovation and the covariance does not see the
    reading at all. The average then has a closed form. With ``D`` the difference of
    the two gains and ``S`` the true predictive covariance::

        E_y[KL] = KL at y = true_mean  +  ½ · tr(Σ_p⁻¹ · D · S · Dᵀ)

    The trace is a Frobenius norm of a whitened ``D``, so like the divergence it is
    built from squares and cannot read small by cancellation.

    Args:
        agent_update: The cell's filter step at its current belief, ``y`` in and its
            posterior out.
        exact_update: The exact step under the cell's model at the exact filter's own
            current belief, ``y`` in and the exact posterior out.
        true_mean: The mean of ``p*(y | u)`` for this step, shape ``(m,)``.
        true_cov: Its covariance, shape ``(m, m)``, positive definite.

    Returns:
        The averaged gap. ``0.0`` exactly when the two updates are one computation.

    Raises:
        ValueError: If either update is not affine in the reading, or if its
            covariance depends on the reading. The closed form does not apply, and a
            number from it would be a formula applied to the wrong object.
    """
    centre = np.asarray(true_mean, dtype=float)
    agent = _affine_in_observation(agent_update, centre, "agent_update")
    exact = _affine_in_observation(exact_update, centre, "exact_update")
    at_centre = gaussian_kl(agent.mean, agent.cov, exact.mean, exact.cov)
    drift = agent.gain - exact.gain  # D
    whitened = np.linalg.solve(
        _cholesky_or_refuse(exact.cov, "the exact posterior covariance"),
        drift @ _cholesky_or_refuse(np.asarray(true_cov, dtype=float), "true_cov"),
    )
    return float(at_centre + 0.5 * np.sum(whitened**2))


@dataclass(frozen=True)
class _AffineStep:
    """A filter step read as ``mean(y) = mean + gain · (y − centre)``, fixed ``cov``."""

    centre: np.ndarray
    mean: np.ndarray
    gain: np.ndarray
    cov: np.ndarray


def _affine_in_observation(
    update: StepUpdate, centre: np.ndarray, name: str
) -> _AffineStep:
    """Read an update's gain off ``m + 1`` probes, then check a further probe agrees.

    The unit offsets recover the gain exactly when the map is affine. The last probe
    moves the other way along every axis at once, so it coincides with none of them,
    and is what catches a map that is not affine: the closed form built on the gain
    is then refused rather than applied.
    """
    base = update(centre)
    mean, cov = np.asarray(base.mean, dtype=float), np.asarray(base.cov, dtype=float)
    columns = []
    for axis in range(centre.shape[0]):
        offset = np.zeros_like(centre)
        offset[axis] = 1.0
        probed = update(centre + offset)
        _require_same_covariance(probed, cov, name)
        columns.append(np.asarray(probed.mean, dtype=float) - mean)
    gain = np.stack(columns, axis=1) if columns else np.zeros((mean.shape[0], 0))

    offset = -np.ones_like(centre)
    probed = update(centre + offset)
    _require_same_covariance(probed, cov, name)
    predicted = mean + gain @ offset
    scale = 1.0 + np.abs(predicted).max() + np.abs(gain).sum()
    if not np.allclose(
        np.asarray(probed.mean, dtype=float),
        predicted,
        rtol=0.0,
        atol=_AFFINE_TOLERANCE * scale,
    ):
        raise ValueError(
            f"{name} is not affine in the reading: its mean at a probe misses the map "
            f"read off unit offsets by more than {_AFFINE_TOLERANCE:g} of its size. "
            f"The closed-form average is for a Gaussian filter step and does not apply."
        )
    return _AffineStep(centre=centre, mean=mean, gain=gain, cov=cov)


def _require_same_covariance(probed: Belief, cov: np.ndarray, name: str) -> None:
    """Refuse an update whose posterior covariance moved with the reading."""
    if not np.array_equal(np.asarray(probed.cov, dtype=float), cov):
        raise ValueError(
            f"{name}'s posterior covariance depends on the reading; the closed-form "
            f"average is for a Gaussian filter step, whose covariance does not."
        )


@dataclass(frozen=True)
class ThreeTermEvaluator:
    """Scores a cell against ``p*`` over a driven run, returning its two divergences.

    Holds the true model, which the world was built from and the agents never see.
    Scoring is done by the experimenter, not by any agent, so this is the one place
    both ``p*`` and a cell's ``p`` are in hand at once.

    Each step of the run is scored given the readings before it. The true model and
    the cell's model each carry an exact filter over the same readings, and the cell's
    own filter runs beside them. At every step the misspecification term compares the
    two exact predictive distributions, and the inference gap averages the cell
    filter's divergence from the exact posterior under the cell's model over the true
    predictive. Both
    are closed forms given the history, so the run is a sampled witness of an exact
    per-step identity rather than an estimate of it.

    Args:
        truth: ``p*``, with fixed noise. A state-dependent ``R(x)`` or ``Q(x)`` is
            refused, since the closed forms are Gaussian ones.
    """

    truth: LinearGaussianModel

    def __post_init__(self) -> None:
        """Refuse a true model the closed forms do not describe."""
        require_fixed_noise(self.truth, _FIXED_NOISE_ONLY)

    def score(
        self, cell: CrossCell, run: DrivenRun, sequence: ExogenousActionSequence
    ) -> CellScore:
        """The cell's two divergences summed over the run's steps, in nats.

        The cell's filter is run afresh from its model's prior over the run's
        readings, so the score depends on the run only through what the world
        emitted and what drove it. The conditioning of every matrix the terms
        factored or inverted comes back beside them.

        Args:
            cell: The model and filter to score.
            run: The readings, from a world built on ``truth``.
            sequence: The actions that drove the run. Checked against the version
                the run recorded, since a run carries the version and not the actions.

        Returns:
            The two terms, each a sum over the run's steps, and their conditioning.

        Raises:
            ValueError: If ``sequence`` is not the one the run recorded, if the run's
                readings are not the true model's, if the cell's model carries
                state-dependent noise, or if the cell's filter step is not affine in
                the reading.
        """
        if sequence.version != run.action_sequence_version:
            raise ValueError(
                f"the run was driven by sequence {run.action_sequence_version!r} and "
                f"the sequence passed is {sequence.version!r}; a score needs the "
                f"actions the run was actually driven by"
            )
        observations = np.asarray(run.observations, dtype=float)
        if observations.shape != (sequence.horizon, self.truth.n_observations):
            raise ValueError(
                f"the run holds readings of shape {observations.shape}, and the true "
                f"model over {sequence.horizon} steps of {self.truth.n_observations} "
                f"readings expects {(sequence.horizon, self.truth.n_observations)}"
            )
        require_fixed_noise(cell.model, _FIXED_NOISE_ONLY)

        exact_truth = KalmanBackend(self.truth)
        exact_model = KalmanBackend(cell.model)
        truth_belief = self.truth.prior
        model_belief = cell.model.prior
        agent_belief = cell.model.prior
        misspecification = 0.0
        inference_gap = 0.0
        s_true, s_model, sigma_exact, sigma_agent = [], [], [], []
        for observation, action in zip(
            observations, np.asarray(sequence.actions, dtype=float), strict=True
        ):
            true_mean, true_cov = observation_predictive(
                self.truth, truth_belief, action
            )
            model_mean, model_cov = observation_predictive(
                cell.model, model_belief, action
            )
            misspecification += gaussian_kl(true_mean, true_cov, model_mean, model_cov)

            def agent_update(reading, prior=agent_belief, action=action):
                return cell.inference_backend.infer_states(reading, prior, action)

            def exact_update(reading, prior=model_belief, action=action):
                return exact_model.infer_states(reading, prior, action)

            inference_gap += inference_gap_step(
                agent_update, exact_update, true_mean, true_cov
            )

            truth_belief = exact_truth.infer_states(observation, truth_belief, action)
            model_belief = exact_update(observation)
            agent_belief = agent_update(observation)
            s_true.append(true_cov)
            s_model.append(model_cov)
            sigma_exact.append(np.asarray(model_belief.cov, dtype=float))
            sigma_agent.append(np.asarray(agent_belief.cov, dtype=float))

        return CellScore(
            model_name=cell.model_name,
            inference_name=cell.inference_name,
            model_is_correct=cell.model_is_correct,
            inference_is_exact=cell.inference_is_exact,
            decomposition=Decomposition(
                misspecification=misspecification, inference_gap=inference_gap
            ),
            conditioning=RunConditioning(
                cond_s_true=condition_numbers(s_true),
                cond_s_model=condition_numbers(s_model),
                cond_sigma_exact=condition_numbers(sigma_exact),
                cond_sigma_agent=condition_numbers(sigma_agent),
            ),
            steps=sequence.horizon,
        )

    def score_cross(
        self, cross: ConstructorCross, run: DrivenRun, sequence: ExogenousActionSequence
    ) -> CrossScore:
        """Every cell of ``cross`` scored over one run.

        Args:
            cross: The cells and the certificate that all of them were built.
            run: The readings, from a world built on ``truth``.
            sequence: The actions that drove the run.

        Returns:
            The scores in cell order, carrying the cross's certificate.
        """
        return CrossScore(
            cells=tuple(self.score(cell, run, sequence) for cell in cross.cells),
            certificate=cross.certificate,
            action_sequence_version=run.action_sequence_version,
        )
