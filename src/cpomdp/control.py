import numpy as np
from numpy.typing import ArrayLike, NDArray

from cpomdp.types import LinearGaussianModel

__all__ = ["LQRController"]


def _validate_cost(
    matrix: NDArray[np.float64], name: str, *, require_definite: bool
) -> None:
    """Symmetry + (semi)definiteness check for a preference cost matrix.

    ``goal_precision`` and ``effort_penalty`` are user input handed in once at
    construction — the trust boundary — so unlike a per-step belief covariance
    they're checked in full here. (``types._validate_covariance`` skips
    definiteness on purpose because it runs on every filter step; this runs once,
    so it doesn't have to.) Both failure modes this catches are the
    silently-wrong-in-a-loop kind that are hardest to trace downstream: a
    non-symmetric matrix (an off-diagonal typo) quietly yields a non-symmetric
    cost-to-go and a wrong gain, and a singular or indefinite ``effort_penalty``
    — which the gain solve inverts against — blows up or returns garbage.

    Args:
        matrix: The already-shape-checked cost matrix.
        name: Field name for error messages.
        require_definite: ``True`` for ``effort_penalty`` (must be positive-
            *definite*, since it is inverted against); ``False`` for
            ``goal_precision`` (positive-*semi*-definite is enough).
    """
    if not np.allclose(matrix, matrix.T):
        raise ValueError(f"{name} must be symmetric.")
    if require_definite:
        # Cholesky succeeds iff (symmetric) positive-definite — the standard test.
        try:
            np.linalg.cholesky(matrix)
        except np.linalg.LinAlgError as exc:
            raise ValueError(
                f"{name} must be positive-definite — the gain solve inverts "
                "against it — but it is singular or indefinite."
            ) from exc
    else:
        eigvals = np.linalg.eigvalsh(matrix)  # symmetric ⇒ real eigenvalues
        tol = 1e-8 * max(1.0, float(np.abs(eigvals).max()))
        if eigvals.min() < -tol:
            raise ValueError(
                f"{name} must be positive-semi-definite, but its smallest "
                f"eigenvalue is {eigvals.min():.3g}."
            )


class LQRController:
    """Steady-state LQR action selection — the action-side dual of the filter.

    Where the Kalman filter front-loads perception (solve the estimation Riccati
    once for the steady-state gain ``K∞``, then ``mean += K∞·prediction_error``),
    this front-loads action: solve the dual *control* Riccati once for ``L∞``,
    then ``action = -L∞·(mean − goal)``. Both gains are data-independent, both are
    computed at construction, and together they are LQG (see RESEARCH.md).

    The load-bearing claim (ADR-003) is that LQR *is* active inference here, not a
    substitute for it. For a fixed linear-Gaussian sensor the covariance recursion
    is control-independent, so Expected Free Energy's epistemic term is identical
    for every action and drops out of the argmin; EFE-minimising selection reduces
    to its pragmatic term, and the pragmatic term under a Gaussian preference is a
    quadratic cost whose optimum is exactly LQR. The epistemic term only re-enters
    once sensing depends on the state or action — out of scope for v0.1.

    The two cost matrices are named for the preference they encode, not by LQR's
    traditional ``Q``/``R`` — those letters already mean the noise covariances on
    the model (``dynamics_noise``/``sensor_noise``), the exact collision ADR-003
    warns about. The names are the same across the whole library: an ``Agent``
    hands these straight through to its controller.

    Args:
        model: The linear-Gaussian model to act in. Must carry a ``control``
            matrix — there is nothing to act with otherwise.
        goal_precision: How sharply the agent prefers the goal, an ``(n, n)``
            matrix. It is exactly the precision of the Gaussian preference centred
            at the goal — ``exp(−½(state−goal)ᵀ·goal_precision·(state−goal))`` —
            so heavier ``goal_precision`` buys a more aggressive controller.
            (LQR's ``Q``.)
        effort_penalty: How much action costs, a ``(p, p)`` matrix. Heavier
            ``effort_penalty`` buys a gentler controller. (LQR's ``R``.)
        tol: Absolute tolerance on successive cost-to-go iterates; convergence is
            declared when they stop moving by more than this.
        max_iter: Iteration cap before the Riccati recursion is declared to have
            failed to converge.

    Raises:
        ValueError: If the model has no ``control`` matrix, or a cost matrix does
            not match the state/action dimensions, is not symmetric, or fails its
            definiteness requirement (``goal_precision`` PSD, ``effort_penalty``
            PD).
        RuntimeError: If the control Riccati does not converge within ``max_iter``
            — typically because ``(dynamics, control)`` is not stabilisable.
    """

    def __init__(
        self,
        model: LinearGaussianModel,
        *,
        goal_precision: ArrayLike,
        effort_penalty: ArrayLike,
        tol: float = 1e-12,
        max_iter: int = 1000,
    ) -> None:
        if model.control is None:
            raise ValueError(
                "LQR needs an action channel: the model has no control matrix, "
                "so there is nothing to act with."
            )
        self.model = model
        self._goal_precision = np.asarray(goal_precision, dtype=float)
        self._effort_penalty = np.asarray(effort_penalty, dtype=float)

        n, p = model.n_states, model.n_controls
        if self._goal_precision.shape != (n, n):
            raise ValueError(
                f"goal_precision must be {n}x{n} to match the {n}-D state, "
                f"got shape {self._goal_precision.shape}"
            )
        if self._effort_penalty.shape != (p, p):
            raise ValueError(
                f"effort_penalty must be {p}x{p} to match the {p}-D action, "
                f"got shape {self._effort_penalty.shape}"
            )
        _validate_cost(self._goal_precision, "goal_precision", require_definite=False)
        _validate_cost(self._effort_penalty, "effort_penalty", require_definite=True)

        self._gain = self._converge_to_steady_state(tol, max_iter)

    @property
    def gain(self) -> NDArray[np.float64]:
        """The steady-state feedback gain L∞, shape (p, n)."""
        return self._gain

    def action(self, mean: NDArray[np.float64], goal: ArrayLike) -> NDArray[np.float64]:
        """The action that drives the estimated state toward ``goal``.

        One matrix-vector product, ``-L∞·(mean − goal)`` — all the work was
        front-loaded into ``L∞`` at construction, so there is no optimisation in
        the loop. The ``mean − goal`` shift turns the regulator (which drives its
        state to zero) into a controller that drives the state to ``goal``.

        Args:
            mean: The current belief mean — the best estimate of the state,
                shape ``(n,)``.
            goal: The state to steer toward, shape ``(n,)``. It must be an
                equilibrium the dynamics can hold at zero action; aim at a
                non-equilibrium and a steady-state offset is left behind.

        Returns:
            The action, shape ``(p,)``.

        Raises:
            ValueError: If ``goal`` is not a 1-D vector of length ``n``.
        """
        # self._gain : (p, n) L∞;  mean, goal : (n,);  returns (p,)
        goal = np.asarray(goal, dtype=float)
        if goal.shape != (self.model.n_states,):
            raise ValueError(
                f"goal must be a 1-D vector of length {self.model.n_states} "
                f"(the state dimension), got shape {goal.shape}"
            )
        return -self._gain @ (mean - goal)

    def _converge_to_steady_state(
        self, tol: float, max_iter: int
    ) -> NDArray[np.float64]:
        """Iterate the control Riccati recursion to its fixed point for ``L∞``.

        The exact dual of ``KalmanBackend._converge_to_steady_state``. The filter
        iterates a *covariance* forward until it stops moving; this iterates a
        *cost-to-go* — the matrix ``P`` of the quadratic value function
        ``V(state) = stateᵀ·P·state`` — until it stops moving. Starting from
        ``goal_precision``, each step applies Bellman's equation::

            P ← goal_precision + Aᵀ P A − (Aᵀ P B)(effort_penalty + Bᵀ P B)⁻¹(Bᵀ P A)

        "the cost from here = what I pay now + the cost from wherever the dynamics
        carry me, minus what acting optimally buys back." For a stabilisable
        ``(A, B)`` this converges to the unique fixed point ``P∞`` (the solution
        of the discrete algebraic Riccati equation), from which the steady-state
        gain follows::

            L∞ = (effort_penalty + Bᵀ P∞ B)⁻¹ (Bᵀ P∞ A)

        (A=dynamics, B=control.) The ``(effort_penalty + Bᵀ P B)`` term is solved
        against with ``np.linalg.solve`` rather than inverted explicitly, for the
        same numerical reason the filter solves against its innovation covariance.

        Returns:
            The steady-state gain ``L∞``, shape ``(p, n)``.

        Raises:
            RuntimeError: If the recursion has not converged within ``max_iter``.
        """
        dynamics = self.model.dynamics  # A  (n×n)
        assert self.model.control is not None  # guard lives in __init__; narrows type
        control = self.model.control  # B  (n×p)
        cost_to_go = self._goal_precision  # P, starting at the running state cost (n×n)

        for _ in range(max_iter):
            # Bellman's equation, one sweep.
            dyn_cost_ctrl = dynamics.T @ cost_to_go @ control  # Aᵀ P B  (n×p)
            # curvature of the action cost — the dual of the Kalman innovation
            # covariance S, the denominator the gain is solved against (p×p)
            inner = self._effort_penalty + control.T @ cost_to_go @ control
            next_cost_to_go = (
                self._goal_precision  # pay now
                + dynamics.T @ cost_to_go @ dynamics  # cost the dynamics carry forward
                - dyn_cost_ctrl
                @ np.linalg.solve(
                    inner, dyn_cost_ctrl.T
                )  # what optimal action buys back
            )

            if np.allclose(cost_to_go, next_cost_to_go, atol=tol, rtol=0.0):
                cost_to_go = next_cost_to_go
                break
            cost_to_go = next_cost_to_go
        else:
            raise RuntimeError(
                f"control Riccati did not converge in {max_iter} iterations; "
                "(dynamics, control) may not be stabilisable, so no steady-state "
                "gain exists."
            )

        inner = self._effort_penalty + control.T @ cost_to_go @ control
        return np.linalg.solve(inner, control.T @ cost_to_go @ dynamics)  # L∞  (p×n)
