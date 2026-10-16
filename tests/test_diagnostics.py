"""The sensor conditions, probed over a sampled reachable set."""

import jax.numpy as jnp
import numpy as np
import pytest

from _support import unit_prior
from cpomdp import (
    Belief,
    CallableSensor,
    Coupling,
    CouplingGraph,
    CouplingGraphBackend,
    FixedSensor,
    GaussianTransition,
    LinearGaussianModel,
    probe_model,
)
from cpomdp.diagnostics import (
    _host_logdet_pd,
    epistemic_value,
    is_positive_definite,
    loewner_order,
)
from cpomdp.dynamics import CallableProcessNoise

ACTIONS = [np.array([u]) for u in np.linspace(-3.0, 3.0, 13)]


def range_noise(x, params):
    """Sharpest at the origin, degrading with distance."""
    return jnp.atleast_2d(1.0 + x[0] ** 2)


def goes_indefinite(x, params):
    """Positive definite only while |x₀| < 1 — a violation the control can reach."""
    return jnp.atleast_2d(1.0 - x[0] ** 2)


def ignores_the_state(x, params):
    """Declared state-dependent, constant in fact."""
    return jnp.atleast_2d(2.0) + 0.0 * x[0]


def chain(noise_fn, *, control_matrix=1.0):
    model = LinearGaussianModel(
        dynamics_matrix=[[1.0]],
        observation_matrix=[[1.0]],
        dynamics_noise=[[1.0]],
        observation_noise=[[1.0]],
        prior=Belief([0.0], [[1.0]]),
        control_matrix=[[control_matrix]],
        observation_model=CallableSensor([[1.0]], noise_fn, None),
    )
    return model, Belief([0.0], [[1.0]])


class TestPositiveDefinite:
    @pytest.mark.parametrize(
        ("matrix", "expected"),
        [
            (np.diag([1.0, 2.0]), True),
            (np.diag([-1.0, 2.0]), False),
            (np.diag([0.0, 2.0]), False),
            (np.diag([-1.0, -2.0]), False),  # a positive determinant, still not one
            (np.array([[1.0, 2.0], [0.0, 1.0]]), False),  # not symmetric
        ],
    )
    def test_classifies(self, matrix, expected):
        assert is_positive_definite(matrix) is expected


class TestLogdetPd:
    def test_matches_the_closed_form(self):
        assert _host_logdet_pd(2.0 * np.eye(2)) == pytest.approx(np.log(4.0), abs=1e-12)
        assert _host_logdet_pd(np.diag([3.0, 0.5])) == pytest.approx(
            np.log(1.5), abs=1e-12
        )

    @pytest.mark.parametrize(
        "matrix",
        [
            np.diag([-1.0, -2.0]),  # determinant +2: what a sign shortcut misses
            np.diag([-1.0, 2.0]),
            np.diag([0.0, 2.0]),  # singular, so the log-determinant is -inf, not a NaN
            np.array([[1.0, 2.0], [0.0, 1.0]]),  # not symmetric
        ],
    )
    def test_non_pd_gives_nan(self, matrix):
        assert np.isnan(_host_logdet_pd(matrix))

    def test_the_sign_shortcut_it_replaces_is_fooled(self):
        # The premise. Without it the rejection above shows only that some matrix
        # returns NaN, not that the guard buys anything over reading slogdet directly.
        assert np.isfinite(np.linalg.slogdet(np.diag([-1.0, -2.0]))[1])


class TestEpistemicValue:
    def test_matches_the_closed_form(self):
        pred, c, r = np.array([[2.0]]), np.array([[1.0]]), np.array([[1.0]])
        assert epistemic_value(pred, c, r) == pytest.approx(0.5 * np.log(3.0))
        assert epistemic_value(pred, c, np.array([[5.0]])) == pytest.approx(
            0.5 * np.log(7.0 / 5.0)
        )

    def test_undefined_noise_gives_nan(self):
        pred, c = np.array([[2.0]]), np.array([[1.0]])
        assert np.isnan(epistemic_value(pred, c, np.array([[-1.0]])))


class TestLoewnerOrder:
    @pytest.mark.parametrize(
        ("a", "b", "expected"),
        [
            (np.diag([1.0, 1.0]), np.diag([2.0, 2.0]), "a<b"),
            (np.diag([2.0, 2.0]), np.diag([1.0, 1.0]), "b<a"),
            (np.diag([1.0, 1.0]), np.diag([1.0, 1.0]), "equal"),
            (np.diag([1.0, 3.0]), np.diag([3.0, 1.0]), "incomparable"),
            (np.diag([1.0, 1.0]), np.diag([1.0, 2.0]), "incomparable"),  # not strict
        ],
    )
    def test_compares(self, a, b, expected):
        assert loewner_order(a, b) == expected


class TestProbeFlatModel:
    def test_state_dependent_sensor_passes_every_condition(self):
        model, belief = chain(range_noise)
        report = probe_model(model, belief, ACTIONS)
        assert report.full_row_rank
        assert report.definite
        assert report.non_constant
        assert report.epistemic_varies
        assert report.loewner_comparable  # scalar noises are always ordered
        assert not report.flattens

    def test_fixed_sensor_flattens(self):
        model = LinearGaussianModel(
            dynamics_matrix=[[1.0]],
            observation_matrix=[[1.0]],
            dynamics_noise=[[1.0]],
            observation_noise=[[1.0]],
            prior=Belief([0.0], [[1.0]]),
            control_matrix=[[1.0]],
        )
        report = probe_model(model, Belief([0.0], [[1.0]]), ACTIONS)
        assert report.flattens
        assert not report.non_constant
        assert not report.process_non_constant
        assert report.process_noise_spread == 0.0
        assert not report.epistemic_varies
        assert report.epistemic_range[0] == pytest.approx(report.epistemic_range[1])

    def test_state_dependent_process_noise_reaches_the_predicted_covariance(self):
        # The EFE kernel adds Q at each predicted mean. Adding the plain field instead
        # reports an epistemic value the kernel never computes, and under a fixed
        # sensor it would read as constant.
        model = LinearGaussianModel(
            dynamics_matrix=[[1.0]],
            observation_matrix=[[1.0]],
            observation_noise=[[1.0]],
            prior=Belief([0.0], [[1.0]]),
            control_matrix=[[1.0]],
            dynamics_noise_model=CallableProcessNoise(range_noise, None),
        )
        report = probe_model(model, Belief([0.0], [[1.0]]), ACTIONS)
        # Σ⁻ = AΣAᵀ + Q(μ⁻) with μ⁻ = u, so 1 + (1 + u²).
        expected = [
            epistemic_value([[2.0 + u[0] ** 2]], [[1.0]], [[1.0]]) for u in ACTIONS
        ]
        assert report.epistemic_varies
        assert report.epistemic_range == pytest.approx((min(expected), max(expected)))
        # R never moved and Q did, so the recursion consults the action: not flat.
        assert not report.non_constant
        assert report.process_non_constant
        assert not report.flattens

    def test_a_process_noise_that_stops_being_a_covariance_is_refused(self):
        # Q(x) is checked at the origin only, when the model is built. A reachable mean
        # where it has gone indefinite would make S indefinite there and drop that
        # sample from the epistemic range with nothing said.
        model = LinearGaussianModel(
            dynamics_matrix=[[1.0]],
            observation_matrix=[[1.0]],
            observation_noise=[[1.0]],
            prior=Belief([0.0], [[0.01]]),
            control_matrix=[[1.0]],
            dynamics_noise_model=CallableProcessNoise(goes_indefinite, None),
        )
        with pytest.raises(ValueError, match=r"dynamics_noise_model\.noise_at"):
            probe_model(model, Belief([0.0], [[0.01]]), ACTIONS)

    def test_noise_the_control_cannot_reach_flattens(self):
        """Declared state-dependent, but no action moves the mean off the prior."""
        model, belief = chain(range_noise, control_matrix=0.0)
        report = probe_model(model, belief, ACTIONS)
        assert report.flattens
        assert report.noise_spread == pytest.approx(0.0)

    def test_noise_that_ignores_the_state_flattens(self):
        model, belief = chain(ignores_the_state)
        assert probe_model(model, belief, ACTIONS).flattens

    def test_reports_where_the_noise_stops_being_a_covariance(self):
        model, belief = chain(goes_indefinite)
        report = probe_model(model, belief, ACTIONS)
        assert not report.definite
        assert report.indefinite_at
        # every flagged mean is genuinely outside the band where 1 − x² stays positive
        assert all(abs(mu[0]) >= 1.0 for mu in report.indefinite_at)

    def test_needs_an_action_to_predict_under(self):
        model, belief = chain(range_noise)
        with pytest.raises(ValueError, match="at least one action"):
            probe_model(model, belief, [])

    def test_summary_names_every_condition(self):
        model, belief = chain(range_noise)
        text = probe_model(model, belief, ACTIONS).summary()
        for line in (
            "full row rank",
            "positive def",
            "R non-constant",
            "Q non-constant",
            "epistemic varies",
        ):
            assert line in text


class TestProbeGraphBackend:
    @staticmethod
    def build(*, alive):
        cue = (
            CallableSensor([[-1.0, 1.0]], range_noise, None)
            if alive
            else FixedSensor([[-1.0, 1.0]], observation_noise=[[2.0]])
        )
        graph = CouplingGraph(
            dims=(1, 2),
            couplings=(
                Coupling(
                    parent=0,
                    child=1,
                    coupling_matrix=[[0.0], [1.0]],
                    coupling_noise=[[0.5, 0.0], [0.0, 0.5]],
                ),
            ),
            observation_models={1: cue},
        )
        transitions = (
            GaussianTransition([[1.0]], dynamics_noise=[[0.1]]),
            GaussianTransition(
                [[1.0, 0.0], [0.0, 1.0]], dynamics_noise=[[0.1, 0.0], [0.0, 0.1]]
            ),
        )
        return CouplingGraphBackend(
            graph,
            transitions,
            prior=unit_prior(graph),
            control_matrix=[[0.0], [1.0], [0.0]],
        )

    @staticmethod
    def belief():
        return Belief(jnp.zeros(3), jnp.eye(3))

    def test_separates_a_live_sensor_from_a_frozen_one(self):
        live = probe_model(self.build(alive=True), self.belief(), ACTIONS)
        frozen = probe_model(self.build(alive=False), self.belief(), ACTIONS)
        assert live.non_constant
        assert live.epistemic_varies
        assert frozen.flattens
        assert not frozen.epistemic_varies

    def test_reports_a_redundant_channel(self):
        """Two channels reading the same row leave C short of full row rank."""
        graph = CouplingGraph(
            dims=(2,),
            couplings=(),
            observation_models={
                0: FixedSensor(
                    [[-1.0, 1.0], [-1.0, 1.0]],
                    observation_noise=[[2.0, 0.0], [0.0, 2.0]],
                )
            },
        )
        backend = CouplingGraphBackend(
            graph,
            (
                GaussianTransition(
                    [[1.0, 0.0], [0.0, 1.0]], dynamics_noise=[[0.1, 0.0], [0.0, 0.1]]
                ),
            ),
            prior=unit_prior(graph),
            control_matrix=[[1.0], [0.0]],
        )
        report = probe_model(backend, Belief(jnp.zeros(2), jnp.eye(2)), ACTIONS)
        assert not report.full_row_rank
        assert (report.rank, report.n_observations) == (1, 2)


def test_condition_numbers_reads_a_stack_and_refuses_anything_else():
    from cpomdp.diagnostics import _condition_numbers

    stack = np.array([[[4.0, 0.0], [0.0, 1.0]], [[1.0, 1.0], [1.0, 1.0]]])
    measured = _condition_numbers(stack)
    assert measured[0] == pytest.approx(4.0)
    assert not np.isfinite(measured[1]) or measured[1] > 1e15
    with pytest.raises(ValueError, match="stack of square matrices"):
        _condition_numbers(np.eye(2))
