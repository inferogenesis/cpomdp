"""The state-dependent-noise results, asserted against the library.

One test per published claim about what a state-dependent observation noise does to a
linear-Gaussian agent. The claims are about the agent's maintained Gaussian recursion —
the covariance it propagates and the epistemic value it scores — so every test here
works in those terms and nothing depends on a particular demonstration or figure.

The load-bearing one is `TestPinning`. Matching the agent's posterior covariance at a
single step identifies the noise covariance that produced it, uniquely, provided `C` has
full row rank. Everything else leans on that: the dual effect, the impossibility of a
fixed noise schedule, and the equivalence between a planning reduction and the absence
of a dual effect. `TestPinning::test_rank_deficient_C_does_not_pin` is the other side of
it — drop the rank condition and two different noises produce the same posterior, so the
identification really does need what it asks for.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from cpomdp import (
    Belief,
    CallableSensor,
    KalmanBackend,
    LinearGaussianModel,
    Preference,
)
from cpomdp.backends.kalman import _gain_and_posterior_cov
from cpomdp.efe import expected_free_energy

# --- the algebra the claims are stated in ----------------------------------------


def predicted_cov(prior_cov, dynamics_matrix, dynamics_noise_model):
    """Σ⁻ = A Σ Aᵀ + Q."""
    a = np.asarray(dynamics_matrix, dtype=float)
    return a @ np.asarray(prior_cov, dtype=float) @ a.T + np.asarray(
        dynamics_noise_model, dtype=float
    )


def posterior_cov(pred_cov, observation_matrix, observation_noise):
    """Σ⁺ = Σ⁻ − Σ⁻ Cᵀ S⁻¹ C Σ⁻, with S = C Σ⁻ Cᵀ + R."""
    cov = np.asarray(pred_cov, dtype=float)
    c = np.asarray(observation_matrix, dtype=float)
    r = np.asarray(observation_noise, dtype=float)
    s = c @ cov @ c.T + r
    return cov - cov @ c.T @ np.linalg.solve(s, c @ cov)


def epistemic(pred_cov, observation_matrix, observation_noise):
    """½ ln det S − ½ ln det R, in nats."""
    cov = np.asarray(pred_cov, dtype=float)
    c = np.asarray(observation_matrix, dtype=float)
    r = np.asarray(observation_noise, dtype=float)
    s = c @ cov @ c.T + r
    return 0.5 * (np.linalg.slogdet(s)[1] - np.linalg.slogdet(r)[1])


def pin_noise(pred_cov, observation_matrix, post_cov):
    """The noise covariance a posterior implies, recovered through the observation map.

    Equal posteriors are equal precisions, so `Cᵀ R⁻¹ C = Λ⁺ − Λ⁻`. Multiplying by `C`
    on the left and `Cᵀ` on the right turns that into `(C Cᵀ) R⁻¹ (C Cᵀ)`, which
    inverts whenever `C` has full row rank — and then `R` is determined.
    """
    c = np.asarray(observation_matrix, dtype=float)
    info = np.linalg.inv(post_cov) - np.linalg.inv(pred_cov)  # Cᵀ R⁻¹ C
    gram_inv = np.linalg.inv(c @ c.T)
    return np.linalg.inv(gram_inv @ c @ info @ c.T @ gram_inv)


# --- the model class: linear everything, additive control, R following the state ---

RANGE_NOISE_PARAMS = None


def range_noise(x, params):
    """R(x) = 1 + x₀² — a sensor that degrades as the state leaves the origin."""
    return jnp.atleast_2d(1.0 + x[0] ** 2)


def scalar_chain(
    *, dynamics_matrix=1.0, control_matrix=1.0, process=1.0, prior_var=1.0, mean=0.0
):
    """The scalar chain the worked example uses: A = B = C = Q = 1, R(x) = 1 + x²."""
    model = LinearGaussianModel(
        dynamics_matrix=[[dynamics_matrix]],
        observation_matrix=[[1.0]],
        dynamics_noise=[[process]],
        observation_noise=[[1.0]],
        prior=Belief([mean], [[prior_var]]),
        control_matrix=[[control_matrix]],
        observation_model=CallableSensor([[1.0]], range_noise, RANGE_NOISE_PARAMS),
    )
    return model, Belief([mean], [[prior_var]])


def noise_at(model, mean):
    """R evaluated at a predicted mean, through the model's own sensor."""
    return np.asarray(
        model.observation_model.linearize(jnp.asarray(mean))[1], dtype=float
    )


def predict_mean(model, belief, action):
    """μ⁻ = A μ + B u."""
    return np.asarray(model.A, dtype=float) @ np.asarray(
        belief.mean, dtype=float
    ) + np.asarray(model.control_matrix, dtype=float) @ np.atleast_1d(action)


class TestWorkedExample:
    """The published one-step numbers, reproduced through the shipped API."""

    @pytest.mark.parametrize(
        ("action", "noise", "gain", "info"),
        [
            (0.0, 1.0, 2.0 / 3.0, 0.5 * np.log(3.0)),
            (2.0, 5.0, 2.0 / 7.0, 0.5 * np.log(7.0 / 5.0)),
        ],
    )
    def test_one_step_numbers(self, action, noise, gain, info):
        model, belief = scalar_chain()
        mu_pred = predict_mean(model, belief, action)
        cov_pred = predicted_cov(belief.cov, model.A, model.Q)

        # The predicted variance is 2 under either action — only the mean moved.
        assert cov_pred == pytest.approx(2.0)
        assert noise_at(model, mu_pred) == pytest.approx(noise)

        k, _ = _gain_and_posterior_cov(
            model.A, model.C, model.Q, jnp.atleast_2d(noise), belief.cov
        )
        assert float(k[0, 0]) == pytest.approx(gain)

        efe = expected_free_energy(
            model, belief, jnp.array([action]), Preference([0.0], [[1.0]])
        )
        assert float(efe.epistemic) == pytest.approx(info)

    def test_staying_put_buys_about_three_times_the_information(self):
        model, belief = scalar_chain()
        pref = Preference([0.0], [[1.0]])
        stay = float(
            expected_free_energy(model, belief, jnp.array([0.0]), pref).epistemic
        )
        move = float(
            expected_free_energy(model, belief, jnp.array([2.0]), pref).epistemic
        )
        assert stay / move == pytest.approx(3.266, abs=1e-3)


class TestCollapse:
    """A fixed sensor earns every policy the same epistemic value."""

    def test_fixed_sensor_epistemic_is_action_invariant(self):
        model = LinearGaussianModel(
            dynamics_matrix=[[1.0]],
            observation_matrix=[[1.0]],
            dynamics_noise=[[1.0]],
            observation_noise=[[1.0]],
            prior=Belief([0.0], [[1.0]]),
            control_matrix=[[1.0]],
        )
        belief = Belief([0.0], [[1.0]])
        pref = Preference([0.0], [[1.0]])
        values = [
            float(expected_free_energy(model, belief, jnp.array([u]), pref).epistemic)
            for u in np.linspace(-5.0, 5.0, 21)
        ]
        assert np.ptp(values) == pytest.approx(0.0, abs=1e-12)


class TestPinning:
    """Matching a posterior covariance identifies the noise that produced it."""

    def test_recovers_the_noise_from_the_posterior(self):
        rng = np.random.default_rng(11)
        for n, m in [(1, 1), (3, 2), (4, 4), (5, 3)]:
            for _ in range(20):
                a = rng.normal(size=(n, n))
                pred = a @ a.T + n * np.eye(n)
                c = rng.normal(size=(m, n))
                if np.linalg.matrix_rank(c) < m:  # the condition the recovery needs
                    continue
                b = rng.normal(size=(m, m))
                r = b @ b.T + m * np.eye(m)
                recovered = pin_noise(pred, c, posterior_cov(pred, c, r))
                assert recovered == pytest.approx(r, abs=1e-8)

    def test_distinct_noises_give_distinct_posteriors(self):
        pred = np.diag([2.0, 3.0])
        c = np.eye(2)
        first = posterior_cov(pred, c, np.diag([1.0, 1.0]))
        second = posterior_cov(pred, c, np.diag([1.0, 4.0]))
        assert not np.allclose(first, second)

    def test_rank_deficient_C_does_not_pin(self):
        """Two channels reading the same row: the posterior fixes a scalar, not R.

        With both rows of `C` equal to the same `vᵀ`, `Cᵀ R⁻¹ C` depends on `R` only
        through the total `1ᵀ R⁻¹ 1`. Any two noises sharing that total are
        indistinguishable from the posterior — which is exactly what full row rank
        rules out.
        """
        c = np.array([[0.0, -1.0, 1.0], [0.0, -1.0, 1.0]])
        assert np.linalg.matrix_rank(c) < c.shape[0]

        first = np.diag([2.0, 2.0])  # 1ᵀR⁻¹1 = 1/2 + 1/2 = 1
        second = np.diag([4.0, 4.0 / 3.0])  # 1ᵀR⁻¹1 = 1/4 + 3/4 = 1
        assert not np.allclose(first, second)

        pred = np.eye(3)
        assert posterior_cov(pred, c, first) == pytest.approx(
            posterior_cov(pred, c, second), abs=1e-12
        )


class TestDualEffect:
    """The action reaches the posterior covariance, and the gain with it."""

    def test_posterior_covariance_depends_on_the_action(self):
        model, belief = scalar_chain()
        cov_pred = predicted_cov(belief.cov, model.A, model.Q)
        posts = {}
        for u in (0.0, 2.0):
            r = noise_at(model, predict_mean(model, belief, u))
            posts[u] = posterior_cov(cov_pred, model.C, r)
        assert not np.allclose(posts[0.0], posts[2.0])

    def test_gain_depends_on_the_action(self):
        model, belief = scalar_chain()
        gains = {}
        for u in (0.0, 2.0):
            r = noise_at(model, predict_mean(model, belief, u))
            k, _ = _gain_and_posterior_cov(model.A, model.C, model.Q, r, belief.cov)
            gains[u] = float(k[0, 0])
        assert gains[0.0] != pytest.approx(gains[2.0])

    def test_no_fixed_noise_schedule_reproduces_the_agent(self):
        """Two actions pin the same step to two different noises, so no schedule fits.

        A candidate reduction must name its noise covariance for step k before it knows
        the policy. Matching the agent under one action forces one value; matching under
        another forces a different one.
        """
        model, belief = scalar_chain()
        cov_pred = predicted_cov(belief.cov, model.A, model.Q)
        pinned = []
        for u in (0.0, 2.0):
            r = noise_at(model, predict_mean(model, belief, u))
            pinned.append(
                pin_noise(cov_pred, model.C, posterior_cov(cov_pred, model.C, r))
            )
        assert pinned[0] == pytest.approx(np.array([[1.0]]))
        assert pinned[1] == pytest.approx(np.array([[5.0]]))
        assert not np.allclose(pinned[0], pinned[1])

    def test_epistemic_value_is_not_constant(self):
        model, belief = scalar_chain()
        pref = Preference([0.0], [[1.0]])
        grid = np.linspace(-3.0, 3.0, 13)
        values = [
            float(expected_free_energy(model, belief, jnp.array([u]), pref).epistemic)
            for u in grid
        ]
        assert np.ptp(values) > 1e-6
        # R = 1 + x² is sharpest at the origin, so that is where the information peaks,
        # and it falls away symmetrically as the action drives the mean out.
        assert grid[int(np.argmax(values))] == pytest.approx(0.0)
        assert values == pytest.approx(values[::-1])


class TestScalarObservations:
    """With one observation channel the epistemic value tracks the variance directly."""

    def test_strictly_decreasing_in_the_noise_variance(self):
        pred = np.array([[2.0]])
        c = np.array([[1.0]])
        variances = np.linspace(0.05, 20.0, 200)
        values = [epistemic(pred, c, np.array([[v]])) for v in variances]
        assert np.all(np.diff(values) < 0)

    def test_two_reachable_variances_separate_the_information(self):
        model, belief = scalar_chain()
        cov_pred = predicted_cov(belief.cov, model.A, model.Q)
        low = noise_at(model, predict_mean(model, belief, 0.0))
        high = noise_at(model, predict_mean(model, belief, 2.0))
        assert low < high
        assert epistemic(cov_pred, model.C, low) > epistemic(cov_pred, model.C, high)


class TestZeroInnovation:
    """Fed the observations it predicts, the filter's means walk the open-loop path."""

    def test_filtering_means_follow_the_open_loop_trajectory(self):
        model, belief = scalar_chain(mean=0.3)
        backend = KalmanBackend(model)
        actions = [0.5, -1.2, 2.0, 0.0]

        open_loop, mean = [], np.asarray(belief.mean, dtype=float)
        for u in actions:  # observations marginalised: the mean never gets corrected
            mean = np.asarray(model.A, dtype=float) @ mean + np.asarray(
                model.control_matrix, dtype=float
            ) @ np.atleast_1d(u)
            open_loop.append(mean.copy())

        current = belief
        for u, planned in zip(actions, open_loop, strict=True):
            reading = np.asarray(model.C, dtype=float) @ planned  # o = C μ⁻
            current = backend.infer_states(
                jnp.asarray(reading), current, jnp.array([u])
            )
            assert np.asarray(current.mean) == pytest.approx(planned, abs=1e-12)


class TestPlanningReductionEquivalence:
    """A planning reduction exists exactly when the action cannot reach Σ⁺."""

    def test_fixed_noise_leaves_the_covariance_policy_independent(self):
        model = LinearGaussianModel(
            dynamics_matrix=[[1.0]],
            observation_matrix=[[1.0]],
            dynamics_noise=[[1.0]],
            observation_noise=[[1.0]],
            prior=Belief([0.0], [[1.0]]),
            control_matrix=[[1.0]],
        )
        belief = Belief([0.0], [[1.0]])
        cov_pred = predicted_cov(belief.cov, model.A, model.Q)
        posts = [posterior_cov(cov_pred, model.C, model.R) for _ in (0.0, 2.0)]
        assert posts[0] == pytest.approx(posts[1])
        # policy-independent covariances -> the pinned schedule is one constant
        pinned = {float(pin_noise(cov_pred, model.C, p)[0, 0]) for p in posts}
        assert len(pinned) == 1

    def test_reachable_state_dependence_destroys_it(self):
        model, belief = scalar_chain()
        cov_pred = predicted_cov(belief.cov, model.A, model.Q)
        posts = [
            posterior_cov(
                cov_pred, model.C, noise_at(model, predict_mean(model, belief, u))
            )
            for u in (0.0, 2.0)
        ]
        assert not np.allclose(posts[0], posts[1])

    def test_noise_that_no_action_reaches_still_flattens(self):
        """A state-dependent R the control cannot move is a constant in every way.

        With no control authority the predicted mean is the same under every action, so
        the noise is pinned to one value and a fixed schedule reproduces the agent.
        """
        model, belief = scalar_chain(control_matrix=0.0)
        cov_pred = predicted_cov(belief.cov, model.A, model.Q)
        pinned = {
            float(
                pin_noise(
                    cov_pred,
                    model.C,
                    posterior_cov(
                        cov_pred,
                        model.C,
                        noise_at(model, predict_mean(model, belief, u)),
                    ),
                )[0, 0]
            )
            for u in (-3.0, 0.0, 5.0)
        }
        assert len(pinned) == 1


class TestDualEffectInvisibleToEpistemicValue:
    """A covariance can move with the action while the epistemic value does not.

    Two channels whose noises trade off along a contour of the determinant: the
    posterior covariance tracks the action, yet every action scores the same. The
    separation between "the covariance moves" and "the objective notices" is real, and
    this is the construction that shows it.
    """

    @staticmethod
    def paired_noise(first):
        """The partner variance holding `(1 + 1/r)(1 + 1/s)` at 3."""
        return (first + 1.0) / (2.0 * first - 1.0)

    def test_epistemic_value_is_constant_while_the_posterior_moves(self):
        pred = np.eye(2)  # A = 0, Q = Σ₀ = I leaves Σ⁻ at the identity for every policy
        c = np.eye(2)
        posts, values = [], []
        for r in np.linspace(1.0, 2.0, 9):
            s = self.paired_noise(r)
            assert 1.0 <= s <= 2.0  # both stay positive definite
            noise = np.diag([r, s])
            posts.append(posterior_cov(pred, c, noise))
            values.append(epistemic(pred, c, noise))

        assert np.ptp(values) == pytest.approx(0.0, abs=1e-12)
        assert values[0] == pytest.approx(0.5 * np.log(3.0))
        assert not np.allclose(posts[0], posts[-1])

    def test_the_posterior_takes_the_published_form(self):
        pred = np.eye(2)
        c = np.eye(2)
        r = 1.4
        s = self.paired_noise(r)
        expected = np.diag([r / (1.0 + r), s / (1.0 + s)])
        assert posterior_cov(pred, c, np.diag([r, s])) == pytest.approx(expected)

    def test_the_pinning_still_rules_out_every_fixed_schedule(self):
        pred = np.eye(2)
        c = np.eye(2)
        pinned = [
            pin_noise(
                pred, c, posterior_cov(pred, c, np.diag([r, self.paired_noise(r)]))
            )
            for r in (1.2, 1.8)
        ]
        assert not np.allclose(pinned[0], pinned[1])


class TestObservationDrivenMean:
    """Policy-independence is not observation-independence.

    A model with no control authority has one open-loop trajectory, so a reduction that
    only has to match the planning covariances exists trivially. It still cannot match
    the agent on every observation sequence: the filtering mean moves with the data, and
    the noise is evaluated wherever the mean ends up.
    """

    def test_data_moves_the_pinned_noise_when_the_control_cannot(self):
        model, belief = scalar_chain(control_matrix=0.0, mean=0.0)
        backend = KalmanBackend(model)

        pinned = []
        for reading in (0.0, 6.0):
            post = backend.infer_states(jnp.array([reading]), belief, jnp.array([0.0]))
            # the next step's noise is read at wherever this observation left the mean
            nxt = predict_mean(model, post, 0.0)
            pinned.append(float(noise_at(model, nxt)[0, 0]))

        assert pinned[0] != pytest.approx(pinned[1])
