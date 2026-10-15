"""Exhaustive EFE search over a declared finite action set — the enumerated family.

`EnumeratedEfeSearch` enumerates every length-H sequence of a `FiniteActionSet` (all
`|A|^H` of them, *varying* sequences, not the constant-action policies `EFESelector`
tiles), scores each with `policy_efe`, and returns the argmin policy and the full `G`
vector. Because the set is finite and fully enumerated the search **decides** its
universal (Prover 3 · enumeration), so it carries the `PROVED` warrant — distinct from
the continuous grid's `CORROBORATED` (a sample of a continuum). A
`CompletenessCertificate` records `expected = |A|^H` against the `visited` count and
asserts they match.

The locks:

- `TestFiniteActionSet`: the declared, versioned set validates its shape and version.
- `TestCompletenessCertificate`: the enumeration covers the full cartesian product, and
  the certificate reports expected == visited under `PROVED`. A shortfall under `PROVED`
  does not construct; the honest label for a partial enumeration is `CORROBORATED`.
- `TestEnumeratedSearch`: the `G` vector and argmin match a plain itertools + policy_efe
  reference loop. That is an orchestration check; the EFE arithmetic is validated in
  policy_efe tests. On the beacon model a **varying** sequence strictly beats every
  constant policy by a declared margin. That is a genuine sequential optimum the
  constant-action family cannot express.
- `TestWarrantSeam`: the two families self-describe — enumerated `PROVED`, grid
  `CORROBORATED` — in distinct vocabulary (standing rule 6).
- `TestCostAndTransforms`: `cost_per_plan == |A|^H * H`; the search survives `jit` and
  handles a multi-dimensional (p>1) action set the grid selector rejects.

Imports the enumeration symbols directly, so until they land this module is
collection-red — the `ImportError` naming `EnumeratedEfeSearch` is the build cue.
"""

import itertools

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from cpomdp.efe import policy_efe
from cpomdp.enumeration import (
    CompletenessCertificate,
    EnumeratedEfeSearch,
    FiniteActionSet,
    IncompleteEnumerationError,
)
from cpomdp.observation import CallableSensor
from cpomdp.selection import EFESelector, Preference
from cpomdp.types import Belief, LinearGaussianModel
from warrantlib import Warrant

# The beacon fixture's best varying policy beats its best constant policy by ~27 nats,
# so a strict win of at least 1 nat is cleared ~27x and sits ~15 orders above float64
# noise. This is compared best-varying vs best-constant over the whole set, so it does
# not depend on argmin's tie-break or the action ordering. See warrant_numbers.md.
VARYING_WIN_MARGIN = 1.0


# --- fixtures ----------------------------------------------------------------------
def _model():
    """A plain fixed-sensor model, p = 1."""
    return LinearGaussianModel(
        dynamics_matrix=[[1.0, 0.1], [0.0, 1.0]],
        observation_matrix=[[1.0, 0.0]],
        dynamics_noise=[[0.1, 0.0], [0.0, 0.1]],
        observation_noise=[[0.5]],
        prior=Belief(mean=[0.0, 0.0], cov=[[1.0, 0.0], [0.0, 1.0]]),
        control_matrix=[[0.0], [1.0]],
    )


def _beacon_model():
    """1-D R(x): the action moves position directly, observation noise is low near +2.

    Action maps straight to position (control [[1.0]], sensor [[1.0]]), so every action
    reaches an observation. Observation noise falls near the beacon at +2, so
    reaching that
    low-noise region buys epistemic value the goal-observation at 0 does not. The
    optimum is to get to the beacon and hold, which no single held action can do.
    """
    sensor = CallableSensor(
        observation_matrix=[[1.0]],
        noise_fn=lambda x, p: jnp.array(
            [[p["base"] + p["sharp"] * (x[0] - p["beacon"]) ** 2]]
        ),
        noise_params={
            "base": jnp.array(0.05),
            "sharp": jnp.array(4.0),
            "beacon": jnp.array(2.0),
        },
    )
    return LinearGaussianModel(
        dynamics_matrix=[[1.0]],
        observation_matrix=[[1.0]],
        dynamics_noise=[[0.05]],
        observation_noise=[[0.5]],
        prior=Belief(mean=[0.0], cov=[[1.0]]),
        control_matrix=[[1.0]],
        observation_model=sensor,
    )


def _belief():
    return Belief(mean=[0.0, 0.0], cov=[[1.0, 0.0], [0.0, 1.0]])


def _pref(goal=0.0):
    return Preference(goal=[goal], precision=[[2.0]])


def _finite_set(values, version="v1"):
    """A p=1 FiniteActionSet from a list of scalar actions."""
    return FiniteActionSet([[v] for v in values], version=version)


def _oracle(model, belief, preference, action_set, horizon):
    """Independent enumeration + scoring: itertools product, a plain policy_efe loop."""
    actions = [tuple(a) for a in np.asarray(action_set.actions)]
    seqs = list(itertools.product(actions, repeat=horizon))
    policies = np.array(seqs, dtype=float)  # (N, H, p)
    g = np.array(
        [
            float(policy_efe(model, belief, jnp.asarray(pol), preference)[0])
            for pol in policies
        ]
    )
    return policies, g


# --- FiniteActionSet ---------------------------------------------------------------
class TestFiniteActionSet:
    def test_shape_and_version(self):
        s = _finite_set([-1.0, 0.0, 1.0], version="moves-v1")
        assert s.size == 3
        assert s.actions.shape == (3, 1)
        assert s.version == "moves-v1"

    def test_rejects_single_action(self):
        with pytest.raises(ValueError, match="at least 2"):
            FiniteActionSet([[0.0]], version="v1")

    def test_rejects_ragged_or_1d(self):
        with pytest.raises(ValueError, match="2-D"):
            FiniteActionSet([0.0, 1.0], version="v1")  # 1-D, not (A, p)

    def test_rejects_empty_version(self):
        with pytest.raises(ValueError, match="version"):
            FiniteActionSet([[0.0], [1.0]], version="")

    @pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
    def test_rejects_a_non_finite_action(self, bad):
        # A NaN action scores NaN on every policy that takes it, and a search over the
        # set would still certify coverage of it.
        with pytest.raises(ValueError, match=r"finite.*rows: 1\."):
            FiniteActionSet([[0.0], [bad]], version="v1")

    def test_a_duplicate_counts_toward_the_size(self):
        # Kept on purpose: the tie-break tests declare duplicates to force ties.
        assert FiniteActionSet([[1.0], [1.0], [0.0]], version="v1").size == 3


# --- CompletenessCertificate -------------------------------------------------------
class TestCompletenessCertificate:
    def test_enumeration_covers_cartesian_product(self):
        model, action_set = _model(), _finite_set([-1.0, 0.0, 1.0])
        search = EnumeratedEfeSearch(model, action_set, horizon=2)
        oracle_policies, _ = _oracle(model, _belief(), _pref(), action_set, 2)
        # same set of policies, and every one distinct
        got = np.asarray(search.policies)
        assert got.shape == oracle_policies.shape == (9, 2, 1)
        np.testing.assert_allclose(
            np.sort(got, axis=0), np.sort(oracle_policies, axis=0)
        )
        assert len({tuple(p.flatten()) for p in got}) == 9

    def test_certificate_is_proved_and_complete(self):
        search = EnumeratedEfeSearch(_model(), _finite_set([-1.0, 0.0, 1.0]), horizon=3)
        cert = search.evaluate(_belief(), _pref()).certificate
        assert cert.expected == 27  # |A|^H = 3^3
        assert cert.visited == 27
        assert cert.complete
        assert cert.warrant is Warrant.PROVED
        assert "PROVED" in str(cert)
        assert "27" in str(cert)

    def test_a_proved_certificate_must_be_complete(self):
        # `expected != visited` with a PROVED warrant is a certificate certifying its
        # own failure. It used to construct and read `complete = False`, which put the
        # contradiction one attribute access away from anyone who did not look.
        with pytest.raises(ValueError, match="complete"):
            CompletenessCertificate(
                expected=9,
                visited=8,
                warrant=Warrant.PROVED,
                action_set_size=3,
                horizon=2,
                action_set_version="v1",
            )

    def test_a_proved_certificate_must_quantify_over_the_declared_set(self):
        # The other half. `expected` is supplied rather than derived, so a count that
        # does not match |A|^H means the certificate is naming a set nobody enumerated
        # — and `expected` alone cannot say, since 81 is 9^2 and 3^4 both.
        with pytest.raises(ValueError, match="declared set"):
            CompletenessCertificate(
                expected=81,
                visited=81,
                warrant=Warrant.PROVED,
                action_set_size=3,
                horizon=2,
                action_set_version="v1",
            )

    def test_an_incomplete_enumeration_is_corroborated(self):
        # The honest label for a partial enumeration: it sampled the set.
        cert = CompletenessCertificate(
            expected=9,
            visited=8,
            warrant=Warrant.CORROBORATED,
            action_set_size=3,
            horizon=2,
            action_set_version="v1",
        )
        assert not cert.complete

    def test_incomplete_enumeration_error_exists(self):
        # The construction guard raises this if visited != expected; itertools makes
        # that impossible in practice, so we pin the type is a distinct exception.
        assert issubclass(IncompleteEnumerationError, Exception)


# --- the search --------------------------------------------------------------------
class TestEnumeratedSearch:
    def test_g_vector_matches_oracle(self):
        model, action_set = _model(), _finite_set([-1.0, 0.0, 1.0])
        search = EnumeratedEfeSearch(model, action_set, horizon=2)
        result = search.evaluate(_belief(), _pref())
        _, g_ref = _oracle(model, _belief(), _pref(), action_set, 2)
        assert result.g.shape == (9,)
        np.testing.assert_allclose(result.g, g_ref, atol=1e-9)

    def test_best_policy_is_the_argmin_member(self):
        model, action_set = _model(), _finite_set([-1.0, 0.0, 1.0])
        search = EnumeratedEfeSearch(model, action_set, horizon=2)
        result = search.evaluate(_belief(), _pref())
        policies, g_ref = _oracle(model, _belief(), _pref(), action_set, 2)
        np.testing.assert_allclose(result.best_policy, policies[int(np.argmin(g_ref))])

    def test_varying_sequence_strictly_beats_every_constant(self):
        # Compare the best VARYING policy against the best CONSTANT one over the whole
        # enumerated set, so the result does not depend on argmin's tie-break or the
        # action ordering. The optimum reaches the low-noise beacon and holds there,
        # which no single held action can do.
        model, action_set = _beacon_model(), _finite_set([-2.0, 0.0, 2.0])
        belief = Belief(mean=[0.0], cov=[[1.0]])
        search = EnumeratedEfeSearch(model, action_set, horizon=2)
        result = search.evaluate(belief, _pref(0.0))
        g = np.asarray(result.g)
        policies = np.asarray(search.policies)  # (N, H, p)
        is_constant = np.array(
            [len({tuple(step) for step in pol}) == 1 for pol in policies]
        )
        best_varying = g[~is_constant].min()
        best_constant = g[is_constant].min()
        assert best_constant - best_varying > VARYING_WIN_MARGIN, (
            f"best constant G {best_constant:.3f} vs best varying G {best_varying:.3f}"
        )
        # the global optimum is therefore a genuinely varying sequence
        assert not is_constant[int(np.argmin(g))]


# --- the warrant seam --------------------------------------------------------------
class TestWarrantSeam:
    def test_two_distinct_warrant_classes(self):
        assert Warrant.PROVED is not Warrant.CORROBORATED
        assert Warrant.PROVED.value == "PROVED"
        assert Warrant.CORROBORATED.value == "CORROBORATED"

    def test_enumerated_search_is_proved(self):
        search = EnumeratedEfeSearch(_model(), _finite_set([-1.0, 1.0]), horizon=2)
        assert search.warrant is Warrant.PROVED

    def test_grid_selector_is_corroborated(self):
        selector = EFESelector(_model(), n_candidates=5, action_bounds=(-1.0, 1.0))
        assert selector.warrant is Warrant.CORROBORATED


# --- cost and transforms -----------------------------------------------------------
class TestCostAndTransforms:
    def test_cost_per_plan_is_exponential(self):
        search = EnumeratedEfeSearch(_model(), _finite_set([-1.0, 0.0, 1.0]), horizon=3)
        assert search.n_policies == 27  # |A|^H
        assert search.cost_per_plan == 27 * 3  # |A|^H * H
        # A search runs per plan. `cost_per_cycle` is a driver's, per agent step.
        assert not hasattr(search, "cost_per_cycle")

    def test_jit_agrees_with_eager(self):
        search = EnumeratedEfeSearch(_model(), _finite_set([-1.0, 0.0, 1.0]), horizon=2)
        belief, pref = _belief(), _pref()
        eager = search.evaluate(belief, pref).g
        jitted = jax.jit(lambda b: search.evaluate(b, pref).g)(belief)
        np.testing.assert_allclose(jitted, eager, atol=1e-12)

    def test_handles_multi_dimensional_actions(self):
        # p = 2 — a set the grid EFESelector rejects (it is 1-D only).
        model = LinearGaussianModel(
            dynamics_matrix=[[1.0, 0.0], [0.0, 1.0]],
            observation_matrix=[[1.0, 0.0]],
            dynamics_noise=[[0.1, 0.0], [0.0, 0.1]],
            observation_noise=[[0.5]],
            prior=Belief(mean=[0.0, 0.0], cov=[[1.0, 0.0], [0.0, 1.0]]),
            control_matrix=[[1.0, 0.0], [0.0, 1.0]],  # (n=2, p=2)
        )
        action_set = FiniteActionSet(
            [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], version="2d-v1"
        )
        search = EnumeratedEfeSearch(model, action_set, horizon=2)
        result = search.evaluate(_belief(), _pref())
        assert search.n_policies == 9  # 3^2
        assert result.best_policy.shape == (2, 2)  # (H, p)


# --- validation --------------------------------------------------------------------
class TestValidation:
    def test_rejects_control_free_model(self):
        model = LinearGaussianModel(
            dynamics_matrix=[[1.0]],
            observation_matrix=[[1.0]],
            dynamics_noise=[[0.1]],
            observation_noise=[[0.5]],
            prior=Belief(mean=[0.0], cov=[[1.0]]),
            control_matrix=None,
        )
        with pytest.raises(ValueError, match="control"):
            EnumeratedEfeSearch(
                model, FiniteActionSet([[0.0], [1.0]], version="v1"), horizon=2
            )

    def test_rejects_action_dim_mismatch(self):
        # model p = 1 but the action set is p = 2.
        bad = FiniteActionSet([[0.0, 0.0], [1.0, 1.0]], version="v1")
        with pytest.raises(ValueError, match="action"):
            EnumeratedEfeSearch(_model(), bad, horizon=2)

    def test_rejects_horizon_below_one(self):
        with pytest.raises(ValueError, match="horizon"):
            EnumeratedEfeSearch(_model(), _finite_set([-1.0, 1.0]), horizon=0)
