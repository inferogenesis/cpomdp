"""The chunked enumerator — the same decision as the front-loaded path, in blocks.

`ChunkedEfeSearch` enumerates the same `A^H` set as `EnumeratedEfeSearch` and earns the
same `PROVED` warrant, but never materialises the policy set or the score vector. At
`17^7` a score vector alone would cost 3.28 GB, so the front-loaded path's ceiling is a
memory limit rather than a statement about the model, and this removes it.

What the locks below are for. Chunking is only admissible if it changes the residency
and nothing else, so the load-bearing assertion is not that the chunked path is
self-consistent — it is that it agrees with the path that produced the published
numbers, bit for bit.

- `TestCrossPathIdentity`: chunked against front-loaded on the same fixture, asserting
  equal `G` under `==` rather than `allclose`, equal argmin index, and an equal policy.
  This is what licenses re-reporting a published number from the new path.
- `TestChunkSizeInvariance`: the answer does not move with the block size, including
  blocks of one, blocks exactly `N`, and blocks larger than `N`.
- `TestTieBreak`: with a duplicated action set every minimum is a many-way tie, and
  both paths must return the *globally lowest* index. Order-invariance of `min` does
  not cover this, so it is asserted rather than argued.
- `TestCompleteness`: `visited` is a loop-carried count of unpadded lanes, not an
  array's length, so a padding bug shows here. Run at an `N` deliberately indivisible
  by the chunk.
- `TestReducerSeam`: an injected fold reproduces a NumPy count over the front-loaded
  score vector, which is how evidence beyond the argmin is obtained without a vector.
- `TestGuards`: the preconditions, including the index dtype's ceiling.
- `TestInterchangeable`: one verb, one shape of answer and one certificate rule on both
  paths, so swapping one search for the other changes no call site.
"""

import subprocess
import sys
import textwrap
from dataclasses import fields

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from cpomdp.enumeration import (
    ChunkedEfeSearch,
    EnumeratedEfeSearch,
    EnumeratedSearchResult,
    ExhaustiveEfeSearch,
    FiniteActionSet,
    SearchResult,
)
from cpomdp.observation import CallableSensor
from cpomdp.selection import Preference
from cpomdp.types import Belief, LinearGaussianModel
from warrantlib import Warrant

# Block sizes spanning every shape the loop can take: one lane per block, a block that
# divides N, blocks that do not, the last block exactly full, and a single block wider
# than the whole enumeration.
CHUNKS = [1, 2, 5, 7, 26, 27, 40]

# The same span against the 64-policy tie fixture.
TIE_CHUNKS = [1, 2, 3, 5, 16, 63, 64, 65]

# Blocks that always leave 27 policies with a padded tail.
PADDING_CHUNKS = [5, 7, 26, 40]


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
    """1-D `R(x)`: observation noise falls near +2, so the epistemic term varies
    with action.

    The fixed-sensor model above would exercise only the pragmatic path. Here the
    epistemic term is live, which is the branch the crossover result runs on.
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


def _beacon_belief():
    return Belief(mean=[0.0], cov=[[1.0]])


def _pref(goal=0.0):
    return Preference(goal=[goal], precision=[[2.0]])


def _finite_set(values, version="v1"):
    return FiniteActionSet([[v] for v in values], version=version)


def _probe_peak_gib(horizon):
    """Peak resident memory of one chunked run, measured in a fresh interpreter.

    `ru_maxrss` is a high-water mark for the whole process, so read in-process it
    reports whatever an earlier fixture allocated. A subprocess makes the
    number about this run.
    """
    probe = textwrap.dedent(
        """
        import resource, sys
        import jax; jax.config.update("jax_enable_x64", True)
        sys.path.insert(0, "tests")
        from test_chunked_enumeration import (
            TestAtScale, _beacon_model, _beacon_belief, _pref,
        )
        from cpomdp.enumeration import ChunkedEfeSearch

        ChunkedEfeSearch(
            _beacon_model(),
            TestAtScale._refined(0.5),
            horizon=int(sys.argv[1]),
            chunk=8192,
        ).evaluate(_beacon_belief(), _pref())
        print(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2)
        """
    )
    finished = subprocess.run(
        [sys.executable, "-c", probe, str(horizon)],
        capture_output=True,
        text=True,
        check=True,
    )
    return float(finished.stdout.split()[-1])


def _front_loaded(model, action_set, horizon, belief, preference):
    """`(g, best_index, best_policy)` from the path that produced the published runs."""
    search = EnumeratedEfeSearch(model, action_set, horizon=horizon)
    result = search.evaluate(belief, preference)
    g = np.asarray(result.g)
    best = int(np.argmin(np.where(np.isnan(g), np.inf, g)))
    return g, best, np.asarray(result.best_policy)


# --- the load-bearing one ------------------------------------------------------------
class TestCrossPathIdentity:
    @pytest.mark.parametrize(
        ("model_fn", "belief_fn", "values", "horizon"),
        [
            (_model, _belief, [-1.0, 0.0, 1.0], 3),
            (_beacon_model, _beacon_belief, [0.0, 1.0, 2.0], 4),
        ],
    )
    def test_chunked_reproduces_the_front_loaded_answer(
        self, model_fn, belief_fn, values, horizon
    ):
        # `==` rather than `allclose`. The two paths run the identical per-policy
        # arithmetic on the identical device in the identical dtype, so a one-ULP
        # disagreement would mean the policies differ, not that the rounding does.
        model, action_set = model_fn(), _finite_set(values)
        belief, preference = belief_fn(), _pref()
        g, best, best_policy = _front_loaded(
            model, action_set, horizon, belief, preference
        )

        search = ChunkedEfeSearch(model, action_set, horizon=horizon, chunk=7)
        result = search.evaluate(belief, preference)

        assert result.best_index == best
        assert float(result.best_g) == g[best]
        np.testing.assert_array_equal(np.asarray(result.best_policy), best_policy)

    def test_indices_name_the_same_policies_on_both_paths(self):
        # Agreement on the argmin would be luck if the two paths disagreed about which
        # policy an index names. The chunked path decodes a base-|A| numeral; the
        # front-loaded one takes `itertools.product`'s order. Pin that they coincide
        # across the whole enumeration, not only at the winner.
        model, action_set = _model(), _finite_set([-1.0, 0.0, 1.0])
        front = EnumeratedEfeSearch(model, action_set, horizon=3)
        search = ChunkedEfeSearch(model, action_set, horizon=3, chunk=5)
        decoded = search.policies_at(jnp.arange(front.n_policies))
        np.testing.assert_array_equal(np.asarray(decoded), np.asarray(front.policies))


class TestChunkSizeInvariance:
    @pytest.mark.parametrize("chunk", CHUNKS)
    def test_the_answer_does_not_move_with_the_block_size(self, chunk):
        model, action_set = _model(), _finite_set([-1.0, 0.0, 1.0])
        belief, preference = _belief(), _pref()
        g, best, best_policy = _front_loaded(model, action_set, 3, belief, preference)

        result = ChunkedEfeSearch(model, action_set, horizon=3, chunk=chunk).evaluate(
            belief, preference
        )

        assert result.best_index == best
        assert float(result.best_g) == g[best]
        assert result.certificate.visited == g.shape[0]
        np.testing.assert_array_equal(np.asarray(result.best_policy), best_policy)


class TestTieBreak:
    @staticmethod
    def _tied_set():
        """Every action duplicated, so every score is a many-way tie by construction."""
        return FiniteActionSet([[-1.0], [-1.0], [1.0], [1.0]], version="tie-v1")

    def test_the_fixture_really_ties(self):
        # A tie-break test on a fixture with no ties passes for the wrong reason.
        g, _, _ = _front_loaded(_model(), self._tied_set(), 3, _belief(), _pref())
        assert np.count_nonzero(g == g.min()) > 1

    @pytest.mark.parametrize("chunk", TIE_CHUNKS)
    def test_a_tie_resolves_to_the_globally_lowest_index(self, chunk):
        # `jnp.argmin` over the whole vector takes the lowest index. Blocks arrive in
        # increasing order and the combine updates on a strict `<`, so the first
        # occurrence wins and the two paths agree. A non-strict `<=` would return the
        # last block's copy instead, silently, on exactly this fixture.
        action_set = self._tied_set()
        g, best, _ = _front_loaded(_model(), action_set, 3, _belief(), _pref())

        search = ChunkedEfeSearch(_model(), action_set, horizon=3, chunk=chunk)
        result = search.evaluate(_belief(), _pref())

        assert best == int(np.flatnonzero(g == g.min()).min())
        assert result.best_index == best


class TestCompleteness:
    @pytest.mark.parametrize("chunk", PADDING_CHUNKS)
    def test_visited_counts_unpadded_lanes_only(self, chunk):
        # 27 policies against these blocks always leaves a padded tail. `visited` is
        # carried by the loop rather than read off a shape, so an unmasked pad would
        # over-count here and the certificate would certify work never done.
        search = ChunkedEfeSearch(
            _model(), _finite_set([-1.0, 0.0, 1.0]), horizon=3, chunk=chunk
        )
        assert search.n_policies % chunk != 0 or chunk >= search.n_policies
        result = search.evaluate(_belief(), _pref())
        assert result.certificate.visited == 27
        assert result.certificate.complete
        assert result.certificate.warrant is Warrant.PROVED

    def test_the_certificate_names_the_set_it_decided_over(self):
        result = ChunkedEfeSearch(
            _model(),
            _finite_set([-1.0, 0.0, 1.0], version="named-v1"),
            horizon=3,
            chunk=7,
        ).evaluate(_belief(), _pref())
        certificate = result.certificate
        assert certificate.action_set_version == "named-v1"
        assert certificate.action_set_size == 3
        assert certificate.horizon == 3
        assert certificate.domain_declared
        assert "named-v1" in str(certificate)

    def test_a_padded_lane_cannot_win(self):
        # Padded lanes decode to policy 0 so the kernel sees a defined input, then are
        # masked to +inf. Were the mask dropped, policy 0 would be re-scored in the
        # tail block and could take the argmin off a later, genuinely lower index.
        model, action_set = _beacon_model(), _finite_set([0.0, 1.0, 2.0])
        g, best, _ = _front_loaded(model, action_set, 4, _beacon_belief(), _pref())
        assert best != 0  # otherwise a leaked pad would agree by accident
        result = ChunkedEfeSearch(model, action_set, horizon=4, chunk=7).evaluate(
            _beacon_belief(), _pref()
        )
        assert result.best_index == best
        assert float(result.best_g) == g[best]


class TestReducerSeam:
    class _BelowBaseline:
        """Counts policies scoring strictly below a baseline, folded per block.

        The shape the refinement and extension axes need: a count over all `|A|^H`
        obtained without a score vector. Padded and NaN lanes arrive as `+inf`, so
        neither can be counted below a finite baseline and the fold needs no mask.
        """

        def __init__(self, baseline: float) -> None:
            self.baseline = baseline

        def init(self):
            return jnp.asarray(0)

        def update(self, carry, g, index, valid):
            return carry + jnp.sum(g < self.baseline)

    @pytest.mark.parametrize("chunk", CHUNKS)
    def test_an_injected_fold_matches_a_numpy_count(self, chunk):
        model, action_set = _model(), _finite_set([-1.0, 0.0, 1.0])
        g, _, _ = _front_loaded(model, action_set, 3, _belief(), _pref())
        baseline = float(np.median(g))

        search = ChunkedEfeSearch(model, action_set, horizon=3, chunk=chunk)
        _, extra = search._reduce(_belief(), _pref(), self._BelowBaseline(baseline))

        assert int(extra) == int(np.count_nonzero(g < baseline))

    def test_no_reducer_leaves_the_carry_empty(self):
        search = ChunkedEfeSearch(
            _model(), _finite_set([-1.0, 0.0, 1.0]), horizon=3, chunk=7
        )
        assert search._reduce(_belief(), _pref(), None)[1] is None


class TestGuards:
    def test_a_chunk_below_one_is_rejected(self):
        with pytest.raises(ValueError, match="chunk must be >= 1"):
            ChunkedEfeSearch(_model(), _finite_set([-1.0, 1.0]), horizon=2, chunk=0)

    def test_a_control_free_model_is_rejected(self):
        model = LinearGaussianModel(
            dynamics_matrix=[[1.0]],
            observation_matrix=[[1.0]],
            dynamics_noise=[[0.1]],
            observation_noise=[[0.5]],
            prior=Belief(mean=[0.0], cov=[[1.0]]),
        )
        with pytest.raises(ValueError, match="control matrix"):
            ChunkedEfeSearch(model, _finite_set([-1.0, 1.0]), horizon=2)

    def test_an_action_dimension_mismatch_is_rejected(self):
        wide = FiniteActionSet([[-1.0, 0.0], [1.0, 0.0]], version="p2-v1")
        with pytest.raises(ValueError, match="action set is p=2"):
            ChunkedEfeSearch(_model(), wide, horizon=2)

    def test_a_horizon_below_one_is_rejected(self):
        with pytest.raises(ValueError, match="horizon must be >= 1"):
            ChunkedEfeSearch(_model(), _finite_set([-1.0, 1.0]), horizon=0)

    def test_an_enumeration_past_the_index_ceiling_is_rejected(self):
        # 2^64 policies is not a run anyone would launch. The guard matters because an
        # index that wraps enumerates the wrong policies and still certifies, so the
        # failure is silent and the certificate endorses it.
        with pytest.raises(ValueError, match="index dtype"):
            ChunkedEfeSearch(_model(), _finite_set([-1.0, 1.0]), horizon=64)


class TestInterchangeable:
    """Swapping one search for the other changes the residency and no call site."""

    @staticmethod
    def _both(horizon=3):
        model, action_set = _beacon_model(), _finite_set([0.0, 1.0, 2.0])
        return (
            EnumeratedEfeSearch(model, action_set, horizon=horizon),
            ChunkedEfeSearch(model, action_set, horizon=horizon, chunk=5),
        )

    def test_one_verb_returns_the_same_decision(self):
        front, chunked = self._both()
        a = front.evaluate(_beacon_belief(), _pref())
        b = chunked.evaluate(_beacon_belief(), _pref())
        np.testing.assert_array_equal(np.asarray(a.best_policy), b.best_policy)
        assert float(a.best_g) == float(b.best_g)
        assert int(a.best_index) == int(b.best_index)
        assert a.certificate == b.certificate

    def test_the_front_loaded_result_is_a_search_result_plus_the_score_vector(self):
        shared = ("best_policy", "best_g", "best_index", "certificate")
        assert tuple(field.name for field in fields(SearchResult)) == shared
        assert issubclass(EnumeratedSearchResult, SearchResult)
        front_loaded_fields = tuple(
            field.name for field in fields(EnumeratedSearchResult)
        )
        assert front_loaded_fields == (*shared, "g")

    def test_each_search_returns_its_own_result_type(self):
        front, chunked = self._both()
        assert type(front.evaluate(_beacon_belief(), _pref())) is EnumeratedSearchResult
        assert type(chunked.evaluate(_beacon_belief(), _pref())) is SearchResult

    @pytest.mark.parametrize("which", [0, 1])
    def test_a_result_is_read_by_name_alone(self, which):
        # A position that named one field on one search and another field on the other
        # would make the swap silently wrong. No position names anything.
        result = self._both()[which].evaluate(_beacon_belief(), _pref())
        with pytest.raises(TypeError):
            iter(result)
        with pytest.raises(TypeError):
            result[1]

    def test_the_front_loaded_best_is_read_off_its_own_vector(self):
        front, _ = self._both()
        result = front.evaluate(_beacon_belief(), _pref())
        assert float(result.best_g) == float(result.g[result.best_index])
        assert result.certificate.complete

    def test_both_are_the_one_kind(self):
        front, chunked = self._both()
        assert isinstance(front, ExhaustiveEfeSearch)
        assert isinstance(chunked, ExhaustiveEfeSearch)

    @pytest.mark.parametrize(
        "name",
        [
            "evaluate",
            "policies_at",
            "n_policies",
            "horizon",
            "cost_per_plan",
            "warrant",
        ],
    )
    def test_the_shared_surface_is_declared_on_the_kind(self, name):
        assert hasattr(ExhaustiveEfeSearch, name)

    def test_the_certificate_lives_on_the_result_alone(self):
        # The chunked count is produced by the loop, so its certificate cannot exist
        # before a run. One rule for both paths: the decision carries its evidence.
        front, chunked = self._both()
        assert not hasattr(front, "certificate")
        assert not hasattr(chunked, "certificate")
        assert not hasattr(chunked, "reduce")


class TestNoFiniteScore:
    """Once R(x) stops being a covariance every policy scores NaN, and none is best.

    `R(x) = 1 - x0^2` is negative wherever `|x0| > 1`, so from a belief at 2 every
    policy in the set scores NaN. The certificate still holds, since it certifies that
    every policy was scored. What changes is that no policy wins.
    """

    @staticmethod
    def _model():
        sensor = CallableSensor(
            observation_matrix=[[1.0]],
            noise_fn=lambda x, p: jnp.array([[p["r"] - x[0] ** 2]]),
            noise_params={"r": jnp.array(1.0)},
        )
        return LinearGaussianModel(
            dynamics_matrix=[[1.0]],
            observation_matrix=[[1.0]],
            dynamics_noise=[[0.05]],
            prior=Belief(mean=[0.0], cov=[[1.0]]),
            control_matrix=[[1.0]],
            observation_model=sensor,
        )

    @pytest.mark.parametrize("search_type", [EnumeratedEfeSearch, ChunkedEfeSearch])
    def test_no_policy_is_returned_as_best(self, search_type):
        search = search_type(self._model(), _finite_set([0.0, 0.5]), horizon=2)
        result = search.evaluate(Belief(mean=[2.0], cov=[[0.1]]), _pref())
        assert np.all(np.isnan(np.asarray(result.best_policy)))
        assert np.isnan(float(result.best_g))
        assert int(result.best_index) == -1
        assert result.certificate.complete

    def test_the_premise_every_score_is_nan(self):
        search = EnumeratedEfeSearch(self._model(), _finite_set([0.0, 0.5]), horizon=2)
        result = search.evaluate(Belief(mean=[2.0], cov=[[0.1]]), _pref())
        assert np.all(np.isnan(np.asarray(result.g)))

    @pytest.mark.parametrize("search_type", [EnumeratedEfeSearch, ChunkedEfeSearch])
    @pytest.mark.parametrize("index", [-1, 4])
    def test_an_index_outside_the_enumeration_decodes_to_nothing(
        self, search_type, index
    ):
        # -1 is the no-winner index. Decoded as a base-|A| numeral it would wrap to a
        # real policy, so the decoder refuses it, and anything past the end.
        search = search_type(self._model(), _finite_set([0.0, 0.5]), horizon=2)
        with pytest.raises(ValueError, match=r"outside 0\.\.3"):
            search.policies_at(jnp.asarray([0, index]))

    @pytest.mark.parametrize("search_type", [EnumeratedEfeSearch, ChunkedEfeSearch])
    def test_compiled_an_index_outside_decodes_to_a_nan_policy(self, search_type):
        # A compiled decode cannot raise. It returns the NaN policy a result with no
        # winner carries, never the real policy -1 would wrap to.
        search = search_type(self._model(), _finite_set([0.0, 0.5]), horizon=2)
        decoded = jax.jit(search.policies_at)(jnp.asarray([3, -1, 4]))
        np.testing.assert_array_equal(decoded[0], search.policies_at(3))
        assert np.all(np.isnan(np.asarray(decoded[1:])))

    @pytest.mark.parametrize("search_type", [EnumeratedEfeSearch, ChunkedEfeSearch])
    def test_the_decode_maps_over_indices(self, search_type):
        search = search_type(self._model(), _finite_set([0.0, 0.5]), horizon=2)
        np.testing.assert_array_equal(
            jax.vmap(search.policies_at)(jnp.arange(4)),
            search.policies_at(jnp.arange(4)),
        )

    @pytest.mark.parametrize("search_type", [EnumeratedEfeSearch, ChunkedEfeSearch])
    def test_one_finite_score_is_still_the_best(self, search_type):
        # From 0.5, a step of 2 lands where R is negative and a step of -0.5 does not.
        search = search_type(self._model(), _finite_set([2.0, -0.5]), horizon=1)
        result = search.evaluate(Belief(mean=[0.5], cov=[[0.1]]), _pref())
        assert int(result.best_index) == 1
        assert np.isfinite(float(result.best_g))
        np.testing.assert_array_equal(np.asarray(result.best_policy), [[-0.5]])


class TestAtScale:
    """The cheap fixtures above all fit, so none of them can show residency.

    Slowest of these is 4.4s at 0.55 GiB,
    so they belong on the pull-request path. The cross-path identity assert is what
    licenses re-reporting a published number, and gating it on merge would leave that
    licence unchecked.

    The headline `9^7` comparison is a one-off recorded in the warrant ledger: its
    front-loaded half costs 5.4 GiB and six minutes. What runs here is that comparison
    one horizon down, on a cheaper model, sensitive to the same class of break.
    """

    @staticmethod
    def _refined(step):
        """The declared step-refined action set over ``[-2, 2]``."""
        count = round(4 / step)
        return FiniteActionSet(
            [[round(-2.0 + k * step, 10)] for k in range(count + 1)],
            version=f"refine-{step:g}",
        )

    def test_paths_agree_at_nine_to_the_sixth(self):
        # 531,441 policies. The front-loaded half needs ~1 GiB, which is the point:
        # this is the largest cell that runs both ways inside a merge budget.
        model, action_set = _beacon_model(), self._refined(0.5)
        belief, preference = _beacon_belief(), _pref()
        g, best, best_policy = _front_loaded(model, action_set, 6, belief, preference)
        assert g.shape[0] == 9**6

        result = ChunkedEfeSearch(model, action_set, horizon=6, chunk=32768).evaluate(
            belief, preference
        )

        assert result.best_index == best
        assert float(result.best_g) == g[best]
        assert result.certificate.visited == 9**6
        np.testing.assert_array_equal(np.asarray(result.best_policy), best_policy)

    def test_the_step_quarter_set_runs_at_a_low_horizon(self):
        # `17^7` is the declared step-0.25 cell and no check will ever reproduce it.
        # `17^4` is 83,521 policies of the *same set*, so the decode, the block loop
        # and the certificate are exercised on merge even though the headline is not.
        # A cell nothing re-runs is unverified at that size; unverifiable is worse.
        action_set = self._refined(0.25)
        assert action_set.size == 17
        result = ChunkedEfeSearch(
            _beacon_model(), action_set, horizon=4, chunk=8192
        ).evaluate(_beacon_belief(), _pref())
        assert result.certificate.visited == 17**4
        assert result.certificate.complete
        assert result.certificate.action_set_version == "refine-0.25"

    def test_residency_tracks_the_block_rather_than_the_enumeration(self):
        # Peak RSS is a high-water mark, so it has to be read in a fresh process or an
        # earlier fixture's allocation is what gets measured. Two chunked runs 729x
        # apart in N at one block size: were residency following the enumeration, the
        # larger would show it.
        peaks = [_probe_peak_gib(horizon) for horizon in (3, 6)]
        small, large = peaks
        # A generous 1.5x admits interpreter variation without admitting O(N).
        assert large < 1.5 * small, f"peak grew with N, not with chunk: {peaks}"


class TestCostAndWarrant:
    def test_cost_and_block_count_are_reported(self):
        search = ChunkedEfeSearch(
            _model(), _finite_set([-1.0, 0.0, 1.0]), horizon=3, chunk=7
        )
        assert search.n_policies == 27
        assert search.cost_per_plan == 27 * 3  # |A|^H * H, RFC-001
        assert not hasattr(search, "cost_per_cycle")
        assert search.n_blocks == 4  # ceil(27 / 7), the last one padded
        assert search.chunk == 7

    def test_the_warrant_is_proved(self):
        # Same decision as the front-loaded path, so the same prover class. The block
        # size is a residency choice and carries no evidential weight.
        search = ChunkedEfeSearch(_model(), _finite_set([-1.0, 1.0]), horizon=2)
        assert search.warrant is Warrant.PROVED
