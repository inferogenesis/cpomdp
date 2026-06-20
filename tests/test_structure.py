"""Spec for ModelStructure (the declarable factor/blanket/channel metadata).

A1 covers the *type* and its pytree-aux threading: construction, inspection, value
equality/hashing, and — the load-bearing part — that the structure rides in the model's
pytree **aux_data** (no array leaves, hashable so jit can key on it, byte-identical
arithmetic). Validation (partition + conditional-independence) is specced separately in
A3/A4. Byte-identical arithmetic across the kernels is A2.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from cpomdp.backends.kalman import KalmanBackend
from cpomdp.efe import expected_free_energy
from cpomdp.selection import Preference
from cpomdp.structure import ModelStructure
from cpomdp.types import Belief, LinearGaussianModel


def _kwargs(**over):
    """Valid 2-state / 1-observation model kwargs; ``over`` swaps fields.

    Unannotated (like ``_valid_kwargs`` in test_types.py) so its return is inferred
    gradually — a wrong-typed override (e.g. ``structure="x"``) then reaches the runtime
    guard instead of being rejected by ty at the ``**`` unpack.
    """
    kwargs = {
        "dynamics": [[1.0, 0.1], [0.0, 1.0]],
        "sensor_model": [[1.0, 0.0]],
        "dynamics_noise": [[0.1, 0.0], [0.0, 0.1]],
        "sensor_noise": [[1.0]],
        "prior": Belief(mean=[0.0, 0.0], cov=[[1.0, 0.0], [0.0, 1.0]]),
    }
    kwargs.update(over)
    return kwargs


def _model(**over) -> LinearGaussianModel:
    """A valid 2-state / 1-observation model; ``over`` swaps individual fields."""
    return LinearGaussianModel(**_kwargs(**over))


class TestModelStructureType:
    def test_from_dicts_matches_direct_tuple_construction(self):
        from_dict = ModelStructure.from_dicts(factors={"pos": [0], "vel": [1]})
        from_tuple = ModelStructure(factors=(("pos", (0,)), ("vel", (1,))))
        assert from_dict == from_tuple

    def test_factor_lookup_and_names(self):
        s = ModelStructure.from_dicts(factors={"pos": [0, 1], "vel": [2, 3]})
        assert s.factor("pos") == (0, 1)
        assert s.factor_names == ("pos", "vel")
        with pytest.raises(KeyError):
            s.factor("nope")

    def test_role_of_returns_label_or_none(self):
        s = ModelStructure.from_dicts(roles={"internal": [0, 1], "external": [2]})
        assert s.role_of(0) == "internal"
        assert s.role_of(2) == "external"
        assert s.role_of(3) is None  # untyped index

    def test_channel_lookup(self):
        s = ModelStructure.from_dicts(channels={"gradient": [0]})
        assert s.channel("gradient") == (0,)

    def test_summary_is_a_string_naming_declarations(self):
        s = ModelStructure.from_dicts(
            factors={"pos": [0, 1]}, channels={"gradient": [0]}
        )
        out = s.summary()
        assert isinstance(out, str)
        assert "pos" in out
        assert "gradient" in out

    def test_value_equality_and_hashing(self):
        s1 = ModelStructure.from_dicts(factors={"pos": [0, 1]})
        s2 = ModelStructure.from_dicts(factors={"pos": [0, 1]})
        s3 = ModelStructure.from_dicts(factors={"pos": [0]})
        assert s1 == s2
        assert hash(s1) == hash(s2)
        assert s1 != s3
        assert isinstance(hash(s1), int)  # MUST hash — it is jit's static aux key

    def test_list_indices_normalize_to_a_hashable_tuple(self):
        # Indices passed as a list must be frozen to a tuple — a list field would be
        # unhashable and break the jit-aux contract (outer form stays a tuple of pairs).
        s = ModelStructure(factors=(("pos", [0, 1]),))
        assert s.factors == (("pos", (0, 1)),)
        assert isinstance(hash(s), int)


class TestStructureOnModel:
    def test_model_defaults_to_no_structure(self):
        assert _model().structure is None

    def test_model_carries_the_structure_it_was_given(self):
        s = ModelStructure.from_dicts(factors={"pos": [0], "vel": [1]})
        assert _model(structure=s).structure is s

    def test_rejects_a_non_structure(self):
        with pytest.raises(TypeError, match="ModelStructure"):
            _model(structure="not a structure")

    def test_structure_adds_no_leaves_but_changes_the_treedef(self):
        s = ModelStructure.from_dicts(factors={"pos": [0], "vel": [1]})
        plain = _model()
        structured = _model(structure=s)
        # Static aux ⇒ contributes no array leaves (same leaves as the plain model) ...
        assert len(jax.tree_util.tree_leaves(structured)) == len(
            jax.tree_util.tree_leaves(plain)
        )
        # ... but it IS part of the treedef, so jit keys on it.
        assert jax.tree_util.tree_structure(structured) != jax.tree_util.tree_structure(
            plain
        )

    def test_structured_model_round_trips_through_flatten_unflatten(self):
        s = ModelStructure.from_dicts(
            factors={"pos": [0], "vel": [1]}, roles={"internal": [0, 1]}
        )
        m = _model(structure=s)
        leaves, treedef = jax.tree_util.tree_flatten(m)
        restored = jax.tree_util.tree_unflatten(treedef, leaves)
        assert restored.structure == s
        np.testing.assert_array_equal(restored.dynamics, m.dynamics)

    def test_differing_structure_gives_a_differing_treedef(self):
        # Locks the documented recompile caveat: two models identical but for their
        # structure are different pytrees, so jit re-specializes when they are swapped.
        m1 = _model(structure=ModelStructure.from_dicts(factors={"a": [0], "b": [1]}))
        m2 = _model(structure=ModelStructure.from_dicts(factors={"x": [0], "y": [1]}))
        assert jax.tree_util.tree_structure(m1) != jax.tree_util.tree_structure(m2)

    def test_jit_over_a_structured_model_survives(self):
        # jit hashes the treedef (which carries the structure aux) for its cache key —
        # a dict/list field would raise "unhashable" right here. A trace proves it runs.
        s = ModelStructure.from_dicts(factors={"pos": [0], "vel": [1]})
        out = jax.jit(lambda model: jnp.trace(model.dynamics))(_model(structure=s))
        assert float(out) == pytest.approx(2.0)


class TestStructureIsInertArithmetic:
    """structure is pure metadata: it must not perturb any kernel by a single ULP.

    The byte-identical (``assert_array_equal``, not ``allclose``) guarantee that
    declaring structure changes nothing the engine computes — RFC-003 section 4.5.
    """

    _STRUCT = ModelStructure.from_dicts(factors={"a": [0], "b": [1]})

    def test_expected_free_energy_is_byte_identical(self):
        plain = _model(control=[[1.0], [0.0]])
        structured = _model(control=[[1.0], [0.0]], structure=self._STRUCT)
        pref = Preference(goal=[0.5], precision=[[2.0]])
        action = jnp.array([0.3])
        g0, parts0 = expected_free_energy(plain, plain.prior, action, pref)
        g1, parts1 = expected_free_energy(structured, structured.prior, action, pref)
        np.testing.assert_array_equal(g0, g1)
        np.testing.assert_array_equal(parts0["pragmatic"], parts1["pragmatic"])
        np.testing.assert_array_equal(parts0["epistemic"], parts1["epistemic"])

    def test_kalman_step_is_byte_identical(self):
        plain = _model(control=[[1.0], [0.0]])
        structured = _model(control=[[1.0], [0.0]], structure=self._STRUCT)
        action = jnp.array([0.3])
        obs = jnp.array([0.7])
        b0 = KalmanBackend(plain).infer_states(obs, plain.prior, action)
        b1 = KalmanBackend(structured).infer_states(obs, structured.prior, action)
        np.testing.assert_array_equal(b0.mean, b1.mean)
        np.testing.assert_array_equal(b0.cov, b1.cov)
