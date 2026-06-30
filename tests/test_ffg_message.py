"""CanonicalGaussian: the FFG message algebra (v0.4 Phase 1, DECISIONS.md ADR-012).

Oracle-first (per the build plan): these tests are written before
``cpomdp.ffg.message.CanonicalGaussian`` exists, so the whole file is RED until
that module lands. Each oracle here is computed independently of the
implementation under test — plain NumPy, a different formula path, or a
literal dense slice — never by importing or re-deriving the kernel's own math.

The spec these tests pin (mirrors ``Belief`` in types.py for shape/validation):

- ``CanonicalGaussian(precision, potential)`` stores Λ = Σ⁻¹ and h = Σ⁻¹μ.
  ``precision`` PSD (not necessarily definite — an uncombined message can be
  rank-deficient); ``potential`` 1-D, shape-matched; both finite.
- ``.to_moment() -> (mean, cov)`` requires `precision` positive-DEFINITE
  (raises clearly on a singular precision, never returns inf/NaN silently).
- ``__add__`` is the factor product: exact elementwise addition of
  ``(precision, potential)`` — no inversion, ever.
- ``.marginalize(over)`` eliminates the indices in `over` via the Schur
  complement, returning the canonical Gaussian on the remaining indices in
  ascending order.
- Registered JAX pytree (leaves: precision, potential) — jit/vmap/grad-clean.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from cpomdp.ffg.message import CanonicalGaussian


def _spd(rng, n):
    """A random n x n symmetric positive-definite matrix (independent of the
    implementation: plain NumPy, A @ A.T + jitter)."""
    a = rng.standard_normal((n, n))
    return a @ a.T + n * np.eye(n)


def _random_system(rng, n):
    """A random (mean, cov) pair and its canonical-form (precision, potential)."""
    cov = _spd(rng, n)
    mean = rng.standard_normal(n)
    precision = np.linalg.inv(cov)
    potential = precision @ mean
    return mean, cov, precision, potential


# --- Construction / validation -----------------------------------------------


def _valid_kwargs(**overrides):
    kwargs = {
        "precision": [[2.0, 0.0], [0.0, 3.0]],
        "potential": [1.0, -1.0],
    }
    kwargs.update(overrides)
    return kwargs


class TestConstruction:
    def test_valid_message_stores_what_you_passed(self):
        m = CanonicalGaussian(**_valid_kwargs())
        np.testing.assert_array_equal(m.precision, [[2.0, 0.0], [0.0, 3.0]])
        np.testing.assert_array_equal(m.potential, [1.0, -1.0])

    def test_coerces_lists_to_float64_arrays(self):
        m = CanonicalGaussian(precision=[[1, 0], [0, 1]], potential=[0, 1])
        assert isinstance(m.potential, jax.Array)
        assert m.potential.dtype == jnp.float64

    def test_ndim_reports_dimension(self):
        assert CanonicalGaussian(**_valid_kwargs()).ndim == 2
        assert CanonicalGaussian(precision=[[1.0]], potential=[0.0]).ndim == 1

    def test_rejects_potential_not_1D(self):
        with pytest.raises(ValueError, match="1-D"):
            CanonicalGaussian(precision=[[1.0]], potential=[[0.0]])

    def test_rejects_nonfinite_potential(self):
        with pytest.raises(ValueError, match="finite"):
            CanonicalGaussian(**_valid_kwargs(potential=[1.0, float("nan")]))

    def test_rejects_precision_not_2D(self):
        with pytest.raises(ValueError, match="2-D"):
            CanonicalGaussian(precision=[1.0, 0.0], potential=[0.0, 0.0])

    def test_rejects_shape_mismatch(self):
        with pytest.raises(ValueError, match="match"):
            CanonicalGaussian(precision=[[1.0]], potential=[0.0, 0.0])

    def test_rejects_asymmetric_precision(self):
        with pytest.raises(ValueError, match="symmetric"):
            CanonicalGaussian(precision=[[1.0, 0.2], [0.9, 1.0]], potential=[0.0, 0.0])

    def test_rejects_indefinite_precision(self):
        # eigenvalues (-1, 3): a genuinely impossible precision matrix.
        with pytest.raises(ValueError, match="positive-semi-definite"):
            CanonicalGaussian(precision=[[1.0, 2.0], [2.0, 1.0]], potential=[0.0, 0.0])

    def test_accepts_singular_precision(self):
        # A not-yet-combined message: zero precision is a legitimate "no
        # information yet" potential, not an error at construction time.
        CanonicalGaussian(precision=[[0.0, 0.0], [0.0, 0.0]], potential=[0.0, 0.0])


# --- Round-trip to moment form -------------------------------------------------


class TestRoundTrip:
    @pytest.mark.parametrize("n", [1, 2, 3, 5])
    def test_round_trips_to_moment_form(self, n):
        rng = np.random.default_rng(0)
        for _ in range(5):
            mean, cov, precision, potential = _random_system(rng, n)
            out_mean, out_cov = CanonicalGaussian(precision, potential).to_moment()
            np.testing.assert_allclose(out_mean, mean, atol=1e-8)
            np.testing.assert_allclose(out_cov, cov, atol=1e-8)

    def test_to_moment_rejects_singular_precision(self):
        m = CanonicalGaussian(precision=[[0.0, 0.0], [0.0, 0.0]], potential=[0.0, 0.0])
        with pytest.raises(ValueError, match="positive-definite"):
            m.to_moment()


# --- Factor product (__add__) --------------------------------------------------


class TestFactorProduct:
    def test_add_is_exact_elementwise_sum(self):
        a = CanonicalGaussian(precision=[[1.0, 0.0], [0.0, 2.0]], potential=[1.0, 0.0])
        b = CanonicalGaussian(precision=[[3.0, 0.0], [0.0, 1.0]], potential=[0.0, 2.0])
        c = a + b
        np.testing.assert_array_equal(c.precision, [[4.0, 0.0], [0.0, 3.0]])
        np.testing.assert_array_equal(c.potential, [1.0, 2.0])

    def test_add_is_commutative(self):
        rng = np.random.default_rng(1)
        _, _, p1, h1 = _random_system(rng, 3)
        _, _, p2, h2 = _random_system(rng, 3)
        a, b = CanonicalGaussian(p1, h1), CanonicalGaussian(p2, h2)
        left, right = a + b, b + a
        np.testing.assert_array_equal(left.precision, right.precision)
        np.testing.assert_array_equal(left.potential, right.potential)

    def test_add_is_associative(self):
        rng = np.random.default_rng(2)
        _, _, p1, h1 = _random_system(rng, 3)
        _, _, p2, h2 = _random_system(rng, 3)
        _, _, p3, h3 = _random_system(rng, 3)
        a, b, c = (
            CanonicalGaussian(p1, h1),
            CanonicalGaussian(p2, h2),
            CanonicalGaussian(p3, h3),
        )
        left, right = (a + b) + c, a + (b + c)
        np.testing.assert_allclose(left.precision, right.precision, rtol=1e-12)
        np.testing.assert_allclose(left.potential, right.potential, rtol=1e-12)

    def test_add_rejects_shape_mismatch(self):
        a = CanonicalGaussian(precision=[[1.0]], potential=[0.0])
        b = CanonicalGaussian(precision=[[1.0, 0.0], [0.0, 1.0]], potential=[0.0, 0.0])
        with pytest.raises(ValueError, match="shape"):
            a + b

    def test_add_never_inverts(self, monkeypatch):
        # The whole point of canonical form: the factor product is addition,
        # never an inversion. Patch both inv and solve to explode, then prove
        # __add__ still works.
        def boom(*args, **kwargs):
            raise AssertionError("__add__ must not invert anything")

        monkeypatch.setattr(jnp.linalg, "inv", boom)
        monkeypatch.setattr(jnp.linalg, "solve", boom)
        a = CanonicalGaussian(precision=[[1.0, 0.0], [0.0, 2.0]], potential=[1.0, 0.0])
        b = CanonicalGaussian(precision=[[3.0, 0.0], [0.0, 1.0]], potential=[0.0, 2.0])
        c = a + b
        np.testing.assert_array_equal(c.precision, [[4.0, 0.0], [0.0, 3.0]])


# --- Marginalize (Schur complement vs. a literal dense slice) -----------------


class TestMarginalize:
    @pytest.mark.parametrize(
        ("n", "over"),
        [
            (2, [1]),
            (3, [0]),
            (3, [1]),
            (3, [0, 2]),
            (4, [1, 3]),
            (5, [0, 2, 4]),
        ],
    )
    def test_matches_dense_reference(self, n, over):
        # The dense reference does NOT use the Schur complement at all: the
        # marginal of a subset of jointly-Gaussian variables is just the
        # corresponding sub-vector/sub-matrix of (mean, cov) directly. That's
        # a completely independent code path from the canonical-form math.
        rng = np.random.default_rng(3)
        mean, cov, precision, potential = _random_system(rng, n)
        keep = sorted(set(range(n)) - set(over))

        joint = CanonicalGaussian(precision, potential)
        out_mean, out_cov = joint.marginalize(over).to_moment()

        np.testing.assert_allclose(out_mean, mean[keep], atol=1e-7)
        np.testing.assert_allclose(out_cov, cov[np.ix_(keep, keep)], atol=1e-7)

    def test_keeps_ascending_order_regardless_of_over_order(self):
        rng = np.random.default_rng(4)
        mean, _cov, precision, potential = _random_system(rng, 4)
        # over given out of order / with the kept indices interleaved.
        joint = CanonicalGaussian(precision, potential)
        out_mean, _ = joint.marginalize([3, 0]).to_moment()
        np.testing.assert_allclose(out_mean, mean[[1, 2]], atol=1e-7)

    def test_marginalize_rejects_singular_eliminated_block(self):
        # precision's [1,1] block (the one being eliminated) is singular.
        precision = [[2.0, 0.0], [0.0, 0.0]]
        joint = CanonicalGaussian(precision=precision, potential=[1.0, 0.0])
        with pytest.raises(ValueError, match="positive-definite"):
            joint.marginalize([1])


# --- JAX pytree / jit / vmap / grad smoke tests --------------------------------


class TestPytreeTransforms:
    def test_flattens_to_two_leaves(self):
        m = CanonicalGaussian(**_valid_kwargs())
        assert len(jax.tree_util.tree_leaves(m)) == 2

    def test_round_trips_flatten_unflatten(self):
        m = CanonicalGaussian(**_valid_kwargs())
        leaves, treedef = jax.tree_util.tree_flatten(m)
        restored = jax.tree_util.tree_unflatten(treedef, leaves)
        assert isinstance(restored, CanonicalGaussian)
        np.testing.assert_array_equal(restored.precision, m.precision)
        np.testing.assert_array_equal(restored.potential, m.potential)

    def test_unflatten_does_not_revalidate(self):
        # JAX rebuilds from leaves under jit/vmap/grad, where leaves are
        # tracers; an asymmetric precision (which __init__ would reject) must
        # pass straight through unflatten.
        asymmetric = jnp.array([[1.0, 0.9], [0.2, 1.0]])
        treedef = jax.tree_util.tree_structure(CanonicalGaussian(**_valid_kwargs()))
        rebuilt = jax.tree_util.tree_unflatten(treedef, [asymmetric, jnp.zeros(2)])
        np.testing.assert_array_equal(rebuilt.precision, asymmetric)

    def test_vmap_maps_over_a_batch_of_messages(self):
        m1 = CanonicalGaussian(precision=[[1.0, 0.0], [0.0, 1.0]], potential=[1.0, 0.0])
        m2 = CanonicalGaussian(precision=[[2.0, 0.0], [0.0, 2.0]], potential=[0.0, 1.0])
        batch = jax.tree_util.tree_map(lambda *xs: jnp.stack(xs), m1, m2)
        potentials = jax.vmap(lambda m: m.potential)(batch)
        np.testing.assert_array_equal(potentials, [[1.0, 0.0], [0.0, 1.0]])

    def test_jit_add_agrees_with_eager(self):
        a = CanonicalGaussian(precision=[[1.0, 0.0], [0.0, 2.0]], potential=[1.0, 0.0])
        b = CanonicalGaussian(precision=[[3.0, 0.0], [0.0, 1.0]], potential=[0.0, 2.0])

        def added_potential(h1, h2):
            ga = CanonicalGaussian(a.precision, h1)
            gb = CanonicalGaussian(b.precision, h2)
            return (ga + gb).potential

        eager = (a + b).potential
        jitted = jax.jit(added_potential)(a.potential, b.potential)
        np.testing.assert_allclose(jitted, eager, atol=1e-12)

    def test_grad_through_add_and_to_moment(self):
        base_precision = jnp.array([[2.0, 0.0], [0.0, 2.0]])

        def mean0_after_add(h1, h2):
            a = CanonicalGaussian(base_precision, h1)
            b = CanonicalGaussian(base_precision, h2)
            mean, _ = (a + b).to_moment()
            return mean[0]

        grad_h1, grad_h2 = jax.grad(mean0_after_add, argnums=(0, 1))(
            jnp.array([1.0, 0.0]), jnp.array([0.0, 1.0])
        )
        assert bool(jnp.all(jnp.isfinite(grad_h1)))
        assert bool(jnp.all(jnp.isfinite(grad_h2)))

    def test_jit_marginalize_agrees_with_eager(self):
        rng = np.random.default_rng(5)
        _, _, precision, potential = _random_system(rng, 3)
        joint = CanonicalGaussian(precision, potential)

        def marginalized_potential(h):
            return CanonicalGaussian(precision, h).marginalize([1]).potential

        eager = joint.marginalize([1]).potential
        jitted = jax.jit(marginalized_potential)(joint.potential)
        np.testing.assert_allclose(jitted, eager, atol=1e-10)

    def test_grad_through_marginalize(self):
        rng = np.random.default_rng(6)
        _, _, precision, potential = _random_system(rng, 3)

        def marginal_mean0(h):
            mean, _ = CanonicalGaussian(precision, h).marginalize([1]).to_moment()
            return mean[0]

        grad = jax.grad(marginal_mean0)(jnp.asarray(potential))
        assert grad.shape == (3,)
        assert bool(jnp.all(jnp.isfinite(grad)))
