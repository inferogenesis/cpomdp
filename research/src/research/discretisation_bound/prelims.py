"""Pages 1 to 4 of the derivation: the cell, the engine's seven steps, the split.

`research/discretisation_bound_hand_derivation.md` transcribes them. Run::

    uv run --no-sync python -m research.discretisation_bound.prelims

What is checked, in the notes' order:

1. **The cell.** `μ* = √10`, `R̄ = 2R₀`, `h_x = 0.015σ`, `h_y = 0.045s_p`, each exact
   and each matching what `research.explorations.threshold` declares.
2. **The lattice as the engine builds it.** Weights `h`, halved at both ends. The
   endpoints exact. The interior nodes are not `x₀ + ih` bit for bit, because
   `linspace` computes `x₀(1 − t) + x_N·t`. Route B assumes exact spacing, so the
   departure is measured here in units of `h` and left to the roundoff disclosure.
3. **Step 4.** The plug-in variance `(1 − K)ŝ²` in its two other forms. The plug-in
   Gaussian is the exact posterior of the model with the noise frozen at `R̂`.
4. **Step 5.** The second line of the KL is the first with `q̃ = q/m_Q` and
   `p̂ = J/m_J` substituted. Termwise, so it holds under any weighted sum. `Σq̃ = 1`
   is not used.
5. **The exact quantity.** `log q − log p(x|y)` expands as the notes write it.
6. **The target and the floor.** `T/k_min` and `100 · 2⁻⁵²`.
7. **Route B's rate on the state axis.** `2π/(h_x/σ) = 400π/3`, the notes' 419.
"""

import math

import sympy

from research.discretisation_bound import cell
from research.explorations import operating_point, threshold

__all__ = [
    "cell_constants",
    "end_weight_residuals",
    "engine_node_departure",
    "engine_weight_departure",
    "exact_kl_residual",
    "kl_second_line_residual",
    "main",
    "plug_in_posterior_residual",
    "plug_in_variance",
    "plug_in_variance_residuals",
    "state_rate",
]


def cell_constants() -> dict[str, sympy.Expr]:
    """The page-1 constants, exact.

    Returns:
        `μ*`, `R̄`, `h_x/σ` and `h_y/s_p` by name.
    """
    return {
        "ridge_mean": cell.RIDGE_MEAN,
        "plug_in_noise": cell.PLUG_IN_NOISE,
        "state_step": cell.STATE_STEP,
        "observation_step": cell.OBSERVATION_STEP,
    }


def engine_weight_departure(spread: float) -> float:
    """How far the engine's weights sit from `h`, halved at both ends.

    Args:
        spread: the prior spread `σ`.

    Returns:
        The largest departure, relative to `h`.
    """
    grid = cell.state_grid(spread)
    step = grid.spacing[0]
    weights = [float(weight) for weight in grid.weights]
    expected = [step / 2, *([step] * (len(weights) - 2)), step / 2]
    return max(
        abs(weight - want) / step
        for weight, want in zip(weights, expected, strict=True)
    )


def engine_node_departure(spread: float) -> tuple[float, bool]:
    """How far the engine's nodes sit from `x₀ + ih`, and whether the ends are exact.

    Args:
        spread: the prior spread `σ`.

    Returns:
        The largest departure in units of `h`, and whether the first and last nodes
        are the box's corners exactly.
    """
    grid = cell.state_grid(spread)
    lower, upper = grid.box[0]
    step = grid.spacing[0]
    nodes = [float(node) for node in grid.nodes[:, 0]]
    departure = max(
        abs(node - (lower + index * step)) / step for index, node in enumerate(nodes)
    )
    return departure, nodes[0] == lower and nodes[-1] == upper


def end_weight_residuals(spread: float) -> tuple[float, float]:
    """The engine's box sum against the end-weight correction, with either sign.

    With weights `h` halved at both ends, `Σ_x f = h Σ_{i=0}^{N} f(x_i) − (h/2)(f(x₀) +
    f(x_N))`. The scan writes the correction with a plus. `f(x) = x` keeps both edge
    values far above roundoff, so the wrong sign shows.

    Args:
        spread: the prior spread `σ`.

    Returns:
        The relative miss with the correction subtracted, then with it added.
    """
    grid = cell.state_grid(spread)
    step = grid.spacing[0]
    nodes = [float(node) for node in grid.nodes[:, 0]]
    engine = math.fsum(
        float(weight) * node for weight, node in zip(grid.weights, nodes, strict=True)
    )
    plain = step * math.fsum(nodes)
    edges = (step / 2) * (nodes[0] + nodes[-1])
    return abs(engine - (plain - edges)) / engine, abs(
        engine - (plain + edges)
    ) / engine


def plug_in_variance(prior_variance: sympy.Expr, noise: sympy.Expr) -> sympy.Expr:
    """Step 4's `v_q = (1 − K)ŝ²`, with `K = ŝ²/(ŝ² + R̂)`.

    Args:
        prior_variance: `ŝ²`, the grid's prior variance.
        noise: `R̂ = R(μ̂)`.

    Returns:
        The plug-in posterior's variance.
    """
    gain = prior_variance / (prior_variance + noise)  # K
    return (1 - gain) * prior_variance


def plug_in_variance_residuals() -> tuple[sympy.Expr, sympy.Expr]:
    """`v_q` against `ŝ²R̂/(ŝ² + R̂)`, and `1/v_q` against `1/ŝ² + 1/R̂`.

    Returns:
        The two residuals, simplified. Both are zero.
    """
    prior_variance, noise = sympy.symbols("shat2 Rhat", positive=True)
    variance = plug_in_variance(prior_variance, noise)
    closed = sympy.simplify(
        variance - prior_variance * noise / (prior_variance + noise)
    )
    precision = sympy.simplify(1 / variance - (1 / prior_variance + 1 / noise))
    return closed, precision


def plug_in_posterior_residual() -> sympy.Expr:
    """`∂/∂x` of `log N(x; μ̂, ŝ²) + log N(y; x, R̂) − log N(x; m_q, v_q)`.

    Zero means the three differ by a constant in `x`. So the plug-in Gaussian is the
    exact posterior of the model whose noise is frozen at `R̂`, which is what makes it
    a Kalman update.

    Returns:
        The derivative, simplified. It is zero.
    """
    state, observation, prior_mean = sympy.symbols("x y muhat", real=True)
    prior_variance, noise = sympy.symbols("shat2 Rhat", positive=True)
    gain = prior_variance / (prior_variance + noise)  # K
    mean = prior_mean + gain * (observation - prior_mean)  # m_q
    variance = plug_in_variance(prior_variance, noise)  # v_q

    def log_normal(point, centre, spread_squared):
        return (
            -((point - centre) ** 2) / (2 * spread_squared)
            - sympy.log(2 * sympy.pi * spread_squared) / 2
        )

    difference = (
        log_normal(state, prior_mean, prior_variance)
        + log_normal(observation, state, noise)
        - log_normal(state, mean, variance)
    )
    return sympy.simplify(sympy.diff(difference, state))


def kl_second_line_residual() -> sympy.Expr:
    """Step 5's two lines, termwise, as one residual.

    Line one is `q̃ (log q̃ − log p̂)`. Line two is
    `(1/m_Q) q (log q − log J − log m_Q + log m_J)`. Substituting `q̃ = q/m_Q` and
    `p̂ = J/m_J` turns one into the other with nothing else assumed. The weighted sum
    `Σ_x` is linear, so the identity holds for it too.

    Returns:
        Line one minus line two, simplified. It is zero.
    """
    plug_in, joint, plug_in_mass, joint_mass = sympy.symbols(
        "q J m_Q m_J", positive=True
    )
    first = (plug_in / plug_in_mass) * (
        sympy.log(plug_in / plug_in_mass) - sympy.log(joint / joint_mass)
    )
    second = (plug_in / plug_in_mass) * (
        sympy.log(plug_in)
        - sympy.log(joint)
        - sympy.log(plug_in_mass)
        + sympy.log(joint_mass)
    )
    return sympy.simplify(sympy.expand_log(first - second))


def exact_kl_residual() -> sympy.Expr:
    """`log q − log p(x|y)`, against `log q − log π − log N(y; x, R) + log p*`.

    `p(x|y) = π(x) N(y; x, R(x)) / p*(y)` is the exact posterior.

    Returns:
        The residual, simplified. It is zero.
    """
    plug_in, prior, likelihood, predictive = sympy.symbols(
        "q pi N pstar", positive=True
    )
    posterior = prior * likelihood / predictive
    residual = (sympy.log(plug_in) - sympy.log(posterior)) - (
        sympy.log(plug_in)
        - sympy.log(prior)
        - sympy.log(likelihood)
        + sympy.log(predictive)
    )
    return sympy.simplify(sympy.expand_log(residual))


def state_rate() -> sympy.Expr:
    """`2π / (h_x/σ)`, the exponent per unit `a/σ` in Route B on the state axis.

    Returns:
        `400π/3`.
    """
    return 2 * sympy.pi / cell.STATE_STEP


def main() -> None:
    """Check pages 1 to 4 and assert each claim."""
    departure = cell.departure_from_declared()
    print("=== Prelims: the cell, exact against the declared floats ===")
    constants = cell_constants()
    for name, value in constants.items():
        print(f"  {name:<17} {sympy.sstr(value):<10} = {float(value):.6g}")
    print(f"  worst departure from research.explorations.threshold: {departure:.1e}")
    assert departure < 1e-15, departure
    assert constants["ridge_mean"] == sympy.sqrt(10)
    assert constants["plug_in_noise"] == 2 * cell.BASE_NOISE
    assert constants["state_step"] == sympy.Rational(3, 200)
    assert constants["observation_step"] == sympy.Rational(9, 200)

    print("\n=== The declared lattice, as the engine builds it ===")
    weights = max(engine_weight_departure(spread) for spread in threshold.SPREADS)
    nodes = [engine_node_departure(spread) for spread in threshold.SPREADS]
    worst_node = max(departure for departure, _ in nodes)
    print(f"  weights against h, halved at both ends: worst {weights:.1e} of h")
    print(f"  nodes against x0 + ih: worst {worst_node:.1e} of h")
    assert weights < 1e-12, weights
    assert all(exact for _, exact in nodes)
    assert worst_node < 1e-9, worst_node
    print("  endpoints exact at all fifteen spreads; interior nodes off by roundoff")
    signs = [end_weight_residuals(spread) for spread in threshold.SPREADS]
    assert max(subtracted for subtracted, _ in signs) < 1e-14, signs
    assert min(added for _, added in signs) > 1e-4, signs
    print("  the end-weight correction is subtracted, not added as the scan writes")

    print("\n=== Steps 4 and 5, and the exact quantity ===")
    closed, precision = plug_in_variance_residuals()
    assert closed == 0, closed
    assert precision == 0, precision
    print("  v_q = (1 - K) s^2 = s^2 R/(s^2 + R), and 1/v_q = 1/s^2 + 1/R")
    assert plug_in_posterior_residual() == 0
    print("  the plug-in Gaussian is the exact posterior with the noise frozen at R")
    assert kl_second_line_residual() == 0
    print("  step 5's second line is the first, substituted; sum q~ = 1 is not used")
    assert exact_kl_residual() == 0
    print("  log q - log p(x|y) expands as the notes write it")

    print("\n=== The target and the floor ===")
    bar = float(cell.GAP_BAR)
    print(
        f"  T = {operating_point.THRESHOLD:.4g}, k_min = {operating_point.K_MIN:g},"
        f" T/k_min = {bar:.4g} nats"
    )
    assert f"{operating_point.THRESHOLD:.3e}" == "5.962e-04"
    assert f"{bar:.2e}" == "5.96e-05", bar
    print(f"  ROUNDOFF_FLOOR = 100 * 2^-52 = {threshold.ROUNDOFF_FLOOR:.3g}")
    assert threshold.ROUNDOFF_FLOOR == 100.0 * 2.0**-52
    assert f"{threshold.ROUNDOFF_FLOOR:.1e}" == "2.2e-14"

    print("\n=== Route B on the state axis ===")
    rate = state_rate()
    print(f"  2*pi/(h_x/sigma) = {sympy.sstr(rate)} = {float(rate):.3f}")
    assert rate == 400 * sympy.pi / 3
    assert round(float(rate)) == 419


if __name__ == "__main__":
    main()
