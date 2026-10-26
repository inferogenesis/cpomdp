"""The H* collapse sweep, as registered in PREREG.md (issue #58).

Runs the 432-cell factorial, writes every cell to ``results.csv``, applies the three
registered acceptance criteria to ``Pi``, triggers the registered alternatives only if
``Pi`` is not accepted, prints the report and draws ``collapse.png``.

Run from the repository root::

    uv run --extra examples python experiments/collapse/sweep.py
"""

import csv
import itertools
from dataclasses import dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from cpomdp import Belief, CallableSensor, LinearGaussianModel, Preference
from cpomdp.efe import expected_free_energy, policy_efe_trace

HERE = Path(__file__).parent

# --- fixed for every cell (PREREG.md, Model) ---
DYNAMICS_GAIN = 1.0  # a
OBSERVATION_GAIN = 1.0  # c
NOISE_SCALE = 1.0  # r0
NOISE_CENTRE = 0.0  # x_s
ACTION_BOUND = 1.0  # u_max
START_MEAN = 0.0  # mu_0^-
HORIZON = 40  # T runs 1 .. HORIZON
BOUND_CHECK_POINTS = 201

# --- the registered grid (PREREG.md, Grid) ---
LENGTH_SCALES = tuple(0.5 * 10 ** (i / 3) for i in range(4))  # lam
CONTROL_GAINS = tuple(0.2 * 10 ** (i / 3) for i in range(4))  # b
GOAL_WIDTHS = (0.5, 1.0, 2.0)  # sigma_g
GOAL_DISTANCES = (3.0, 6.0, 12.0)  # d
NOISE_PAIRS = ((0.01, 0.02), (0.02, 0.10), (0.05, 0.50))  # (q, Sigma_0^-)

# --- the registered analysis thresholds (PREREG.md, Analysis and acceptance) ---
FIT_R2 = 0.90
MARGIN_OVER_RAW = 0.15
PI_OVERLAP = 0.05  # |ln Pi_i - ln Pi_j|
AGREEMENT = np.log(1.15)  # |ln H_i - ln H_j|
MIN_VALID = 100


def log_noise(state, noise_params):
    """``R(x) = r0 * exp(-(x - x_s) / lam)`` as a 1x1 covariance."""
    noise_scale, noise_centre, length_scale = noise_params
    return jnp.reshape(
        noise_scale * jnp.exp(-(state[0] - noise_centre) / length_scale), (1, 1)
    )


@dataclass(frozen=True)
class Cell:
    """One grid point: the swept parameters."""

    length_scale: float  # lam
    control_gain: float  # b
    goal_width: float  # sigma_g
    goal_distance: float  # d
    dynamics_noise: float  # q
    start_cov: float  # Sigma_0^-, the first predicted covariance

    def model(self) -> LinearGaussianModel:
        """The scalar model. ``Sigma_0 = Sigma_0^- - q``, so Σ⁺ at step 1 is Σ_0^-."""
        sensor = CallableSensor(
            [[OBSERVATION_GAIN]],
            log_noise,
            (
                jnp.asarray(NOISE_SCALE),
                jnp.asarray(NOISE_CENTRE),
                jnp.asarray(self.length_scale),
            ),
        )
        return LinearGaussianModel(
            dynamics_matrix=[[DYNAMICS_GAIN]],
            dynamics_noise=[[self.dynamics_noise]],
            control_matrix=[[self.control_gain]],
            observation_model=sensor,
            prior=self.prior(),
        )

    def prior(self) -> Belief:
        """The planning-start belief ``(mu_0, Sigma_0^- - q)``."""
        return Belief([START_MEAN], [[self.start_cov - self.dynamics_noise]])

    def goal(self) -> float:
        """``x* = mu_0^- - d``, on the blind side."""
        return START_MEAN - self.goal_distance

    def preference(self) -> Preference:
        """Observation-space target ``c*x*`` at precision ``1/sigma_g^2``."""
        return Preference(
            [OBSERVATION_GAIN * self.goal()], [[1.0 / self.goal_width**2]]
        )

    def groups(self) -> dict[str, float]:
        """The registered dimensionless groups, evaluated at the planning start."""
        noise_at_start = NOISE_SCALE * np.exp(
            -(START_MEAN - NOISE_CENTRE) / self.length_scale
        )
        ell = self.length_scale  # ell_R = |d ln R/dx|^-1 = lam
        step = abs(self.control_gain) * ACTION_BOUND  # delta_u
        distance = abs(START_MEAN - self.goal())
        kappa = step / ell
        snr = OBSERVATION_GAIN**2 * self.start_cov / noise_at_start  # s
        g_epi = 0.5 * kappa * snr / (1 + snr)
        c_prag = step * distance / self.goal_width**2
        return {
            "kappa": kappa,
            "s": snr,
            "Pi": g_epi / c_prag,
            "Pi_inf": self.goal_width**2 / (4 * ell * distance),
            "N_sat": np.log(1 / snr) / kappa,
            "Pi_amb": NOISE_SCALE / (2 * OBSERVATION_GAIN**2 * ell * distance),
        }


def grid() -> list[Cell]:
    """The 432-cell factorial, in a fixed order."""
    return [
        Cell(lam, b, sigma_g, d, q, sigma0)
        for lam, b, sigma_g, d, (q, sigma0) in itertools.product(
            LENGTH_SCALES, CONTROL_GAINS, GOAL_WIDTHS, GOAL_DISTANCES, NOISE_PAIRS
        )
    ]


_trace = jax.jit(policy_efe_trace)
_one_step = jax.jit(jax.vmap(expected_free_energy, in_axes=(None, None, 0, None)))


def check_policies_on_bounds(cell: Cell) -> None:
    """Stop unless pi_prag is -u_max and pi_epi is +u_max at T = 1 (PREREG.md)."""
    model = cell.model()
    actions = jnp.linspace(-ACTION_BOUND, ACTION_BOUND, BOUND_CHECK_POINTS)
    predicted_obs = OBSERVATION_GAIN * (
        DYNAMICS_GAIN * START_MEAN + cell.control_gain * actions
    )
    goal_mismatch = 0.5 * (predicted_obs - OBSERVATION_GAIN * cell.goal()) ** 2
    epistemic = _one_step(
        model, cell.prior(), actions[:, None], cell.preference()
    ).epistemic
    if int(jnp.argmin(goal_mismatch)) != 0:
        raise AssertionError(f"pi_prag is not on -u_max in {cell}")
    if int(jnp.argmax(epistemic)) != BOUND_CHECK_POINTS - 1:
        raise AssertionError(f"pi_epi is not on +u_max in {cell}")


def margin(cell: Cell) -> np.ndarray:
    """``Delta(T) = G(pi_epi, T) - G(pi_prag, T)`` for T = 1..HORIZON.

    One 40-step trace per policy and a prefix sum: the rollout is a causal scan, so
    its first T steps are the T-step rollout.
    """
    model, prior, preference = cell.model(), cell.prior(), cell.preference()
    walk_up = jnp.full((HORIZON, 1), +ACTION_BOUND)  # pi_epi
    walk_down = jnp.full((HORIZON, 1), -ACTION_BOUND)  # pi_prag
    g_epi = np.cumsum(np.asarray(_trace(model, prior, walk_up, preference).g))
    g_prag = np.cumsum(np.asarray(_trace(model, prior, walk_down, preference).g))
    return g_epi - g_prag


def categorise(delta: np.ndarray) -> tuple[str, int | None, float | None]:
    """The registered category, and ``(H*_int, H*_cts)`` for a valid cell."""
    if not np.all(np.isfinite(delta)):
        return "nonfinite", None, None
    if np.any(delta == 0):
        return "degenerate", None, None
    sign_changes = np.flatnonzero(np.diff(np.sign(delta)))
    if len(sign_changes) > 1:
        return "multi_cross", None, None
    if delta[0] < 0:
        return "cross_at_1", None, None
    if len(sign_changes) == 0:
        return "no_cross", None, None
    h_int = int(sign_changes[0]) + 2  # delta[i] is T = i + 1; the crossing is at i + 1
    before, after = delta[h_int - 2], delta[h_int - 1]
    return "valid", h_int, (h_int - 1) + before / (before - after)


def run() -> list[dict]:
    """Score every cell and return one row per cell."""
    rows = []
    for cell in grid():
        check_policies_on_bounds(cell)
        delta = margin(cell)
        category, h_int, h_cts = categorise(delta)
        rows.append(
            {
                "lam": cell.length_scale,
                "b": cell.control_gain,
                "sigma_g": cell.goal_width,
                "d": cell.goal_distance,
                "q": cell.dynamics_noise,
                "Sigma0": cell.start_cov,
                **cell.groups(),
                "category": category,
                "H_int": h_int,
                "H_cts": h_cts,
                "delta_1": float(delta[0]),
                "delta_40": float(delta[-1]),
            }
        )
    return rows


def write_csv(rows: list[dict], path: Path) -> None:
    """Write every cell, valid or not."""
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    k: (
                        ""
                        if v is None
                        else repr(float(v))
                        if isinstance(v, float)
                        else v
                    )
                    for k, v in row.items()
                }
            )


def r_squared(response: np.ndarray, *regressors: np.ndarray) -> float:
    """OLS ``R^2`` with an intercept."""
    design = np.column_stack([np.ones_like(response), *regressors])
    coef, *_ = np.linalg.lstsq(design, response, rcond=None)
    residual = response - design @ coef
    return 1 - residual @ residual / np.sum((response - response.mean()) ** 2)


RAW_PARAMETERS = ("lam", "b", "sigma_g", "d", "Sigma0", "q")


def quadrant(row: dict) -> tuple[bool, bool]:
    """Median split on ``lam`` and ``b`` over the grid values."""
    return (
        row["lam"] > np.median(LENGTH_SCALES),
        row["b"] > np.median(CONTROL_GAINS),
    )


def assess(valid: list[dict], group: str, raw_r2: dict[str, float]) -> dict:
    """Criteria 1-3 for one group, in ADR-029 vocabulary."""
    log_h = np.log([r["H_cts"] for r in valid])
    log_group = np.log([r[group] for r in valid])
    fit = r_squared(log_h, log_group)

    c1 = "PASS" if fit > FIT_R2 else "FAIL"
    if group in RAW_PARAMETERS or group == "s":
        c2 = "NOT APPLICABLE"  # s == Sigma_0^- here, a raw parameter
    else:
        shortfall = {p: fit - r2 for p, r2 in raw_r2.items()}
        c2 = "PASS" if min(shortfall.values()) >= MARGIN_OVER_RAW else "FAIL"

    pairs = agreeing = 0
    worst = 0.0
    for i, j in itertools.combinations(range(len(valid)), 2):
        if quadrant(valid[i]) == quadrant(valid[j]):
            continue
        if abs(log_group[i] - log_group[j]) > PI_OVERLAP:
            continue
        pairs += 1
        gap = abs(log_h[i] - log_h[j])
        worst = max(worst, gap)
        agreeing += gap <= AGREEMENT
    c3 = "NOT APPLICABLE" if pairs == 0 else "PASS" if agreeing == pairs else "FAIL"

    return {
        "group": group,
        "r2": fit,
        "c1": c1,
        "c2": c2,
        "c3": c3,
        "pairs": pairs,
        "agreeing": agreeing,
        "worst_ratio": float(np.exp(worst)),
        "accepted": (c1, c2, c3) == ("PASS", "PASS", "PASS"),
    }


def report(rows: list[dict]) -> list[dict]:
    """Print the registered analysis and return the valid cells."""
    categories = {}
    for row in rows:
        categories[row["category"]] = categories.get(row["category"], 0) + 1
    print("categories:", dict(sorted(categories.items())))

    valid = [r for r in rows if r["category"] == "valid"]
    print(f"valid cells: {len(valid)} of {len(rows)}")
    if len(valid) < MIN_VALID:
        print("NOT APPLICABLE: fewer than", MIN_VALID, "valid cells")
        return valid

    log_h = np.log([r["H_cts"] for r in valid])
    raw_r2 = {
        p: r_squared(log_h, np.log([r[p] for r in valid])) for p in RAW_PARAMETERS
    }
    print("raw-parameter R^2:", {p: round(float(v), 4) for p, v in raw_r2.items()})

    pi = assess(valid, "Pi", raw_r2)
    _print_assessment(pi)
    if pi["accepted"]:
        print("collapse onto Pi: ACCEPTED")
        return valid

    print("collapse onto Pi: NOT ACCEPTED -- running the registered alternatives")
    for group in ("kappa", "Pi_inf", "s", "N_sat", "Pi_amb"):
        _print_assessment(assess(valid, group, raw_r2))

    log_pi = np.log([r["Pi"] for r in valid])
    log_kappa = np.log([r["kappa"] for r in valid])
    design = np.column_stack([np.ones_like(log_h), log_pi, log_kappa])
    (_, beta, gamma), *_ = np.linalg.lstsq(design, log_h, rcond=None)
    two = r_squared(log_h, log_pi, log_kappa)
    print(
        f"two-group fit Pi^beta kappa^gamma: R^2={two:.4f} "
        f"beta={beta:.4f} gamma={gamma:.4f} (a two-group fit, not a collapse)"
    )
    return valid


def _print_assessment(a: dict) -> None:
    print(
        f"  {a['group']:7s} R^2={a['r2']:.4f}  c1={a['c1']}  c2={a['c2']}  "
        f"c3={a['c3']} ({a['agreeing']}/{a['pairs']} pairs agree, worst "
        f"ratio {a['worst_ratio']:.3f})  accepted={a['accepted']}"
    )


def plot(valid: list[dict], path: Path) -> None:
    """``ln H*_cts`` against ``ln Pi``, coloured by quadrant."""
    import matplotlib as mpl

    mpl.use("Agg")
    import matplotlib.pyplot as plt

    labels = {
        (False, False): "lam low, b low",
        (False, True): "lam low, b high",
        (True, False): "lam high, b low",
        (True, True): "lam high, b high",
    }
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for key, label in labels.items():
        sub = [r for r in valid if quadrant(r) == key]
        ax.scatter([r["Pi"] for r in sub], [r["H_cts"] for r in sub], s=14, label=label)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Pi = g_epi / c_prag")
    ax.set_ylabel("H*_cts")
    ax.set_title(f"H* against Pi, {len(valid)} valid cells")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)


def main() -> None:
    """Run, write, report, plot."""
    rows = run()
    write_csv(rows, HERE / "results.csv")
    valid = report(rows)
    if valid:
        plot(valid, HERE / "collapse.png")


if __name__ == "__main__":
    main()
