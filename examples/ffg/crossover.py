"""Where the crossover lives: the horizon at which the best plan stops reaching.

At a short horizon the best plan on the coupled-tree cue task is a direct reach for the
goal. At a long enough horizon it becomes a two-phase walk: drive to the cue, sense the
hidden context, reverse, and commit. This demo locates the horizon H* where the switch
happens and shows why.

Two measurements, and the distinction between them is the whole point.

- **The decisive one is selection-free.** For each horizon the *exhaustive* argmin over
  every ``A^H`` action sequence is computed by ``EnumeratedEfeSearch.over_backend``. It
  is prior-ward (a reach) through H = 6 and cue-ward (a two-phase walk) at H = 7, and
  each horizon is an exact finite enumeration, so no policy is chosen with knowledge of
  the outcome. This is what carries the result.

- **The mechanism curve is exposition, and it is post-selection.** The specific walk the
  search picks out at H = 7, ``[+1,-2,-2,0,...]``, is scored back across the horizon
  against the coast-to-stop reach ``[-2,-1,0,...]`` to read out the epistemic and
  pragmatic split. That pair was *found by* the search, so its crossing is shown to
  explain the flip, never to establish it.

The mechanism is the surprising part. The epistemic pull is flat (~1.7 nats, the
one-step value); it does not accumulate. What moves is the pragmatic gradient: after
sensing at the cue the contracted context belief lowers the *commit* channel's expected
ambiguity by ~0.67 nats per step, and that saving accumulates until it overtakes the
one-time detour cost. So the reach's advantage decays with the horizon and the walk wins
at H = 7 -- the gradient decays below the constant pull, not the pull past the gradient.

``--check`` asserts the flip, the mechanism split, an independent NumPy oracle on the
headline number, and the kernel's numerical inertness, with no plotting deps.
``--refinement`` re-runs the registered refinement cells on the chunked path, which is
minutes to hours and so sits off every gate::

    uv run --no-sync python examples/ffg/crossover.py --check
    uv run --no-sync python examples/ffg/crossover.py --refinement
    uv run --no-sync python examples/ffg/crossover.py
"""

from __future__ import annotations

import itertools
import sys
from typing import NamedTuple

import cue_maze as demo_maze
import epistemic_dissociation_figure as demo
import jax
import jax.numpy as jnp
import numpy as np

from cpomdp.crossover import crossover_statistic
from cpomdp.diagnostics import is_positive_definite, rollout_conditioning
from cpomdp.efe import policy_efe_ffg, policy_efe_ffg_trace
from cpomdp.enumeration import (
    ChunkedEfeSearch,
    CompletenessCertificate,
    EnumeratedEfeSearch,
    FiniteActionSet,
)
from cpomdp.selection import Preference
from cpomdp.types import Belief
from warrantlib import (
    CheckReport,
    Outcome,
    Provenance,
    Tier,
    Warrant,
    check_summary,
)

# The rollout-hygiene bars this shares with tests/test_rollout_hygiene.py: a floor under
# min-eigenvalue(Σ_post) and a ceiling on cond(Σ⁺), cond(S), cond(Σ_post).
MIN_EIG_FLOOR = 1e-9
COND_CEILING = 1e8

# The registered action set the anchors were fixed against. It clips the reach at the
# grid edge -2, so it needs two steps to reach the goal at -3. A set containing -3,
# which reaches the goal in one step from the start, flips one horizon sooner (see the
# +edge measurement below). H* on this set is therefore an upper bound, and the honest
# headline number. -3 is the one-step reach from the start, not an established optimum,
# since the walk arrives at the cue at +1. The seven-action set below supplies that
# one-step return and also gives 6, so extension saturates; wider sets are unmeasured.
V1 = [-2.0, -1.0, 0.0, 1.0, 2.0]  # the registered action set (clips the reach at -2)
V1_EDGE = [-3.0, *V1]  # + the one-step reach -3; flips one horizon sooner
# + the one-step *return*: the walk reaches the cue at +1, and the goal at -3 is then a
# displacement of -4. The extension axis of the fourth D3 falsifier, registered at
# `H* <= 6` before it ran (see fep_falsification_battery.md, 2026-08-20).
V1_EXT = [-4.0, *V1_EDGE]
ACTION_SET = FiniteActionSet([[a] for a in V1], version="v1")
EDGE_SET = FiniteActionSet([[a] for a in V1_EDGE], version="v1-edge")
EXT_SET = FiniteActionSet([[a] for a in V1_EXT], version="v1-ext")
# The refinement axis: the same [-2, 2] range at finer spacing, so the largest magnitude
# does not move and neither branch's step count can. Registered as a stability test at
# |dH*| <= 1 (fep_falsification_battery.md, 2026-08-20). Both run chunked: front-loaded
# peak is 22.6 GiB at 9^8 and 210 GiB at 17^7, against a chunked 0.46 GiB (ADR-036).
REFINE_05 = [round(-2.0 + 0.5 * i, 2) for i in range(9)]
REFINE_025 = [round(-2.0 + 0.25 * i, 2) for i in range(17)]
REFINE_05_SET = FiniteActionSet([[a] for a in REFINE_05], version="v1-refine-0.5")
REFINE_025_SET = FiniteActionSet([[a] for a in REFINE_025], version="v1-refine-0.25")
FLIP_H = 7  # the crossover horizon on ACTION_SET
EDGE_H_STAR = 6  # measured on V1_EDGE, and on V1_EXT too, which is the saturation
MAX_H = 7  # the exhaustive sweep budget (feasibility is |A|^H * H, printed below)
# Declared feasibility bound: enumeration is feasible to H_MAX (5^9 * 9 = 17.6M scored
# steps, measured); the crossover at H=7 sits well inside it and is cue-ward through 9.
H_MAX = 9
EPS_PLATEAU = 0.2  # the epistemic pull varies by less than this: flat, not growing


def _setup():
    """``(backend, belief, preference, info_block)`` for the coupled-tree cue task."""
    backend = demo.build_backend(epistemic_alive=True, cue_x=demo.CUE_DETOUR_X)
    belief = demo.start_belief()
    preference = Preference(
        goal=[0.0, 0.0],
        precision=[[demo.GOAL_PRECISION, 0.0], [0.0, demo.INFO_PRECISION]],
    )
    info_block = tuple(backend.block(demo.CONTEXT))
    return backend, belief, preference, info_block


def _cue_ward(policy) -> bool:
    """A plan visits the cue if its position (cumulative action from x=0) reaches it."""
    return bool(np.cumsum(np.asarray(policy).ravel()).max() >= 0.5)


def _walk(horizon: int):
    """The two-phase walk the search selects at H*: cue, sense, reverse, then park."""
    actions = [1.0, -2.0, -2.0][: min(horizon, 3)] + [0.0] * max(0, horizon - 3)
    return jnp.asarray(actions).reshape(-1, 1)


def _reach(horizon: int):
    """The coast-to-stop reach: decelerate onto the goal at x=-3 and hold."""
    actions = [-2.0, -1.0][: min(horizon, 2)] + [0.0] * max(0, horizon - 2)
    return jnp.asarray(actions).reshape(-1, 1)


# --- 1. the decisive, selection-free measurement -----------------------------------
def argmin_at(horizon: int, action_set=ACTION_SET):
    """The exhaustive argmin at a horizon: ``(n_policies, Gmin, argmin, cue_ward)``."""
    backend, belief, preference, info_block = _setup()
    search = EnumeratedEfeSearch.over_backend(
        backend, action_set, info_block=info_block, horizon=horizon
    )
    result = search.evaluate(belief, preference)
    policy = np.asarray(result.best_policy).ravel()
    return search.n_policies, float(result.g.min()), policy, _cue_ward(policy)


def exhaustive_flip(action_set=ACTION_SET, max_horizon=MAX_H):
    """Per-horizon exhaustive argmin ``(H, n_policies, Gmin, argmin, cue_ward)``."""
    return [(h, *argmin_at(h, action_set)) for h in range(1, max_horizon + 1)]


class FlipMeasurement(NamedTuple):
    """One horizon's exhaustive enumeration, as the falsifiers read it.

    Separating the measurement from the report is what lets the reporting logic be
    exercised on a refuting result. Nothing in a live run can produce one, and a branch
    no run can reach is a branch no run has tested.

    Attributes:
        horizon: the horizon enumerated.
        certificate: that enumeration's completeness certificate.
        delta_g: ``G(walk) - G(reach)`` at this horizon. Negative means the walk wins.
        bound: the error ``COND_CEILING`` allows on that difference.
        cue_ward: whether the *exhaustive* argmin visits the cue. The direction the
            falsifier asks about, from the enumeration rather than from the pair.
    """

    horizon: int
    certificate: CompletenessCertificate
    delta_g: float
    bound: float
    cue_ward: bool

    @property
    def separated(self) -> bool:
        """Whether the margin clears the error bound, so the ordering is decidable."""
        return abs(self.delta_g) > self.bound


class ExtensionMeasurement(NamedTuple):
    """The extension axis of the fourth D3 falsifier, as its report reads it.

    Attributes:
        h_star: the first horizon whose exhaustive argmin is cue-ward, or ``None`` if
            none is within the swept range.
        certificate: the completeness certificate at that horizon, or at the last
            swept one when no flip was found.
        policy: the argmin there, so the mechanism is readable beside the number.
        sensed: whether the lattice can land on the cue at all. A set that cannot
            produces a null about the geometry rather than about the objective, which
            is what the registered void guard exists to separate.
    """

    h_star: int | None
    certificate: CompletenessCertificate
    policy: object
    sensed: bool


#: The registered bar for the extension axis, from the 2026-08-20 pre-registration.
EXT_BAR = 6

#: Where that prediction was registered, and the ref whose tree measured against it.
#: Registered before the run, which is what the ordering claim rests on and what
#: `Provenance` exists to let a reviewer check.
EXT_PROVENANCE = Provenance(
    registered_at="86d1f22",
    measured_at="17a1be7",
    registered="extension axis registered at H* <= 6 (battery, 2026-08-20)",
)


class RefinementRow(NamedTuple):
    """One refined-set horizon, measured on the chunked path.

    Attributes:
        horizon: the horizon enumerated.
        gmin: the exhaustive minimum EFE there.
        policy: the argmin, so a half-step action would be visible in it.
        cue_ward: whether that argmin visits the cue.
        certificate: the completeness certificate the loop's own count produced.
    """

    horizon: int
    gmin: float
    policy: tuple[float, ...]
    cue_ward: bool
    certificate: CompletenessCertificate


def measure_refinement(action_set, horizon: int) -> RefinementRow:
    """Enumerate a refined set at one horizon, in blocks.

    Chunked rather than front-loaded, because these sets are why the chunked path
    exists: the score vector alone is 3.28 GB at ``17^7``. Peak here is
    block-determined and flat in ``|A|^H``, measured at 0.46 GiB on the step-0.5 set.
    """
    # The registered void guard, which names a *refined* set as its case: one that
    # cannot land on the cue gives a null about the geometry rather than the objective.
    # Computed here rather than asserted in prose, and refused rather than reported.
    if demo_maze.best_reachable_noise(action_set, 1, horizon) > demo_maze.R_LO:
        raise ValueError(
            f"{action_set.version} cannot land on the cue within H = {horizon}, so any "
            "null it produces is geometry and not a result (registered void guard)"
        )
    backend, belief, preference, info_block = _setup()
    search = ChunkedEfeSearch.over_backend(
        backend, action_set, info_block=info_block, horizon=horizon
    )
    result = search.reduce(belief, preference)
    policy = np.asarray(result.best_policy).ravel()
    return RefinementRow(
        horizon=horizon,
        gmin=float(result.best_g),
        policy=tuple(float(a) for a in policy),
        cue_ward=_cue_ward(policy),
        certificate=result.certificate,
    )


#: The step-0.5 refinement, measured on the chunked path at `3619016` and reproducible
#: with `--refinement`. Recorded rather than re-run on any gate: 5.3M policies at H = 6
#: and 4.8M at H = 7 is minutes, not the seconds a check may take. What makes this a
#: record rather than the write-up claim it replaces is that a commit builds the set,
#: each row carries its completeness certificate, and the provenance names both refs.
REFINEMENT_05_ROWS = (
    RefinementRow(
        horizon=6,
        gmin=364.642964185792,
        policy=(-2.0, -1.0, 0.0, 0.0, 0.0, 0.0),
        cue_ward=False,
        certificate=CompletenessCertificate(
            expected=9**6,
            visited=9**6,
            warrant=Warrant.PROVED,
            action_set_size=9,
            horizon=6,
            action_set_version="v1-refine-0.5",
        ),
    ),
    RefinementRow(
        horizon=7,
        gmin=425.163110098734,
        policy=(1.0, -2.0, -2.0, 0.0, 0.0, 0.0, 0.0),
        cue_ward=True,
        certificate=CompletenessCertificate(
            expected=9**7,
            visited=9**7,
            warrant=Warrant.PROVED,
            action_set_size=9,
            horizon=7,
            action_set_version="v1-refine-0.5",
        ),
    ),
)

#: The step-0.25 refinement, measured at `c37fac3` and reproducible with `--refinement`.
#: 410,338,673 policies at H = 7, 2.5 hours at 46k policies/s, peak 0.47 GiB. Both
#: scores are byte-identical to the step-0.5 cell across 86x the policies, and no
#: quarter-step action reaches either argmin.
REFINEMENT_025_ROWS = (
    RefinementRow(
        horizon=6,
        gmin=364.642964185792,
        policy=(-2.0, -1.0, 0.0, 0.0, 0.0, 0.0),
        cue_ward=False,
        certificate=CompletenessCertificate(
            expected=17**6,
            visited=17**6,
            warrant=Warrant.PROVED,
            action_set_size=17,
            horizon=6,
            action_set_version="v1-refine-0.25",
        ),
    ),
    RefinementRow(
        horizon=7,
        gmin=425.163110098734,
        policy=(1.0, -2.0, -2.0, 0.0, 0.0, 0.0, 0.0),
        cue_ward=True,
        certificate=CompletenessCertificate(
            expected=17**7,
            visited=17**7,
            warrant=Warrant.PROVED,
            action_set_size=17,
            horizon=7,
            action_set_version="v1-refine-0.25",
        ),
    ),
)

#: The crossover rows' registration. One commit derived the separation bar from the
#: declared conditioning ceiling and reported these falsifiers against it, so the two
#: refs are one and the render says the history orders nothing. Module scope rather than
#: a local, so `tests/test_provenance_ordering.py` collects it: a registration only a
#: function body holds is a registration nothing checks.
FLIP_PROVENANCE = Provenance(
    registered_at="efc43e2",
    measured_at="efc43e2",
    registered="the flip separation bar, derived from the conditioning ceiling",
)


class RefinementCell(NamedTuple):
    """One refined action set, its measured rows, and the ref that measured them.

    Bundled rather than kept as parallel constants so a cell can be injected whole. The
    falsifier reads its outcome off these, and a refuting cell is what lets the refuting
    branch be executed by a test instead of only described.

    Attributes:
        label: the spacing, as the detail line names it.
        rows: one measured row per horizon.
        provenance: which ref registered the stability test, and which measured this
            cell. The two cells were measured at different commits, so the ref belongs
            here rather than on a single constant covering both.
    """

    label: str
    rows: tuple[RefinementRow, ...]
    provenance: Provenance


#: The registered tolerance for the refinement axis: `|dH*| <= 1` is PASS, `>= 2` FAIL
#: (fep_falsification_battery.md, PRE-REGISTRATION 2026-08-20). The falsifier reads this
#: rather than testing equality, which would fail a measurement the registration passes.
REFINEMENT_TOLERANCE = 1

#: The two measured cells. Each names the ref that measured it: step-0.5 ran at
#: `3619016` and step-0.25 at `c37fac3`, so one provenance covering both would cite a
#: tree that never produced half the rows.
REFINEMENT_CELLS = (
    RefinementCell(
        label="step-0.5",
        rows=REFINEMENT_05_ROWS,
        provenance=Provenance(
            registered_at="86d1f22",
            measured_at="3619016",
            registered="refinement stability at |dH*| <= 1 (battery, 2026-08-20)",
        ),
    ),
    RefinementCell(
        label="step-0.25",
        rows=REFINEMENT_025_ROWS,
        provenance=Provenance(
            registered_at="86d1f22",
            measured_at="c37fac3",
            registered="refinement stability at |dH*| <= 1 (battery, 2026-08-20)",
        ),
    ),
)

#: The published values the re-measurement was checked against, and the tolerance the
#: 2026-08-21 amendment registered for them: the half-ulp of the last printed digit.
PUBLISHED_REFINEMENT_GMIN = {6: 364.6430, 7: 425.1631}
GMIN_TOLERANCE = 5e-5


def refinement_h_star(rows=REFINEMENT_05_ROWS) -> int | None:
    """The first cue-ward horizon on a recorded refinement sweep."""
    return next((r.horizon for r in rows if r.cue_ward), None)


def measure_extension(max_horizon: int = EXT_BAR) -> ExtensionMeasurement:
    """Sweep the extension set for its crossover, with a certificate either way.

    A sweep that finds no cue-ward argmin has still enumerated every horizon in full, so
    it is decided rather than sampled and carries the certificate at the last one. The
    report needs evidence whichever way the falsifier goes.
    """
    # The registered void guard, computed rather than asserted. A refined or extended
    # set that cannot reach the cue returns "no crossover" for a reason that has nothing
    # to do with the objective, and the outcome for that is NOT APPLICABLE.
    sensed = demo_maze.best_reachable_noise(EXT_SET, 1, max_horizon) <= demo_maze.R_LO
    # One enumeration per horizon, and the certificate comes off that same search. Going
    # back for it through a second `over_backend` would enumerate the set twice.
    backend, belief, preference, info_block = _setup()
    policy = ()
    for horizon in range(1, max_horizon + 1):
        search = EnumeratedEfeSearch.over_backend(
            backend, EXT_SET, info_block=info_block, horizon=horizon
        )
        policy = np.asarray(search.evaluate(belief, preference).best_policy).ravel()
        if _cue_ward(policy):
            return ExtensionMeasurement(horizon, search.certificate, policy, sensed)
    return ExtensionMeasurement(None, search.certificate, policy, sensed)


def measure_flip(horizon: int) -> FlipMeasurement:
    """Enumerate at ``horizon`` and read off everything the falsifiers need."""
    n_policies, _, _, cue_ward = argmin_at(horizon)
    backend, _, _, info_block = _setup()
    certificate = EnumeratedEfeSearch.over_backend(
        backend, ACTION_SET, info_block=info_block, horizon=horizon
    ).certificate
    walk, reach = _numpy_score(_walk(horizon)), _numpy_score(_reach(horizon))
    assert certificate.expected == n_policies  # the two routes enumerate the same set
    return FlipMeasurement(
        horizon=horizon,
        certificate=certificate,
        delta_g=walk - reach,
        bound=flip_margin_error(walk, reach),
        cue_ward=cue_ward,
    )


# --- 2. the mechanism split (post-selection exposition) ----------------------------
def mechanism_curve(max_horizon=MAX_H):
    """Per-horizon ``(H, Δε, Δc, ΔG)`` for the selected walk against the coast reach."""
    backend, belief, preference, info_block = _setup()
    rows = []
    for horizon in range(1, max_horizon + 1):
        stat = crossover_statistic(
            backend,
            belief,
            _walk(horizon),
            _reach(horizon),
            preference,
            info_block=info_block,
        )
        rows.append(
            (
                horizon,
                float(stat.delta_epsilon),
                float(stat.delta_c),
                float(stat.delta_g),
            )
        )
    return rows


# --- 3. conditioning + an independent NumPy oracle on the headline number ----------
def _host_logdet_pd(matrix) -> float:
    """``ln det`` of a symmetric positive-definite matrix, NaN otherwise.

    ``slogdet`` read only after a Cholesky succeeds, built from public names.
    """
    if not is_positive_definite(matrix):
        return float("nan")
    return float(np.linalg.slogdet(np.asarray(matrix, dtype=float))[1])


def _numpy_score(policy) -> float:
    """Pure-NumPy predict-then-contract score.

    Reuses the backend's belief propagation (separately oracle-tested) but recomputes
    the EFE kernel -- pragmatic mean, ambiguity, node-restricted info gain, contraction
    -- independently of ``_ffg_efe_step``. So agreement is a cross-implementation check
    of the scoring kernel, not of shared plumbing.
    """
    backend, belief, preference, info_block = _setup()
    observation_matrix = np.asarray(backend.model.observation_matrix)  # C
    precision = np.asarray(preference.precision)  # Lambda
    goal = np.asarray(preference.goal)
    idx = np.array(info_block)

    carry = belief
    total = 0.0
    for action in np.asarray(policy).ravel():
        predicted = backend.predicted_belief(carry, jnp.array([float(action)]))
        mean = np.asarray(predicted.mean)  # mu+
        cov = np.asarray(predicted.cov)  # Sigma+
        noise = np.asarray(backend.model.observation_noise_at(predicted.mean))  # R(mu+)
        obs = observation_matrix @ mean  # o+ = C mu+
        predicted_observation_cov = (
            observation_matrix @ cov @ observation_matrix.T + noise
        )  # S
        pragmatic = 0.5 * (obs - goal) @ precision @ (obs - goal) + 0.5 * np.trace(
            precision @ predicted_observation_cov
        )
        post_info = (
            np.linalg.inv(cov)
            + observation_matrix.T @ np.linalg.inv(noise) @ observation_matrix
        )
        cov_post_info = np.linalg.inv(post_info)  # info-form posterior, the epistemic
        # `_host_logdet_pd` is the host guard: slogdet behind a Cholesky, so a block
        # with an even number of negative eigenvalues returns NaN rather than a
        # plausible number. The kernel's `_logdet_pd` is a separate implementation that
        # reads its answer off a Cholesky factor under `jit`. They stay distinct so the
        # agreement assertion in `check()` compares two routines rather than one.
        epistemic = 0.5 * (
            _host_logdet_pd(cov[np.ix_(idx, idx)])
            - _host_logdet_pd(cov_post_info[np.ix_(idx, idx)])
        )
        total += pragmatic - epistemic
        # Kalman-form contraction for the carry; the mean stays predict-only.
        cov_post = (
            cov
            - cov
            @ observation_matrix.T
            @ np.linalg.inv(predicted_observation_cov)
            @ observation_matrix
            @ cov
        )
        carry = Belief(mean=predicted.mean, cov=jnp.asarray(cov_post))
    return total


def flip_margin_error(g_one: float, g_two: float) -> float:
    """The error the declared conditioning ceiling allows on ``g_one - g_two``.

    A float64 solve at condition number ``k`` returns a result whose relative error is
    about ``k * eps``. The rollout-hygiene discipline declares ``cond <= COND_CEILING``
    and ``tests/test_rollout_hygiene.py`` gates every ``Σ⁺``, ``S`` and ``Σ_post``
    against it, so each score is good to ``|G| * COND_CEILING * eps`` and their
    difference to twice that.

    This is the bar the flip is measured against. It is derived from a ceiling that was
    already declared and already enforced, not chosen to fit the measured margin. The
    measured conditioning is far below the ceiling (``max cond = 1003`` on the H* walk),
    so the true error is some five orders smaller again and this bound is loose in the
    safe direction.

    Args:
        g_one: one policy's summed EFE.
        g_two: the other's.

    Returns:
        A bound on the numerical error in their difference, in nats.
    """
    eps = float(np.finfo(float).eps)
    return 2.0 * max(abs(g_one), abs(g_two)) * COND_CEILING * eps


def conditioning(policy):
    """The registered rollout conditioning for a plan.

    Delegates to ``cpomdp.diagnostics.rollout_conditioning``, so the reported quantities
    -- ``cond(Σ⁺)``, ``cond(S)``, ``cond(Σ_post)``, ``min_eig(Σ_post)``, and the all-PD
    flag -- and their bars are exactly the ones ``tests/test_rollout_hygiene.py`` gates
    on. The headline margin is small in relative terms, so this is where it is shown
    benign. ``min_eig(Σ_post)`` staying above the floor means neither Cholesky guard
    fires, the kernel's ``_logdet_pd`` or this module's ``_host_logdet_pd``, and the
    epistemic never goes NaN.
    """
    backend, belief, preference, info_block = _setup()
    trace = policy_efe_ffg_trace(
        backend,
        belief,
        jnp.asarray(policy).reshape(-1, 1),
        preference,
        info_block=info_block,
    )
    return rollout_conditioning(trace)


def epistemic_counterfactual(horizon=FLIP_H, actions=None):
    """Is the flip epistemic? Compare the argmin WITH the epistemic term to WITHOUT it.

    Enumerates at ``horizon`` and returns ``(g_cue_ward, pragmatic_cue_ward,
    all_g_finite)``. When the full-``G`` argmin is cue-ward while the pragmatic-only
    argmin is still prior-ward, the epistemic term is what causes the flip here -- the
    crossover is epistemic, not a pragmatic phenomenon wearing an epistemic label.
    ``all_g_finite`` doubles as the guard check: no enumerated policy scores NaN, so the
    NaN-safe argmin is never exercised here.
    """
    actions = V1 if actions is None else actions
    backend, belief, preference, info_block = _setup()
    combos = np.array(list(itertools.product(actions, repeat=horizon)), dtype=float)

    @jax.jit
    def batch(pols):
        def one(pol):
            efe = policy_efe_ffg(
                backend, belief, pol, preference, info_block=info_block
            )
            return efe.g, efe.pragmatic

        return jax.vmap(one)(pols)

    gs, prags = [], []
    for start in range(0, combos.shape[0], 16384):
        chunk = jnp.asarray(combos[start : start + 16384]).reshape(-1, horizon, 1)
        g, prag = batch(chunk)
        gs.append(np.asarray(g))
        prags.append(np.asarray(prag))
    g_all = np.concatenate(gs)
    prag_all = np.concatenate(prags)

    def cue_ward(scores):  # NaN-safe argmin, matching EnumeratedEfeSearch
        best = int(np.argmin(np.where(np.isnan(scores), np.inf, scores)))
        return bool(np.cumsum(combos[best]).max() >= 0.5)

    return cue_ward(g_all), cue_ward(prag_all), bool(np.all(np.isfinite(g_all)))


# --- the registered falsifiers, reported by both the bare run and the gate ----------
def _bar(*measurements: FlipMeasurement) -> str:
    """The margins and the bound each was read against, one clause per horizon."""
    return "; ".join(
        f"|ΔG({m.horizon})| = {abs(m.delta_g):.4f} vs bound {m.bound:.2e} "
        f"({abs(m.delta_g) / m.bound:.0e}x)"
        for m in measurements
    )


def falsifiers(
    at_prior: FlipMeasurement | None = None,
    at_flip: FlipMeasurement | None = None,
    extension: ExtensionMeasurement | None = None,
    refinement: tuple[RefinementCell, ...] | None = None,
) -> tuple[CheckReport, ...]:
    """The five D3 falsifiers registered for this crossover, one report each.

    A falsifier does not pass. ``NOT TRIGGERED`` is "it ran and the condition did not
    obtain", so the claim survives it. ``NOT APPLICABLE`` is void by construction, so
    it is evidence for nothing and is not a survivor. ``NOT RUN HERE`` was measured
    elsewhere. Row 3 is the only one that produced no evidence: it is void by
    construction, with no observation draw to vary, so its prover cell is ``—`` because
    attributing one would claim evidence that was never produced. No row returns
    ``NOT RUN HERE`` any more.

    Rows 1, 2, 4 and 5 read on both axes at once. On the prover axis they are
    ``PROVED``, resting on exhaustive enumeration and carrying its certificates. On the
    tier axis they are ``B``, the margin read against the error ``COND_CEILING``
    allows on a difference of two scores.

    Every outcome and every detail here is computed from the measurements. The direction
    comes from ``cue_ward``, off the exhaustive argmin, so a reversed result reports
    ``FIRED`` instead of printing the old sentence beside a cleared magnitude test. A
    margin inside its bound reports ``NOT RESOLVED``, because the ordering would then be
    genuinely undetermined and the honest report of a tie is a tie.

    Row 2 quantifies over two horizons, so it takes both measurements and both
    certificates. Reading it off H* alone would let it survive on the H* margin while
    ``ΔG(H*-1)`` sat inside its own bound.

    Args:
        at_prior: the measurement at ``H*-1``. Enumerated live when omitted.
        at_flip: the measurement at ``H*``. Enumerated live when omitted.
        extension: the extension-axis sweep. Enumerated live when omitted.
        refinement: the refinement-axis cells. The recorded ones when omitted; a
            refuting set is what lets this row's FIRED branch be executed.
    """
    at_prior = measure_flip(FLIP_H - 1) if at_prior is None else at_prior
    at_flip = measure_flip(FLIP_H) if at_flip is None else at_flip
    extension = measure_extension() if extension is None else extension
    refinement = REFINEMENT_CELLS if refinement is None else refinement

    def crossover_exists() -> Outcome:
        """Fires when the exhaustive argmin is not cue-ward at H*."""
        if not at_flip.separated:
            return Outcome.NOT_RESOLVED
        return Outcome.NOT_TRIGGERED if at_flip.cue_ward else Outcome.FIRED

    def flip_is_clean() -> Outcome:
        """Fires unless the argmin is prior-ward at H*-1 and cue-ward at H*."""
        if not (at_prior.separated and at_flip.separated):
            return Outcome.NOT_RESOLVED
        clean = at_flip.cue_ward and not at_prior.cue_ward
        return Outcome.NOT_TRIGGERED if clean else Outcome.FIRED

    def extension_holds() -> Outcome:
        """Fires when the extension pushes H* past its registered bar of 6.

        A set that cannot land on the cue is void by construction, so its null is about
        the geometry and reports NOT APPLICABLE rather than counting as a survivor.
        """
        if not extension.sensed:
            return Outcome.NOT_APPLICABLE
        if extension.h_star is None:
            return Outcome.FIRED
        return Outcome.NOT_TRIGGERED if extension.h_star <= EXT_BAR else Outcome.FIRED

    def extension_warrant() -> Warrant | None:
        """The prover class the sweep earns, or None where it could not have fired.

        Exhaustive over a declared finite set, so the bar is decided rather than
        sampled. A void set produced no evidence here, so it carries no warrant at all.
        """
        if not extension.sensed:
            return None
        return extension.certificate.warrant

    def extension_detail() -> str:
        """The measured horizon, the argmin that produced it, and the bar it met."""
        if not extension.sensed:
            return (
                "v1-ext cannot land on the cue, so any null is geometry rather than a "
                "result (registered void guard, cue_maze.best_reachable_noise)"
            )
        if extension.h_star is None:
            return (
                f"no cue-ward argmin through H = {EXT_BAR} on v1-ext, so H* > "
                f"{EXT_BAR} against the registered bar"
            )
        policy = np.asarray(extension.policy).ravel()
        # Every clause after the number is derived from it. A hardcoded reading beside
        # a computed number is the defect rows 1 and 2 were rewritten to remove.
        verdict = (
            "inside the bar"
            if extension.h_star <= EXT_BAR
            else "past the bar, so the extension moved H*"
        )
        movement = (
            "H* does not move from {-3,...,2}, so extension saturates"
            if extension.h_star == EDGE_H_STAR
            else f"H* moves from {EDGE_H_STAR} on {{-3,...,2}} to {extension.h_star}"
        )
        return (
            f"H* = {extension.h_star} on v1-ext against a registered bar of "
            f"H* <= {EXT_BAR}, {verdict}. argmin {policy}. {movement}"
        )

    def refinement_deltas():
        """``(label, h_star, |dH*|)`` per cell, with ``None`` where nothing flipped."""
        rows = []
        for cell in refinement:
            h_star = refinement_h_star(cell.rows)
            delta = None if h_star is None else abs(h_star - FLIP_H)
            rows.append((cell.label, h_star, delta))
        return rows

    def refinement_holds() -> Outcome:
        """Fires when a cell moves H* past the registered tolerance.

        Reads the registered bar rather than testing equality. `|dH*| <= 1` is the
        pre-registered PASS, so a cell landing on 6 or 8 survives this falsifier and
        must not be reported as refuting it.
        """
        deltas = [delta for _, _, delta in refinement_deltas()]
        if any(delta is None for delta in deltas):
            return Outcome.FIRED
        return (
            Outcome.NOT_TRIGGERED
            if all(delta <= REFINEMENT_TOLERANCE for delta in deltas)
            else Outcome.FIRED
        )

    def refinement_detail() -> str:
        """Every clause computed: the horizons, the spread, the argmins, the scores."""
        measured = refinement_deltas()
        per_cell = ", ".join(f"{label} H* = {h_star}" for label, h_star, _ in measured)
        deltas = [delta for _, _, delta in measured]
        if any(delta is None for delta in deltas):
            return (
                f"{per_cell} against the coarse {FLIP_H}: a cell has no cue-ward "
                f"argmin at all, so the registered bar of {REFINEMENT_TOLERANCE} "
                "cannot be met"
            )
        worst = max(deltas)
        verdict = (
            f"|dH*| = {worst} within the registered bar of {REFINEMENT_TOLERANCE}"
            if worst <= REFINEMENT_TOLERANCE
            else f"|dH*| = {worst} past the registered bar of {REFINEMENT_TOLERANCE}"
        )
        every_row = [row for cell in refinement for row in cell.rows]
        sub_step = [
            row.horizon
            for row in every_row
            if any(float(a) != int(a) for a in row.policy)
        ]
        lattice = (
            "no sub-step action reaches any argmin"
            if not sub_step
            else f"a sub-step action reaches the argmin at H = {sub_step}"
        )
        by_horizon = {}
        for row in every_row:
            by_horizon.setdefault(row.horizon, set()).add(row.gmin)
        agree = all(len(scores) == 1 for scores in by_horizon.values())
        agreement = (
            "the cells agree to the digit"
            if agree
            else "the cells disagree on at least one score"
        )
        return (
            f"{per_cell} against the coarse {FLIP_H}, so {verdict}. {lattice}, and "
            f"{agreement}. Recorded from the chunked runs, reproducible with "
            "--refinement"
        )

    def where(measurement: FlipMeasurement) -> str:
        return "cue-ward" if measurement.cue_ward else "prior-ward"

    # Travels with the number everywhere it is quoted: the declared set clips the reach
    # at -2 while -3 reaches the goal in one step, so 7 is an upper bound on H*, and a
    # set containing -3 flips at 6. The error bound certifies the arithmetic, which was
    # never the exposure here.
    upper = (
        f"H* = {FLIP_H} is an upper bound "
        "(set clips the reach at -2, one-step reach -3)"
    )
    # Travels with the number for the same reason `upper` does. `evaluate` scores whole
    # length-H sequences and never re-plans, so this argmin is an open-loop one. The
    # same statistic under `RecedingHorizonSelector` is a different measurement, and a
    # row that does not name the seam reads as either (ADR-034).
    seam = "scored open-loop (EnumeratedEfeSearch.evaluate, no re-planning)"
    return (
        CheckReport(
            name="1. no crossover at feasible H",
            check_id="crossover.no_crossover_at_feasible_h",
            warrant=Warrant.PROVED,
            outcome=crossover_exists(),
            tier=Tier.BOUNDED,
            detail=(
                f"argmin is {where(at_flip)} at H = {at_flip.horizon}, inside "
                f"H_MAX = {H_MAX}, {seam}. {upper}. {_bar(at_flip)}"
            ),
            evidence=(at_flip.certificate,),
            provenance=(FLIP_PROVENANCE,),
        ),
        CheckReport(
            name="2. flip not clean at H*/H*-1",
            check_id="crossover.flip_not_clean_at_h_star",
            warrant=Warrant.PROVED,
            outcome=flip_is_clean(),
            tier=Tier.BOUNDED,
            detail=(
                f"argmin is {where(at_prior)} at H = {at_prior.horizon} and "
                f"{where(at_flip)} at H = {at_flip.horizon}, {seam}. {upper}. "
                f"{_bar(at_prior, at_flip)}"
            ),
            evidence=(at_prior.certificate, at_flip.certificate),
            provenance=(FLIP_PROVENANCE,),
        ),
        CheckReport(
            name="3. not reproducible across seeds",
            check_id="crossover.not_reproducible_across_seeds",
            warrant=None,
            outcome=Outcome.NOT_APPLICABLE,
            tier=Tier.COMPUTED,
            detail="void by construction: no observation draw to vary across seeds",
        ),
        CheckReport(
            name="4. H* unstable under refinement",
            check_id="crossover.h_star_unstable_under_refinement",
            warrant=Warrant.PROVED,
            outcome=refinement_holds(),
            tier=Tier.BOUNDED,
            detail=refinement_detail(),
            evidence=tuple(row.certificate for cell in refinement for row in cell.rows),
            provenance=tuple(cell.provenance for cell in refinement),
        ),
        CheckReport(
            name="5. H* unstable under extension",
            check_id="crossover.h_star_unstable_under_extension",
            warrant=extension_warrant(),
            outcome=extension_holds(),
            tier=Tier.BOUNDED,
            detail=extension_detail(),
            evidence=() if not extension.sensed else (extension.certificate,),
            provenance=(EXT_PROVENANCE,),
        ),
    )


def _print_falsifiers(reports: tuple[CheckReport, ...]) -> None:
    """Print the registered falsifiers on both axes, then the run's counts.

    Prover and tier print as separate columns because they answer separate questions.
    A row can be decided and have nothing stated behind its margin, or measured against
    a tight bar and only sampled. Collapsing them into one verdict loses whichever half
    the reader needed.

    Takes the reports rather than building them, so the caller's enumerations are paid
    for once.
    """
    print(f"\nRegistered D3 falsifiers ({len(reports)} registered):")
    print(f"   {'':<33} {'outcome':<15} {'prover':<13} {'tier':<5} why")
    for report in reports:
        prover = report.warrant.value if report.warrant else "—"
        print(
            f"   {report.name:<33} {report.outcome.value:<15} "
            f"{prover:<13} {report.tier.value:<5} {report.detail}"
        )
    print(f"\n{check_summary(reports)}")
    print("   full accounting: research/r10_open_loop_crossover.md, chapter 7")


# --- the four measurement tables ----------------------------------------------------
def _print_tables() -> None:
    flip = exhaustive_flip()
    mech = mechanism_curve()
    edge = exhaustive_flip(action_set=EDGE_SET, max_horizon=FLIP_H - 1)
    ext = exhaustive_flip(action_set=EXT_SET, max_horizon=EXT_BAR)

    print("1. Exhaustive argmin per horizon (selection-free, exact enumeration):")
    print(
        f"   {'H':>2} {'|A|^H':>7} {'cost |A|^H*H':>12} {'Gmin':>10} "
        f"{'argmin':<22} plan"
    )
    for horizon, n, gmin, policy, cue in flip:
        kind = "walk (cue-ward)" if cue else "reach (prior-ward)"
        print(
            f"   {horizon:>2} {n:>7} {n * horizon:>12} {gmin:>10.4f} "
            f"{policy!s:<22} {kind}"
        )
    h_star = next((h for h, _, _, _, cue in flip if cue), None)
    print(
        f"   -> the argmin flips reach -> walk at H* = {h_star}, "
        "scored open-loop (no re-planning)\n"
    )

    print("2. Mechanism (post-selection: the walk the search picked, scored back):")
    print(f"   {'H':>2} {'Δε (pull)':>11} {'Δc (grad)':>11} {'ΔG':>10}   reading")
    for horizon, de, dc, dg in mech:
        print(
            f"   {horizon:>2} {de:>11.4f} {dc:>11.4f} {dg:>10.4f}   "
            f"{'walk wins' if dg < 0 else 'reach wins'}"
        )
    pulls = [de for _, de, _, _ in mech]
    print(
        f"   -> pull flat (range {max(pulls) - min(pulls):.3f}); gradient decays "
        f"{mech[0][2]:.2f} -> {mech[-1][2]:.2f}, crossing it at H* = {FLIP_H}\n"
    )

    print(
        f"3. Registered rollout conditioning along the H={FLIP_H} walk "
        f"(bars: min_eig(Σ_post) > {MIN_EIG_FLOOR:.0e}, cond < {COND_CEILING:.0e}):"
    )
    rc = conditioning(_walk(FLIP_H))
    print(
        f"   {'k':>2} {'cond(Σ+)':>10} {'cond(S)':>9} {'cond(Σ_post)':>13} "
        f"{'minEig(Σ_post)':>15}"
    )
    for k in range(FLIP_H):
        print(
            f"   {k:>2} {rc.cond_predicted_cov[k]:>10.1f} "
            f"{rc.cond_predicted_observation_cov[k]:>9.1f} "
            f"{rc.cond_posterior_cov[k]:>13.1f} {rc.min_eig_posterior_cov[k]:>15.3e}"
        )
    max_cond = max(
        rc.cond_predicted_cov.max(),
        rc.cond_predicted_observation_cov.max(),
        rc.cond_posterior_cov.max(),
    )
    min_eig = float(rc.min_eig_posterior_cov.min())
    print(
        f"   all PD = {rc.all_positive_definite}; min eig(Σ_post) = {min_eig:.3e} "
        f"({min_eig / MIN_EIG_FLOOR:.0e}x the floor); max cond = {max_cond:.0f}"
    )
    backend, belief, preference, info_block = _setup()
    walk_ship = float(
        policy_efe_ffg(
            backend, belief, _walk(FLIP_H), preference, info_block=info_block
        ).g
    )
    walk_np, reach_np = _numpy_score(_walk(FLIP_H)), _numpy_score(_reach(FLIP_H))
    print(
        f"   NumPy oracle: ΔG = {walk_np - reach_np:+.6f} "
        f"(|ΔG|/|G| = {abs(walk_np - reach_np) / reach_np:.1e}); "
        f"|G_walk shipped-numpy| = {abs(walk_ship - walk_np):.1e} < atol 1e-9"
    )
    print("   (ln det differs: kernel Cholesky, oracle slogdet, both PD-guarded)")

    g_cue, prag_cue, finite = epistemic_counterfactual(FLIP_H)
    print(
        f"\n4. Epistemic counterfactual at H={FLIP_H} (all {5**FLIP_H} G finite = "
        f"{finite}):"
    )
    print(f"   argmin cue-ward -- with epistemic: {g_cue}; without: {prag_cue}")
    print("   -> the flat ~1.7-nat epistemic pull is what flips it at H=7; zero the")
    print("      epistemic term and the reach still wins, crossing only at H~10.")

    edge_star = next((h for h, _, _, _, cue in edge if cue), None)
    ext_star = next((h for h, _, _, _, cue in ext if cue), None)
    ext_policy = next((p for _, _, _, p, cue in ext if cue), None)
    refine_cost = 9**7 * 7
    hmax_cost = 5**H_MAX * H_MAX
    print("\n5. Action-set dependence and feasibility:")
    print(f"   one-step reach -3 -> H* = {edge_star}; the registered set clips the")
    print(f"   reach to -2, so H* = {FLIP_H} is an upper bound.")
    verdict = (
        f"PASSES (bar H* <= {EXT_BAR})"
        if ext_star is not None and ext_star <= EXT_BAR
        else f"FAILS (bar H* <= {EXT_BAR})"
    )
    # Derived like the report's, and for the same reason: a fixed sentence beside a
    # computed number reads as a finding on any measurement, including one that refutes.
    movement = (
        f"the horizon does not move from {EDGE_H_STAR}, so extension saturates"
        if ext_star == EDGE_H_STAR
        else f"the horizon moves from {EDGE_H_STAR} to {ext_star}"
    )
    print(f"   extension {{-4,...,2}} -> H* = {ext_star}, argmin {ext_policy}.")
    print(f"   registered before the run, so this {verdict}. {movement}.")
    print(f"   refinement: measured, recorded, not re-run here ({refine_cost} scored")
    print("   steps at step-0.5 alone). H* = 7 at step-0.5 and step-0.25 alike, the")
    print("   two agreeing to the digit, and no sub-step action in any argmin.")
    print("   Reproduce with --refinement.")
    print("   analytic bound: relief <= 0.77/step vs 2.77 detour -> H* >= 6.")
    print(f"   feasibility: declared to H_MAX = {H_MAX} (cost {hmax_cost} scored")
    print("   steps); the argmin is cue-ward at H = 7, 8, 9, open-loop throughout.")
    _print_falsifiers(falsifiers())


# --- the gate ----------------------------------------------------------------------
def check() -> None:
    """Assert the flip, the mechanism split, the oracle, and the kernel's inertness.

    The exhaustive enumerations are scoped to the flip boundary and H* (the horizons the
    assertions bear on), plus the extension sweep's 137,256, so the gate enumerates
    ~325k policies rather than the full one. The recorded refinement rows are read, not
    re-enumerated.
    """
    # 1. Decisive: the exhaustive argmin is a reach at H*-1 and a cue-ward walk at H*.
    #    Measured once here and handed to `falsifiers()` below, so the two enumerations
    #    are paid for once rather than per consumer.
    at_prior, at_flip = measure_flip(FLIP_H - 1), measure_flip(FLIP_H)
    assert at_prior.cue_ward is False, "argmin should still be a reach at H*-1"
    assert at_flip.cue_ward is True, "argmin should be a cue-ward walk at H*"

    # 2. Mechanism: pull flat, gradient decays below it, clean sign change at H*/H*-1.
    mech = mechanism_curve()
    pulls = [de for _, de, _, _ in mech]
    assert max(pulls) - min(pulls) < EPS_PLATEAU  # epistemic does not accumulate
    dg = {h: g for h, _, _, g in mech}
    assert dg[FLIP_H - 1] > 0  # reach still wins at H*-1
    assert dg[FLIP_H] < 0  # walk wins at H* -- a clean single sign change
    # No registered falsifier fired. A margin inside its bound reports NOT RESOLVED and
    # a reversed argmin reports FIRED, both from the measurements above, so this reads
    # the outcomes rather than re-deriving them. NOT RESOLVED and NOT APPLICABLE are
    # findings to report; only FIRED is a gate failure.
    # Enumerated once and passed in, so the sweep is not paid for twice.
    extension = measure_extension()
    reports = falsifiers(at_prior, at_flip, extension)
    fired = [r.name for r in reports if r.outcome is Outcome.FIRED]
    # The registered bar is `H* <= 6`, and the registration calls 5 plausible. Asserting
    # equality here would fail a measurement that passed its own registration.
    assert extension.sensed, "v1-ext cannot reach the cue: void by geometry"
    assert extension.h_star is not None, "no cue-ward argmin on v1-ext"
    assert extension.h_star <= EXT_BAR, (
        f"extension H* = {extension.h_star}, registered bar was <= {EXT_BAR}"
    )
    assert extension.certificate.warrant is Warrant.PROVED
    assert not fired, f"registered falsifiers fired: {fired}"
    for report in reports:
        if report.outcome is Outcome.NOT_RESOLVED:
            print(f"   NOT RESOLVED: {report.name} — {report.detail}")
    grad = {h: c for h, _, c, _ in mech}
    assert grad[1] > pulls[0]  # the gradient starts above the pull
    assert grad[FLIP_H] < pulls[-1]  # ... and has decayed below it by H*

    # 3. The headline number matches an independent NumPy kernel at H* (atol 1e-9). The
    #    oracle takes its ln det magnitude from slogdet, the shipped kernel off a
    #    Cholesky factor. Both reject a non-PD argument (see the audit tests).
    backend, belief, preference, info_block = _setup()
    for policy in (_walk(FLIP_H), _reach(FLIP_H)):
        shipped = float(
            policy_efe_ffg(backend, belief, policy, preference, info_block=info_block).g
        )
        assert abs(shipped - _numpy_score(policy)) < 1e-9

    # 4. The flip is epistemic, and the NaN guard is inert. With the epistemic term the
    #    argmin is cue-ward at H*; with it zeroed the argmin is still prior-ward, so the
    #    epistemic term is what flips it. Every enumerated G is finite.
    g_cue_ward, pragmatic_cue_ward, all_finite = epistemic_counterfactual(FLIP_H)
    assert all_finite, "some enumerated G is NaN -- the guard would be load-bearing"
    assert g_cue_ward is True  # with the epistemic term, the walk wins at H*
    assert pragmatic_cue_ward is False  # without it, the reach still wins -> epistemic

    # 5. The registered rollout conditioning clears its bars: Sigma_post stays positive
    #    definite and off the floor, so the Cholesky guard never fires (margin is real).
    rc = conditioning(_walk(FLIP_H))
    assert rc.all_positive_definite
    assert rc.min_eig_posterior_cov.min() > MIN_EIG_FLOOR
    assert rc.cond_predicted_cov.max() < COND_CEILING
    assert rc.cond_predicted_observation_cov.max() < COND_CEILING
    assert rc.cond_posterior_cov.max() < COND_CEILING

    # The equal-billing measurement (H* one sooner with the one-step reach -3) and the
    # refinement-stability check are heavier enumerations; they print in the bare run
    # and are recorded in the write-up rather than gated on every CI pass.
    print(f"Crossover at H* = {FLIP_H} on the registered set ({FLIP_H - 1} with the")
    print("one-step reach): the exhaustive argmin flips reach -> two-phase walk. The")
    print("gradient decays below a flat ~1.7-nat epistemic pull; zero it and the")
    print("flip moves to H~10, so the epistemic is load-bearing. Oracle- and")
    print("conditioning-confirmed.")
    _print_falsifiers(reports)


def print_refinement() -> None:
    """Re-run every registered refinement cell and print it, certificates included.

    Off both the test and the `--check` paths on purpose: the cheapest cell is 531,441
    policies and the dearest 410,338,673, which is minutes to hours rather than the
    seconds a gate may take. This is the reproduction route for the recorded
    constants, run by hand when the numbers are challenged.
    """
    # Exactly the recorded rows, so the route reproduces what it claims to. The 9^8
    # cell is absent on purpose: the registration made it contingent on H* rising under
    # refinement, it did not rise, and no row records it.
    cells = tuple((REFINE_05_SET, row.horizon) for row in REFINEMENT_05_ROWS) + tuple(
        (REFINE_025_SET, row.horizon) for row in REFINEMENT_025_ROWS
    )
    print("Refinement axis, chunked (peak is flat in |A|^H, ADR-036):")
    print(f"   {'set':<18} {'H':>2} {'policies':>12} {'Gmin':>12}  argmin")
    for action_set, horizon in cells:
        row = measure_refinement(action_set, horizon)
        print(
            f"   {action_set.version:<18} {row.horizon:>2} "
            f"{row.certificate.visited:>12,} {row.gmin:>12.4f}  {row.policy}"
        )
        print(f"      {row.certificate}")


def main():
    """``--check`` asserts, ``--refinement`` re-runs the recorded cells, bare prints."""
    if "--check" in sys.argv:
        check()
        return
    if "--refinement" in sys.argv:
        print_refinement()
        return
    _print_tables()


if __name__ == "__main__":
    jax.config.update("jax_enable_x64", True)
    main()
