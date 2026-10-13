"""Runs the example ``--check`` gates so their paper-facing assertions gate in CI.

Each demo's ``check()`` asserts the numbers the paper quotes. Calling them here turns
the manual ``--check`` into a regression gate: the assertions fire on every test run,
not only when the script is run by hand. Every ``check()`` here is plotting-free
(matplotlib is imported lazily in the render path), so they run in the base test
environment without the ``examples`` extra. ``conftest.py`` puts ``examples/`` and
``examples/ffg/`` on the path.
"""

import bacillus_uncertain_food
import chemotaxis_figure
import coupling_graph_figure
import crossover
import crossover_horizon_figure
import crossover_sweep
import efe_collapse_figure
import epistemic_dissociation_figure
import pytest
import surge_margin

from cpomdp.enumeration import CompletenessCertificate
from cpomdp.warrant import Outcome, Tier, Warrant, check_summary


def test_single_chain_theorem_check():
    """Theorem 1 (i)/(iii) on the single-chain model class: the epistemic value varies,
    its frozen-R twin is flat, and the observation noise traces a curve across the
    grid."""
    efe_collapse_figure.check()


def test_surge_margin_check():
    """The applied R(x) demo: a DP flow meter's own noise moves the surge control line.

    Eleven registered falsifiers over the valve sweep. The fixed-R twin's epistemic term
    is flat and its argmin G is its argmin pragmatic (the ADR-003 collapse), the R(Q)
    agent is held off that optimum, and its standoff falls monotonically across an 8:1
    ladder in meter quality while the twin's does not move. The closed-form row asserts
    autodiff `ell'(m)` against `-2/Q`, which is what ties the demo's sensor to the
    shape the gap work is written in.

    The fifth row is the control on the rest: the offset survives a 10:1 sweep of the
    economic weight, which moves the standoff further than meter quality does. That is
    what separates the demo's claim (the direction) from its output (the numbers). The
    sixth is the library's own sensor probe. The seventh checks every grid argmin
    against the root of the stationarity law, which says why: the offset is
    `kappa/(Lambda Q) + R/Q`, an epistemic part whose scale Lambda sets and a pragmatic
    part from the preference being over noisy readings. The eighth runs the library's
    own `Agent` in closed loop and checks it settles where the law puts it. The ninth
    checks that every settled rung stays clear of the preference and that a worse meter
    still settles nearer it. The last two swap in a Coriolis meter, whose noise grows
    with flow: the offset flips sign, and the library's loop settles past the surge
    line.

    Fifteen vmapped sweeps of 600 grid actions and three 40-step agent loops, each
    computed once, a few seconds, so it stays on the pull-request path.
    """
    surge_margin.check()


def test_epistemic_dissociation_check():
    """The four T-maze results, including Result 4's horizon-1 pull < gradient order."""
    epistemic_dissociation_figure.check()


def test_uncertain_food_backend_agreement():
    """`KalmanBackend` and the FFG `ChainBackend` agree on the flagship's sensor.

    The channel reads one state block while its noise is keyed on another, a topology
    neither backend's own suite covers. Both get one scripted `(observation, action)`
    sequence rather than two closed loops, so a near-tied argmin cannot send them down
    different trajectories and be misread as a disagreement.

    This is the flagship's only gate. It is also the README's hero demo, so importing
    the module here fails the suite if the demo stops loading at all.
    """
    max_mean_diff, max_cov_diff = bacillus_uncertain_food.check_backend_agreement()
    assert max_mean_diff < 1e-7, f"means diverge by {max_mean_diff:.2e}"
    assert max_cov_diff < 1e-7, f"covariances diverge by {max_cov_diff:.2e}"


def test_coupling_graph_two_routes_agree():
    """`CouplingGraph.infer` on a branching tree matches the hand-flattened Kalman.

    The demo prints the verdict, and until it was wired up here nothing consumed it: a
    disagreement printed `FAIL` and exited zero.
    """
    coupling_graph_figure.check()


def test_chemotaxis_two_routes_agree():
    """The same equivalence on the five-node chemotaxis network, hub inferred through
    its leaves. Its verdict was equally unconsumed."""
    chemotaxis_figure.check()


def test_crossover_sweep_check():
    """The constant reach/walk pair never crosses over — the search-family artefact that
    makes the exhaustive varying-sequence search necessary."""
    crossover_sweep.check()


def test_crossover_horizon_check():
    """The open-plane margin between a direct plan and a detour. The two animated
    agents choose differently. The margin crosses zero exactly once as the horizon
    grows. The epistemic pull stays flat while the pragmatic gradient decays under it,
    and a frozen-R twin never crosses at any horizon in range.

    Two 16-horizon sweeps, live and frozen. ~12s on the reference machine, under the
    marker's threshold, and this is the demo's only gate. It stays on the PR path."""
    crossover_horizon_figure.check()


def _measurement(horizon, *, delta_g, cue_ward, bound=1e-5):
    """A synthetic `FlipMeasurement`, so the reporting can be driven off any result.

    The live run only ever produces one shape, so every branch that a refuting or tied
    result would take is unreachable from it. Injecting the measurement is what turns
    those branches into something a test can execute.
    """
    return crossover.FlipMeasurement(
        horizon=horizon,
        certificate=CompletenessCertificate(
            expected=5**horizon,
            visited=5**horizon,
            warrant=Warrant.PROVED,
            action_set_size=5,
            horizon=horizon,
            action_set_version="v1",
        ),
        delta_g=delta_g,
        bound=bound,
        cue_ward=cue_ward,
    )


def _extension(h_star=6, *, sensed=True):
    """A synthetic `ExtensionMeasurement`, for the same reason as `_measurement`.

    Injecting it also keeps this class off a live enumeration. Omitting it makes
    `falsifiers` sweep `v1-ext` for real, which is 137,256 policies on what is meant to
    be the pull-request path.
    """
    horizon = h_star or 6
    return crossover.ExtensionMeasurement(
        h_star=h_star,
        certificate=CompletenessCertificate(
            expected=7**horizon,
            visited=7**horizon,
            warrant=Warrant.PROVED,
            action_set_size=7,
            horizon=horizon,
            action_set_version="v1-ext",
        ),
        policy=[1.0, -4.0] + [0.0] * (horizon - 2),
        sensed=sensed,
    )


def _refinement(h_stars=(7, 7), *, gmins=None, policies=None):
    """Synthetic refinement cells, so the refuting branch is executable.

    The recorded cells only ever describe one result. Feeding a different `H*`, a
    disagreeing score or a sub-step argmin is what turns each clause of row 4's detail
    into something a test can hold to.
    """
    cells = []
    for index, (label, h_star) in enumerate(
        zip(("step-0.5", "step-0.25"), h_stars, strict=True)
    ):
        size = 9 if index == 0 else 17
        rows = []
        for horizon in (h_star - 1, h_star):
            gmin = 100.0 + horizon
            if gmins is not None:
                gmin = gmins[index][horizon - (h_star - 1)]
            policy = [-2.0] + [0.0] * (horizon - 1)
            if horizon == h_star:
                policy = [1.0, -2.0] + [0.0] * (horizon - 2)
            if policies is not None and horizon == h_star and policies[index]:
                policy = policies[index]
            rows.append(
                crossover.RefinementRow(
                    horizon=horizon,
                    gmin=gmin,
                    policy=tuple(policy),
                    cue_ward=horizon == h_star,
                    certificate=CompletenessCertificate(
                        expected=size**horizon,
                        visited=size**horizon,
                        warrant=Warrant.PROVED,
                        action_set_size=size,
                        horizon=horizon,
                        action_set_version=f"v1-refine-{label}",
                    ),
                )
            )
        cells.append(
            crossover.RefinementCell(
                label=label,
                rows=tuple(rows),
                provenance=crossover.REFINEMENT_CELLS[index].provenance,
            )
        )
    return tuple(cells)


def _rows(at_prior, at_flip, extension=None):
    """The two measured falsifiers, keyed by name prefix."""
    reports = crossover.falsifiers(at_prior, at_flip, extension or _extension())
    return reports[0], reports[1]


class TestCrossoverFalsifierReporting:
    """The reporting logic, driven off measurements rather than off a live run."""

    # The measured shape: reach wins at H*-1, walk wins at H*, both clear of the bound.
    SURVIVING = (
        {"delta_g": +0.5, "cue_ward": False},
        {"delta_g": -0.152, "cue_ward": True},
    )

    def _pair(self, prior=None, flip=None):
        prior = {**self.SURVIVING[0], **(prior or {})}
        flip = {**self.SURVIVING[1], **(flip or {})}
        return _measurement(6, **prior), _measurement(7, **flip)

    def test_the_measured_shape_survives_both(self):
        one, two = _rows(*self._pair())
        assert one.outcome is Outcome.NOT_TRIGGERED
        assert two.outcome is Outcome.NOT_TRIGGERED

    def _row_four(self, **kwargs):
        return crossover.falsifiers(
            *self._pair(), extension=_extension(), refinement=_refinement(**kwargs)
        )[3]

    def test_a_refinement_within_the_bar_survives_row_four(self):
        # The registered test is |dH*| <= 1, so a cell landing on 6 or 8 PASSES. Strict
        # equality against 7 reported those as refuting, which is the bar the
        # pre-registration wrote and not the one the code read.
        for h_stars in ((7, 7), (6, 7), (7, 8), (6, 8)):
            row = self._row_four(h_stars=h_stars)
            assert row.outcome is Outcome.NOT_TRIGGERED, h_stars
            assert "within the registered bar" in row.detail, h_stars

    def test_a_refinement_past_the_bar_fires_row_four(self):
        row = self._row_four(h_stars=(9, 7))
        assert row.outcome is Outcome.FIRED
        assert "past the registered bar" in row.detail
        assert "within the registered bar" not in row.detail

    def test_row_four_does_not_claim_a_clean_lattice_when_one_is_not(self):
        # The "no sub-step action" clause is evidential, so it has to be computed.
        row = self._row_four(policies=([1.0, -1.5, 0.0, 0.0, 0.0, 0.0, 0.0], None))
        assert "a sub-step action reaches the argmin" in row.detail
        assert "no sub-step action" not in row.detail

    def test_row_four_does_not_claim_agreement_when_the_cells_disagree(self):
        row = self._row_four(gmins=((10.0, 20.0), (10.0, 999.0)))
        assert "the cells disagree" in row.detail
        assert "agree to the digit" not in row.detail

    def test_row_four_carries_a_provenance_per_cell(self):
        # The two cells were measured at different commits, so one ref covering both
        # would cite a tree that never produced half the rows.
        row = crossover.falsifiers()[3]
        assert len(row.provenance) == len(crossover.REFINEMENT_CELLS)
        assert {p.measured_at for p in row.provenance} == {"3619016", "c37fac3"}

    def test_an_extension_past_the_bar_fires_row_five(self):
        # Unreachable from a live run, which only produces H* = 6. Injecting the
        # measurement is what turns the refutation branch into an executable one.
        reports = crossover.falsifiers(*self._pair(), extension=_extension(7))
        assert reports[4].outcome is Outcome.FIRED
        assert "registered bar" in reports[4].detail
        # A moved horizon must not print the sentence describing the one that did not
        # move. Asserting only "registered bar" let the saturating text through, which
        # is how this survived a review round.
        assert "saturates" not in reports[4].detail
        assert "moves from" in reports[4].detail

    def test_no_crossover_on_the_extension_fires_row_five(self):
        reports = crossover.falsifiers(*self._pair(), extension=_extension(None))
        assert reports[4].outcome is Outcome.FIRED
        assert "no cue-ward argmin" in reports[4].detail
        assert "saturates" not in reports[4].detail

    def test_the_recorded_refinement_matches_the_published_numbers(self):
        # The re-measurement's whole point. `research/r10_open_loop_crossover.md`
        # published these before any commit built the set; the amendment of 2026-08-21
        # registered agreement as equality within the half-ulp of the last printed
        # digit. Cheap because the rows are recorded, not re-enumerated.
        for row in crossover.REFINEMENT_05_ROWS:
            published = crossover.PUBLISHED_REFINEMENT_GMIN[row.horizon]
            assert abs(row.gmin - published) <= crossover.GMIN_TOLERANCE, (
                f"H={row.horizon}: measured {row.gmin} against published {published}"
            )
            assert row.certificate.complete
        assert crossover.refinement_h_star() == crossover.FLIP_H
        assert (
            crossover.refinement_h_star(crossover.REFINEMENT_025_ROWS)
            == crossover.FLIP_H
        )

    def test_every_declared_set_can_land_on_the_cue(self):
        # `warrant_numbers.md` states the void guard returns R_LO on all four sets. It
        # is cheap because it enumerates reachable *positions*, not policies, so the
        # claim is computed here rather than asserted in prose.
        import cue_maze

        sets = (
            crossover.ACTION_SET,
            crossover.EXT_SET,
            crossover.REFINE_05_SET,
            crossover.REFINE_025_SET,
        )
        for action_set in sets:
            sharpest = cue_maze.best_reachable_noise(action_set, 1, crossover.FLIP_H)
            assert sharpest <= cue_maze.R_LO, (
                f"{action_set.version} cannot reach the cue: {sharpest}"
            )

    def test_the_two_refinement_cells_agree_exactly(self):
        # 86x the policies and twice the actions, and the scores do not move a digit.
        # Byte-identity is expected where the argmin lies in the coarser set, which is
        # the code-correctness half. The evidential half is that it does lie there.
        by_horizon = {r.horizon: r for r in crossover.REFINEMENT_025_ROWS}
        for coarse in crossover.REFINEMENT_05_ROWS:
            fine = by_horizon[coarse.horizon]
            assert fine.gmin == coarse.gmin, coarse.horizon
            assert fine.policy == coarse.policy, coarse.horizon
            assert fine.certificate.complete

    def test_no_half_step_action_reaches_the_recorded_argmins(self):
        # The evidential half: byte-identity to the coarse set is expected because it
        # is a subset. What refinement could have shown is an intermediate action
        # scoring lower, and none does.
        for row in (*crossover.REFINEMENT_05_ROWS, *crossover.REFINEMENT_025_ROWS):
            assert all(float(a) == int(a) for a in row.policy), row.policy

    def test_a_set_that_cannot_sense_the_cue_is_void(self):
        # The registered void guard. A null from a set that cannot reach the cue is
        # geometry, so it is not a survivor and carries no warrant.
        reports = crossover.falsifiers(
            *self._pair(), extension=_extension(sensed=False)
        )
        assert reports[4].outcome is Outcome.NOT_APPLICABLE
        assert reports[4].warrant is None
        assert reports[4].evidence == ()

    def test_both_rows_name_the_action_mode(self):
        # `RecedingHorizonSelector` and `OpenLoopSelector` genuinely differ, and the
        # same sentence reads true under either, so a row that does not name its seam
        # is quotable as the mode the reader had in mind. The ledger requires R10 under
        # its seam. Without this the `{seam}` interpolation can be deleted and the whole
        # suite stays green, which is how the qualifier would rot.
        for report in _rows(*self._pair()):
            assert "open-loop" in report.detail
            assert "no re-planning" in report.detail

    def test_a_reversed_flip_fires_row_one(self):
        # The defect this replaces: `abs(ΔG) > bound` cleared on a reversed result too,
        # so a refutation printed as a survivor beside a hardcoded "argmin is cue-ward".
        one, _ = _rows(*self._pair(flip={"delta_g": +0.152, "cue_ward": False}))
        assert one.outcome is Outcome.FIRED
        assert "prior-ward at H = 7" in one.detail

    def test_a_flip_that_was_already_cue_ward_fires_row_two(self):
        # Cue-ward at both horizons is not a clean flip, however wide the margins.
        _, two = _rows(*self._pair(prior={"delta_g": -0.4, "cue_ward": True}))
        assert two.outcome is Outcome.FIRED

    def test_a_tie_at_the_flip_resolves_neither_row(self):
        one, two = _rows(*self._pair(flip={"delta_g": 1e-9, "cue_ward": True}))
        assert one.outcome is Outcome.NOT_RESOLVED
        assert two.outcome is Outcome.NOT_RESOLVED

    def test_a_tie_at_the_prior_horizon_resolves_only_row_two(self):
        # Row 2 quantifies over both horizons, row 1 over H* alone. Reading row 2 off
        # H* would let it survive while ΔG(H*-1) sat inside its own bound.
        one, two = _rows(*self._pair(prior={"delta_g": 1e-9, "cue_ward": False}))
        assert one.outcome is Outcome.NOT_TRIGGERED
        assert two.outcome is Outcome.NOT_RESOLVED

    def test_row_two_carries_a_certificate_for_each_horizon(self):
        _, two = _rows(*self._pair())
        assert [c.expected for c in two.evidence] == [5**6, 5**7]

    def test_row_one_carries_the_flip_horizon_certificate(self):
        one, _ = _rows(*self._pair())
        assert [c.expected for c in one.evidence] == [5**7]

    def test_the_detail_is_generated_from_the_measurement(self):
        # Nothing about the result is hardcoded, so a changed measurement changes the
        # sentence rather than leaving a stale one beside a new outcome.
        one, _ = _rows(*self._pair(flip={"delta_g": -0.9, "cue_ward": True}))
        assert "0.9000" in one.detail

    def test_a_fired_row_reaches_the_summary(self):
        reports = crossover.falsifiers(
            *self._pair(flip={"delta_g": +0.152, "cue_ward": False}),
            extension=_extension(),
        )
        # A reversed flip fires both: row 1 because the argmin is not cue-ward at H*,
        # row 2 because prior-ward-then-cue-ward is what "clean" means.
        summary = check_summary(reports)
        assert "FIRED" in summary
        assert "2 fired" in summary


@pytest.mark.slow
def test_crossover_falsifiers_are_reports():
    """The five registered falsifiers on a live measurement of the H* boundary.

    The `PROVED` rows must carry the real completeness certificates. A fabricated one
    would satisfy the constructor and claim an enumeration that never ran, which is the
    failure the precondition exists to catch. Enumerating both horizons is the cost
    `check()` already pays, so this is marked slow and gates on merge; the reporting
    logic itself is covered on the pull-request path by the class above.
    """
    reports = crossover.falsifiers()
    assert len(reports) == 5
    # The number `warrant_numbers.md` records for the extension cell. A change here is a
    # change to a published result, so it moves with the diff that justifies it.
    extension = crossover.measure_extension()
    assert extension.h_star == 6
    assert extension.sensed
    proved = [r for r in reports if r.warrant is Warrant.PROVED]
    assert len(proved) == 4
    for report in proved:
        assert report.evidence
        # Every PROVED row names where its bar was registered, so a reader can check
        # the ordering rather than take it. Rows 1 and 2 registered and measured in one
        # commit and say so; row 5 was registered a commit ahead of its measurement.
        assert report.provenance
        # Evidence widened to two kinds when `SymbolicReduction` landed. These rows are
        # decided by enumeration, so which kind they carry is part of the assertion.
        certificates = [
            item
            for item in report.evidence
            if isinstance(item, CompletenessCertificate)
        ]
        assert len(certificates) == len(report.evidence)
        assert all(certificate.complete for certificate in certificates)
        # The other axis: decided by enumeration, and measured against a stated bar.
        # A `BOUNDED` row has to name the bar, or the tier is a label with nothing under
        # it.
        assert report.tier is Tier.BOUNDED
        assert report.outcome is Outcome.NOT_TRIGGERED

    # The crossover rows carry a qualifier the extension row does not: H* = 7 is an
    # upper bound because the declared set clips the reach at -2, and a `BOUNDED` row
    # reads stronger than the claim is without it. `bound` is what the margin was read
    # against. Row 5 is measured against a registered bar instead, so it says `bar`.
    for report in proved:
        if report.name.startswith(("1.", "2.")):
            assert "bound" in report.detail
            assert "upper bound" in report.detail
        else:
            # Rows 4 and 5 are measured against a registered bar instead of carrying
            # the upper-bound qualifier, so they say `bar`.
            assert "registered bar" in report.detail

    # Only row 3 produces no evidence now: it is void by construction, with no
    # observation draw to vary. Row 4 was NOT_RUN_HERE until the step-0.5 refinement was
    # measured, and a row that has been run carries its warrant.
    unrun = {r.outcome for r in reports if r.warrant is None}
    assert unrun == {Outcome.NOT_APPLICABLE}

    summary = check_summary(reports)
    assert "PROVED" in summary
    assert "FIRED" not in summary
    assert "5 registered, 4 tested here, none fired" in summary


@pytest.mark.slow
def test_crossover_check():
    """The exhaustive argmin flips reach -> two-phase walk at H*=7: the crossover, its
    flat-pull / decaying-gradient mechanism, and the headline number against a NumPy
    oracle. Enumerates ~325k policies (H=6, H=7), so it is marked slow and deselected on
    PRs; it gates on merge-to-main and release."""
    crossover.check()
