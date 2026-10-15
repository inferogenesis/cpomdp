"""The shared `Warrant` vocabulary and the check report that carries it.

`Warrant` has three levels: `PROVED` (exhaustive enumeration), `CERTIFIED` (validated
numerics over a compact domain) and `CORROBORATED` (a grid sample). It lives in
`warrantlib`, where a check suite can reach it without cpomdp.

`CheckReport` is what a registered falsifier emits: a warrant, an `Outcome`, a `Tier`,
and a reason. A falsifier does not pass, so `PASS` is not in the vocabulary at all. The
prover column disambiguates warrant, never the outcome. A check that never ran carries
no warrant, because attributing one claims evidence it did not produce.

The vocabulary lives in `warrantlib`. cpomdp's top level re-exports `Warrant` alone,
the type its selectors and searches report. `cpomdp.enumeration` keeps
`CompletenessCertificate`, which v0.4.4 shipped there.
"""

import dataclasses
from typing import Any

import pytest

from cpomdp.enumeration import (
    CompletenessCertificate,
    EnumeratedEfeSearch,
    FiniteActionSet,
    OpenLoopSelector,
    RecedingHorizonSelector,
)
from cpomdp.selection import EFESelector, Preference
from cpomdp.types import Belief, LinearGaussianModel
from warrantlib import (
    CheckReport,
    Outcome,
    Provenance,
    SymbolicReduction,
    Tier,
    Warrant,
    check_summary,
)


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


def _finite_set(actions):
    return FiniteActionSet([[a] for a in actions], version="test-v1")


def _certificate():
    """A complete certificate: the evidence an enumerated `PROVED` claim carries."""
    return CompletenessCertificate(
        expected=4,
        visited=4,
        warrant=Warrant.PROVED,
        action_set_size=2,
        horizon=2,
        action_set_version="test-v1",
    )


def _reduction():
    """A filled-in reduction: the evidence a Prover 2 claim carries.

    That registration file carries two results dated 2026-08-07, one giving `c₂` in
    closed form and one asserting `c₂ > 0` at leading order. A date alone points at both
    and establishes neither, so the correspondence names the heading too.
    """
    return SymbolicReduction(
        claim="the σ² coefficient of the inference gap is ℓ'(μ)²/4",
        correspondence=(
            "research/gate_d4_registration.md: RESULT 2026-08-07 (c₂ in closed form)"
        ),
        assumptions=("R smooth and positive at μ", "formal in σ, no convergence claim"),
    )


def _provenance():
    """A provenance whose two refs differ, so the ordering marker does not fire."""
    return Provenance(
        registered_at="a76cf1b",
        measured_at="9baaa22",
        registered="RESULT 2026-08-07, c₂ in closed form",
    )


def _same_ref_provenance():
    """A provenance registered and measured at one ref, which history cannot order."""
    return Provenance(
        registered_at="9baaa22",
        measured_at="9baaa22",
        registered="the truncation operator, defined in this module's docstring",
    )


def _rows(summary):
    """A summary's count rows as `(warrant, outcome, count)`, header dropped.

    The rows are column-aligned, so a substring check cannot say which row a count sits
    in. Splitting on the warrant word and the trailing integer reads the fields back
    without pinning the column widths.
    """
    parsed = []
    for line in summary.splitlines()[1:]:
        warrant, rest = line.split(maxsplit=1)
        outcome, count = rest.rsplit(maxsplit=1)
        parsed.append((warrant, outcome, int(count)))
    return parsed


class TestWarrantLevels:
    def test_the_three_prover_classes(self):
        assert Warrant.PROVED.value == "PROVED"
        assert Warrant.CERTIFIED.value == "CERTIFIED"
        assert Warrant.CORROBORATED.value == "CORROBORATED"

    def test_no_fourth_level(self):
        # A fourth member is a fourth prover class: a decision, not an edit.
        assert [w.name for w in Warrant] == ["PROVED", "CERTIFIED", "CORROBORATED"]

    def test_round_trips_through_its_value(self):
        # Checks report by value, so a value read back must land on the same member.
        for level in Warrant:
            assert Warrant(level.value) is level


class TestOutcome:
    def test_a_falsifier_never_passes(self):
        # Standing rule 6 is satisfied by not printing PASS, not by printing PASS
        # beside a column that disambiguates it. The word must not be reachable.
        values = {outcome.value for outcome in Outcome}
        assert "PASS" not in values
        assert not any(name == "PASS" for name in Outcome.__members__)

    def test_the_five_outcomes(self):
        assert Outcome.NOT_TRIGGERED.value == "NOT TRIGGERED"
        assert Outcome.FIRED.value == "FIRED"
        assert Outcome.NOT_RESOLVED.value == "NOT RESOLVED"
        assert Outcome.NOT_APPLICABLE.value == "NOT APPLICABLE"
        assert Outcome.NOT_RUN_HERE.value == "NOT RUN HERE"

    def test_no_sixth_outcome(self):
        assert [o.name for o in Outcome] == [
            "NOT_TRIGGERED",
            "FIRED",
            "NOT_RESOLVED",
            "NOT_APPLICABLE",
            "NOT_RUN_HERE",
        ]

    def test_the_three_that_did_not_run_stay_distinct(self):
        # Void by construction, measured elsewhere, and a genuine tie are three
        # different things. Collapsing them loses the survivor accounting.
        assert (
            len({Outcome.NOT_APPLICABLE, Outcome.NOT_RUN_HERE, Outcome.NOT_RESOLVED})
            == 3
        )

    def test_round_trips_through_its_value(self):
        for outcome in Outcome:
            assert Outcome(outcome.value) is outcome


class TestTier:
    def test_the_three_tiers(self):
        # EXACT: closed-form reference at machine precision. BOUNDED: a stated bar or
        # certified bracket. COMPUTED: no statable bar.
        assert Tier.EXACT.value == "exact"
        assert Tier.BOUNDED.value == "bounded"
        assert Tier.COMPUTED.value == "computed"

    def test_no_fourth_tier(self):
        assert [t.name for t in Tier] == ["EXACT", "BOUNDED", "COMPUTED"]

    def test_no_letter_stands_in_for_a_tier(self):
        # The letters ranked by position, so a reader had to know the ordering to read
        # a row. Each value now says what it means on its own.
        assert not {"A", "B", "C"} & {tier.value for tier in Tier}
        assert not {"A", "B", "C"} & set(Tier.__members__)


class TestCheckReport:
    def _report(
        self,
        *,
        warrant=Warrant.CORROBORATED,
        outcome=Outcome.NOT_TRIGGERED,
        tier=Tier.COMPUTED,
        evidence=(),
        provenance=None,
    ):
        # Defaults to the report that needs no evidence: a sample, computed, green.
        # PROVED also requires a provenance; these cases are about the other fields, so
        # one is supplied unless the case names its own.
        if provenance is None:
            provenance = (_provenance(),) if warrant is Warrant.PROVED else ()
        return CheckReport(
            name="flip-decided",
            check_id="tests.flip_decided",
            warrant=warrant,
            outcome=outcome,
            tier=tier,
            detail="exhaustive argmin over crossover-v1^7 is cue-ward",
            evidence=evidence,
            provenance=provenance,
        )

    def test_carries_all_four_labels(self):
        report = self._report()
        assert report.name == "flip-decided"
        assert report.warrant is Warrant.CORROBORATED
        assert report.outcome is Outcome.NOT_TRIGGERED
        assert report.tier is Tier.COMPUTED

    def test_detail_is_required(self):
        # A check that cannot say why it reports what it reports is a bare outcome
        # with extra fields. Omitting the reason must not construct.
        with pytest.raises(TypeError):
            CheckReport(  # ty: ignore[missing-argument]
                name="flip-decided",
                warrant=Warrant.CORROBORATED,
                outcome=Outcome.NOT_TRIGGERED,
                tier=Tier.COMPUTED,
            )

    def test_is_frozen(self):
        # A report is a record of what a check found. Editing one after the fact is
        # editing the finding.
        with pytest.raises(dataclasses.FrozenInstanceError):
            self._report().outcome = Outcome.FIRED

    def test_evidence_is_absent_by_default(self):
        assert self._report().evidence == ()

    def test_carries_the_evidence_it_was_given(self):
        cert = _certificate()
        report = self._report(warrant=Warrant.PROVED, tier=Tier.EXACT, evidence=(cert,))
        assert report.evidence == (cert,)

    def test_warrant_and_outcome_are_independent(self):
        # The pairing this vocabulary exists for: a run that survived every falsifier
        # and decided nothing.
        report = self._report()
        assert report.outcome is Outcome.NOT_TRIGGERED
        assert report.warrant is Warrant.CORROBORATED

    def test_renders_one_line_naming_all_of_it(self):
        # The tier reads as its own word. `"C"` passed here as a substring of
        # CORROBORATED, so the row could have lost the tier and the check stayed green.
        line = str(self._report())
        assert line.count("\n") == 0
        assert line == (
            "flip-decided: NOT TRIGGERED (CORROBORATED, tier computed). "
            "exhaustive argmin over crossover-v1^7 is cue-ward"
        )

    def test_a_check_with_no_warrant_renders_a_dash(self):
        line = str(self._report(warrant=None, outcome=Outcome.NOT_APPLICABLE))
        assert "—" in line


class TestCheckId:
    """The key a ledger and a manifest join on, kept apart from the prose name."""

    def _report(self, check_id):
        return CheckReport(
            name="flip-decided",
            check_id=check_id,
            warrant=Warrant.CORROBORATED,
            outcome=Outcome.NOT_TRIGGERED,
            tier=Tier.COMPUTED,
            detail="exhaustive argmin over crossover-v1^7 is cue-ward",
        )

    def test_the_id_is_required(self):
        # A report with no id cannot be reconciled against a manifest or matched to
        # itself in a later run, which is every job the id exists for.
        with pytest.raises(TypeError):
            CheckReport(  # ty: ignore[missing-argument]
                name="flip-decided",
                warrant=Warrant.CORROBORATED,
                outcome=Outcome.NOT_TRIGGERED,
                tier=Tier.COMPUTED,
                detail="exhaustive argmin over crossover-v1^7 is cue-ward",
            )

    def test_it_is_carried_beside_the_name(self):
        report = self._report("gap_series.c2_closed_form")
        assert report.check_id == "gap_series.c2_closed_form"
        assert report.name == "flip-decided"

    @pytest.mark.parametrize(
        "check_id",
        [
            "gap_series.c2_closed_form",
            "series_kernel.k5_first_cumulant",
            "R10.crossover_flip",
            "c2",
            "A1",
            "a.b.c.d",
        ],
    )
    def test_a_key_shaped_id_constructs(self, check_id):
        assert self._report(check_id).check_id == check_id

    @pytest.mark.parametrize(
        "check_id",
        [
            "K5 first cumulant is the mean",  # the prose name, which is not a key
            "gap-series.c2",  # hyphen
            "gap_series..c2",  # empty segment
            ".gap_series",  # leading dot
            "gap_series.",  # trailing dot
            "",  # nothing at all
            "gap_series.c₂",  # the maths glyph the prose uses
            "gap_series/c2",  # a path
            "gap series",  # a space
        ],
    )
    def test_an_id_that_is_not_a_key_does_not_construct(self, check_id):
        with pytest.raises(ValueError, match="check_id"):
            self._report(check_id)

    def test_a_line_break_does_not_construct(self):
        with pytest.raises(ValueError, match="check_id"):
            self._report("gap_series.c2\nc4")

    @pytest.mark.parametrize("value", [None, 3, b"gap_series.c2", ["gap_series.c2"]])
    def test_an_id_that_is_not_text_does_not_construct(self, value):
        with pytest.raises(ValueError, match="check_id"):
            self._report(value)

    def test_the_message_says_why_a_key_is_not_the_prose_name(self):
        with pytest.raises(ValueError, match="not a key") as raised:
            self._report("K5 first cumulant is the mean")
        message = str(raised.value)
        assert "check_id" in message
        assert "flip-decided" in message  # which check, so a run names the offender


class TestNameAndDetailAreReadableToo:
    """The two prose fields are held to the rule the other text fields already are.

    A blank name is a summary row nobody can attribute and a blank detail is the bare
    outcome the `detail` field exists to prevent. Both used to construct, so the shipped
    schema refused records the constructor had already accepted.
    """

    def _report(self, *, name: Any = "flip-decided", detail: Any = "sampled, no bar"):
        return CheckReport(
            name=name,
            check_id="crossover.flip_not_clean_at_h_star",
            warrant=Warrant.CORROBORATED,
            outcome=Outcome.NOT_TRIGGERED,
            tier=Tier.COMPUTED,
            detail=detail,
        )

    @pytest.mark.parametrize("field", ["name", "detail"])
    @pytest.mark.parametrize("blank", ["", "   ", "\u200b", "\ufeff"])
    def test_a_blank_field_does_not_construct(self, field, blank):
        with pytest.raises(ValueError, match=field):
            self._report(**{field: blank})

    @pytest.mark.parametrize("field", ["name", "detail"])
    def test_a_line_break_does_not_construct(self, field):
        # Both render on one line beside the check, so a second line arrives in the
        # middle of a summary row.
        with pytest.raises(ValueError, match=field):
            self._report(**{field: "first\nsecond"})

    @pytest.mark.parametrize("field", ["name", "detail"])
    def test_a_field_that_is_not_text_does_not_construct(self, field):
        with pytest.raises(ValueError, match=field):
            self._report(**{field: None})


class TestChecksThatNeverRanCarryNoWarrant:
    """A check that produced no evidence has no prover verdict to report."""

    def _report(self, *, warrant, outcome):
        return CheckReport(
            name="void-check",
            check_id="tests.flip_decided",
            warrant=warrant,
            outcome=outcome,
            tier=Tier.COMPUTED,
            detail="why",
        )

    @pytest.mark.parametrize("outcome", [Outcome.NOT_APPLICABLE, Outcome.NOT_RUN_HERE])
    def test_a_check_that_never_ran_may_not_claim_a_prover(self, outcome):
        # CORROBORATED means sampling-grade evidence was obtained. A falsifier void by
        # construction sampled nothing, and one measured elsewhere sampled nothing
        # here, so either would be attributing a warrant to a check that produced none.
        with pytest.raises(ValueError, match="no evidence here"):
            self._report(warrant=Warrant.CORROBORATED, outcome=outcome)

    @pytest.mark.parametrize("outcome", [Outcome.NOT_APPLICABLE, Outcome.NOT_RUN_HERE])
    def test_none_is_the_representable_state(self, outcome):
        assert self._report(warrant=None, outcome=outcome).warrant is None

    @pytest.mark.parametrize(
        "outcome",
        [Outcome.NOT_TRIGGERED, Outcome.FIRED, Outcome.NOT_RESOLVED],
    )
    def test_a_check_that_ran_may_carry_one(self, outcome):
        # A genuine tie ran and produced evidence. It stays warrantable.
        report = self._report(warrant=Warrant.CORROBORATED, outcome=outcome)
        assert report.warrant is Warrant.CORROBORATED


class TestSymbolicReduction:
    """What backs a Prover 2 claim: the correspondence a CAS cannot supply.

    A CAS checks that one expression equals another. Whether those expressions are the
    ones the analytic claim is about is a human obligation (`research/warrant_ledger.md`
    section 1), so the type exists to make an unrecorded obligation unrepresentable
    rather than merely discouraged.
    """

    def test_carries_the_claim_and_where_a_human_checked_it(self):
        reduction = _reduction()
        assert reduction.claim.startswith("the σ² coefficient")
        assert "RESULT 2026-08-07 (c₂ in closed form)" in reduction.correspondence
        assert len(reduction.assumptions) == 2

    def test_is_frozen(self):
        # Evidence edited after the check ran is not the evidence the check had.
        with pytest.raises(dataclasses.FrozenInstanceError):
            _reduction().claim = "something else"

    def test_assumptions_are_absent_by_default(self):
        bare = SymbolicReduction(claim="c₂ = ℓ₁²/4", correspondence="RESULT")
        assert bare.assumptions == ()

    def test_assumptions_must_be_a_tuple(self):
        # A bare string is a sequence, so it would record one assumption per character
        # and the scope of the identity would read as gibberish rather than as wrong.
        with pytest.raises(ValueError, match="tuple"):
            SymbolicReduction(
                claim="c₂ = ℓ₁²/4",
                correspondence="registration RESULT 2026-08-07",
                assumptions="R smooth at μ",  # ty: ignore[invalid-argument-type]
            )

    def test_a_claim_nobody_stated_does_not_construct(self):
        # An empty field makes the type evidence-shaped with nothing in it, which is
        # the failure `CheckReport` rejects one level up. Catching it here stops
        # "PROVED needs evidence" from being satisfiable by a blank.
        with pytest.raises(ValueError, match="claim"):
            SymbolicReduction(claim="   ", correspondence="registration RESULT")

    def test_a_correspondence_nobody_established_does_not_construct(self):
        with pytest.raises(ValueError, match="correspondence"):
            SymbolicReduction(claim="c₂ = ℓ₁²/4", correspondence="")

    @pytest.mark.parametrize(
        "blank",
        ["", "   ", "​", "­­", "﻿ ⁠", "\t\n "],
        ids=["empty", "spaces", "zero-width", "soft-hyphens", "mark-joiner", "control"],
    )
    def test_a_field_of_invisible_characters_is_blank(self, blank):
        # `str.strip()` removes whitespace and leaves the zero-width formatting
        # characters behind, so a presence check resting on it alone accepts a claim
        # that puts nothing on the page. Blank here means blank to a reader.
        with pytest.raises(ValueError, match="blank claim"):
            SymbolicReduction(claim=blank, correspondence="registration RESULT")

    @pytest.mark.parametrize("value", [None, 3, ("c₂ = ℓ₁²/4",)])
    def test_a_claim_that_is_not_text_does_not_construct(self, value):
        # Not a string means nothing renders it, and `.strip()` on its own would raise
        # AttributeError rather than say which field was wrong.
        with pytest.raises(ValueError, match="claim"):
            SymbolicReduction(
                claim=value,
                correspondence="registration RESULT 2026-08-07",
            )

    @pytest.mark.parametrize("value", [None, 3, ("RESULT",)])
    def test_a_correspondence_that_is_not_text_does_not_construct(self, value):
        with pytest.raises(ValueError, match="correspondence"):
            SymbolicReduction(
                claim="c₂ = ℓ₁²/4",
                correspondence=value,
            )

    def test_a_line_break_in_the_claim_does_not_construct(self):
        # The reduction renders as one line beside the check it backs, so a second line
        # would arrive in the middle of a summary row.
        with pytest.raises(ValueError, match="line break in claim"):
            SymbolicReduction(
                claim="c₂ = ℓ₁²/4\nand c₄ separately",
                correspondence="registration RESULT 2026-08-07",
            )

    def test_a_trailing_newline_does_not_construct(self):
        # `strip()` hides this one from a blank check while the stored field keeps it,
        # so the render breaks on a field that looked filled in.
        with pytest.raises(ValueError, match="line break in correspondence"):
            SymbolicReduction(claim="c₂ = ℓ₁²/4", correspondence="RESULT\n")

    def test_a_blank_assumption_does_not_construct(self):
        # An empty entry renders as a gap in the scope list rather than as a caveat,
        # and the message names which entry so a long list can be corrected.
        with pytest.raises(ValueError, match="blank assumption 2"):
            SymbolicReduction(
                claim="c₂ = ℓ₁²/4",
                correspondence="registration RESULT",
                assumptions=("R smooth at μ", "  "),
            )

    def test_an_assumption_that_is_not_text_does_not_construct(self):
        with pytest.raises(ValueError, match="assumption 1"):
            SymbolicReduction(
                claim="c₂ = ℓ₁²/4",
                correspondence="registration RESULT",
                assumptions=(None,),  # ty: ignore[invalid-argument-type]
            )

    def test_an_assumption_with_a_line_break_does_not_construct(self):
        with pytest.raises(ValueError, match="line break in assumption 1"):
            SymbolicReduction(
                claim="c₂ = ℓ₁²/4",
                correspondence="registration RESULT",
                assumptions=("R smooth at μ\nand positive there",),
            )

    def test_renders_one_line_naming_the_claim_and_the_correspondence(self):
        line = str(_reduction())
        assert line.count("\n") == 0
        assert "ℓ'(μ)²/4" in line
        assert "RESULT 2026-08-07 (c₂ in closed form)" in line

    def test_renders_the_assumptions_it_carries(self):
        # The scope travels with the evidence, so a reader sees it without reading the
        # algebra.
        line = str(_reduction())
        assert "R smooth and positive at μ" in line
        assert "no convergence claim" in line

    def test_an_unscoped_reduction_says_so(self):
        # Silence would read as an unconditional identity.
        line = str(SymbolicReduction(claim="c₂ = ℓ₁²/4", correspondence="RESULT"))
        assert "no assumptions recorded" in line


class TestProvenance:
    """Which ref registered a claim, and which one measured it."""

    def test_carries_both_refs_and_what_was_registered(self):
        prov = _provenance()
        assert prov.registered_at == "a76cf1b"
        assert prov.measured_at == "9baaa22"
        assert "closed form" in prov.registered

    def test_is_frozen(self):
        # Editing a provenance after the check ran is editing the ordering claim.
        with pytest.raises(dataclasses.FrozenInstanceError):
            _provenance().registered_at = "deadbee"

    @pytest.mark.parametrize(
        "ref",
        [
            "9baaa22",
            "9baaa22f31c7e0a5d4b8c6f2e1a09d3b5c7e8f01",
            "https://github.com/inferogenesis/cpomdp/commit/9baaa22",
            "10.5281/zenodo.1234567",
            "doi:10.5281/zenodo.1234567",
        ],
        ids=["short sha", "full sha", "url", "doi", "prefixed doi"],
    )
    def test_a_ref_that_resolves_to_a_fixed_thing_constructs(self, ref):
        assert (
            Provenance(
                registered_at=ref, measured_at="9baaa22", registered="the bar"
            ).registered_at
            == ref
        )

    @pytest.mark.parametrize(
        "ref",
        [
            "research/gate_d4_registration.md",
            "main",
            "65-gap-series-c4",
            "v0.4.5",
            "HEAD",
            "HEAD~3",
            "9baaa2",
            "the commit before the refit",
        ],
        ids=[
            "a path",
            "a branch",
            "a topic branch",
            "a tag",
            "HEAD",
            "a revision expression",
            "six hex digits",
            "prose",
        ],
    )
    def test_a_ref_naming_something_that_moves_does_not_construct(self, ref):
        # A path or a branch satisfies a presence check exactly as well as a commit
        # does, and sends a reviewer to a different tree every time it is resolved.
        with pytest.raises(ValueError, match="resolves to nothing fixed"):
            Provenance(registered_at=ref, measured_at="9baaa22", registered="the bar")

    def test_the_measured_ref_is_held_to_the_same_rule(self):
        with pytest.raises(ValueError, match="resolves to nothing fixed"):
            Provenance(registered_at="9baaa22", measured_at="main", registered="x")

    @pytest.mark.parametrize("value", [None, 3, ("9baaa22",)])
    def test_a_ref_that_is_not_text_does_not_construct(self, value):
        with pytest.raises(ValueError, match="registered_at"):
            Provenance(registered_at=value, measured_at="9baaa22", registered="x")

    def test_a_blank_statement_does_not_construct(self):
        # A bare ref sends a reviewer to a diff and leaves them to guess which part of
        # it was the registration.
        with pytest.raises(ValueError, match="blank registered"):
            Provenance(registered_at="a76cf1b", measured_at="9baaa22", registered="  ")

    def test_a_line_break_in_a_ref_does_not_construct(self):
        with pytest.raises(ValueError, match="line break in registered_at"):
            Provenance(registered_at="a76cf1b\n", measured_at="9baaa22", registered="x")

    def test_distinct_refs_render_both_and_claim_an_ordering(self):
        line = str(_provenance())
        assert "registered at a76cf1b" in line
        assert "measured at 9baaa22" in line
        assert "not established" not in line

    def test_one_ref_says_history_orders_nothing(self):
        # Registering and measuring in one commit is allowed. What is refused is
        # letting it read as an ordering the history establishes.
        line = str(_same_ref_provenance())
        assert "registered and measured at 9baaa22" in line
        assert "not established by history" in line

    def test_the_same_ref_is_visible_without_reading_the_render(self):
        assert _same_ref_provenance().same_ref is True
        assert _provenance().same_ref is False

    def test_an_abbreviated_ref_is_the_same_ref(self):
        # Otherwise lengthening one of the two hashes walks away from the marker while
        # naming the same commit.
        prov = Provenance(
            registered_at="9baaa22",
            measured_at="9baaa22f31c",
            registered="the truncation operator",
        )
        assert prov.same_ref is True

    def test_two_different_refs_of_equal_length_are_not_the_same(self):
        prov = Provenance(
            registered_at="a76cf1b", measured_at="9baaa22", registered="x"
        )
        assert prov.same_ref is False

    def test_renders_one_line(self):
        # It sits beside the check it backs, so a second line lands mid-summary.
        assert str(_same_ref_provenance()).count("\n") == 0


class TestProvedNeedsEvidence:
    """`PROVED` with nothing behind it does not construct."""

    def _report(
        self, *, warrant, evidence=(), outcome=Outcome.NOT_TRIGGERED, provenance=None
    ):
        # These cases are about evidence, so a valid provenance rides along unless the
        # case is about provenance itself and passes its own.
        if provenance is None:
            provenance = (_provenance(),) if warrant is Warrant.PROVED else ()
        return CheckReport(
            name="flip-decided",
            check_id="tests.flip_decided",
            warrant=warrant,
            outcome=outcome,
            tier=Tier.EXACT,
            detail="exhaustive argmin over crossover-v1^7 is cue-ward",
            evidence=evidence,
            provenance=provenance,
        )

    def test_proved_without_evidence_does_not_construct(self):
        with pytest.raises(ValueError, match="PROVED"):
            self._report(warrant=Warrant.PROVED)

    def test_proved_with_a_certificate_constructs(self):
        cert = _certificate()
        report = self._report(warrant=Warrant.PROVED, evidence=(cert,))
        assert report.evidence == (cert,)

    def test_a_failing_proved_check_still_needs_evidence(self):
        # The outcome does not exempt it. A refutation carrying PROVED claims the
        # refutation was decided, which needs the same backing as a decided pass.
        with pytest.raises(ValueError, match="PROVED"):
            self._report(warrant=Warrant.PROVED, outcome=Outcome.FIRED)

    def test_evidence_must_be_a_tuple(self):
        # A claim over several enumerations rests on all their certificates. A bare one
        # would iterate as a sequence of fields somewhere downstream, or read as a
        # complete backing for a claim it only half covers.
        with pytest.raises(ValueError, match="tuple"):
            self._report(warrant=Warrant.PROVED, evidence=_certificate())

    def test_carries_a_certificate_per_enumeration(self):
        # A check quantified over two horizons carries both.
        pair = (_certificate(), _certificate())
        assert self._report(warrant=Warrant.PROVED, evidence=pair).evidence == pair

    def test_proved_with_a_symbolic_reduction_constructs(self):
        # Prover 2 decides its claim without enumerating anything, so a certificate is
        # the wrong evidence for it and its absence is not a missing backing.
        reduction = _reduction()
        report = self._report(warrant=Warrant.PROVED, evidence=(reduction,))
        assert report.evidence == (reduction,)

    def test_a_claim_resting_on_both_kinds_carries_both(self):
        # A symbolic identity asserted over an enumerated set of families rests on the
        # reduction and on the certificate, and one of them understates it.
        both = (_reduction(), _certificate())
        assert self._report(warrant=Warrant.PROVED, evidence=both).evidence == both

    @pytest.mark.parametrize(
        "item",
        ["research/gate_d4_registration.md", 4, None, Warrant.PROVED],
        ids=["a reference", "a count", "nothing", "the label again"],
    )
    def test_evidence_that_is_not_evidence_does_not_construct(self, item):
        # A precondition asking only whether the tuple is non-empty is satisfied by a
        # prose reference to where the proof lives, which is the plausible mistake and
        # backs the claim exactly as much as an empty tuple does.
        with pytest.raises(ValueError, match="as evidence"):
            self._report(warrant=Warrant.PROVED, evidence=(item,))

    def test_one_unbacked_item_beside_a_certificate_does_not_construct(self):
        # The tuple exists so a claim over several enumerations carries all of them,
        # so checking the first item alone would let the rest through unread.
        with pytest.raises(ValueError, match="as evidence"):
            self._report(
                warrant=Warrant.PROVED,
                evidence=(_certificate(), "and the H = 8 run"),
            )

    def test_a_weaker_level_may_not_carry_junk_either(self):
        # CERTIFIED needs no evidence, so a tuple on one is something the report claims
        # to be carrying rather than an unused field.
        with pytest.raises(ValueError, match="as evidence"):
            self._report(warrant=Warrant.CERTIFIED, evidence=("the bound",))

    @pytest.mark.parametrize("warrant", [Warrant.CERTIFIED, Warrant.CORROBORATED])
    def test_the_weaker_levels_need_none(self, warrant):
        # A bound and a sample carry their own story in `detail`. Only a claim to have
        # decided a universal needs something enumerable behind it.
        assert self._report(warrant=warrant).evidence == ()

    def test_proved_without_provenance_does_not_construct(self):
        # A decided universal says where the bar was fixed, so a reader can check the
        # ordering rather than take it on trust.
        with pytest.raises(ValueError, match="no provenance"):
            self._report(
                warrant=Warrant.PROVED, evidence=(_certificate(),), provenance=()
            )

    def test_a_failing_proved_check_still_needs_provenance(self):
        # A refutation carrying PROVED claims the refutation was decided, and the
        # ordering matters there for the same reason.
        with pytest.raises(ValueError, match="no provenance"):
            self._report(
                warrant=Warrant.PROVED,
                evidence=(_certificate(),),
                outcome=Outcome.FIRED,
                provenance=(),
            )

    def test_provenance_must_be_a_tuple(self):
        with pytest.raises(ValueError, match="tuple"):
            self._report(
                warrant=Warrant.PROVED,
                evidence=(_certificate(),),
                provenance=_provenance(),
            )

    @pytest.mark.parametrize(
        "item",
        ["9baaa22", "registered before the refit", 4, None],
        ids=["a bare ref", "a sentence", "a count", "nothing"],
    )
    def test_provenance_that_is_not_a_provenance_does_not_construct(self, item):
        # A bare ref satisfies a presence check as well as the real thing and carries
        # neither the other end of the ordering nor what was registered.
        with pytest.raises(ValueError, match="as provenance"):
            self._report(
                warrant=Warrant.PROVED,
                evidence=(_certificate(),),
                provenance=(item,),
            )

    def test_a_weaker_level_may_not_carry_junk_provenance_either(self):
        with pytest.raises(ValueError, match="as provenance"):
            self._report(warrant=Warrant.CORROBORATED, provenance=("9baaa22",))

    @pytest.mark.parametrize("warrant", [Warrant.CERTIFIED, Warrant.CORROBORATED])
    def test_the_weaker_levels_need_no_provenance(self, warrant):
        assert self._report(warrant=warrant).provenance == ()

    def test_a_weaker_level_may_record_one_anyway(self):
        # A registered bar is worth recording at CERTIFIED. It is not required there,
        # because a bound decides no universal either way.
        prov = _provenance()
        report = self._report(warrant=Warrant.CERTIFIED, provenance=(prov,))
        assert report.provenance == (prov,)

    def test_a_claim_resting_on_two_registrations_carries_both(self):
        # The same argument the evidence tuple already makes: carrying one of two
        # registrations understates what a reviewer has to check.
        pair = (_provenance(), _same_ref_provenance())
        report = self._report(
            warrant=Warrant.PROVED, evidence=(_certificate(),), provenance=pair
        )
        assert report.provenance == pair

    def test_the_report_renders_its_provenance_after_the_detail(self):
        line = str(self._report(warrant=Warrant.PROVED, evidence=(_certificate(),)))
        assert line.count("\n") == 0
        assert line.index("cue-ward") < line.index("provenance:")


class TestCheckSummary:
    """Counts per (warrant × outcome), so a green run says what it decided."""

    def _report(self, name, warrant, outcome, evidence=()):
        # These cases pass short prose names and read counts back, so the id is derived
        # here rather than declared. A suite may not do this: a derived key moves when
        # the prose does, which is what `check_id` exists to stop.
        return CheckReport(
            name=name,
            check_id="tests." + name.lower().replace(" ", "_"),
            warrant=warrant,
            outcome=outcome,
            tier=Tier.COMPUTED,
            detail="why",
            evidence=evidence,
            # The counts are what these cases read; PROVED needs a provenance to
            # construct at all.
            provenance=(_provenance(),) if warrant is Warrant.PROVED else (),
        )

    def test_counts_each_pair(self):
        # Read back as rows rather than as substrings: a bare `"2" in summary` is also
        # satisfied by the "2 tested here" in the header, so it cannot say the count
        # landed on the pair that earned it.
        summary = check_summary(
            [
                self._report("a", Warrant.CORROBORATED, Outcome.NOT_TRIGGERED),
                self._report("b", Warrant.CORROBORATED, Outcome.NOT_TRIGGERED),
                self._report("c", Warrant.CERTIFIED, Outcome.FIRED),
            ]
        )
        assert _rows(summary) == [
            ("CERTIFIED", "FIRED", 1),
            ("CORROBORATED", "NOT TRIGGERED", 2),
        ]

    def test_a_corroborative_surviving_run_reads_as_one(self):
        # The reason this function exists. Three falsifiers, none fired, none decisive.
        # One row is the assertion: nothing else is anywhere in the block.
        summary = check_summary(
            [
                self._report(str(i), Warrant.CORROBORATED, Outcome.NOT_TRIGGERED)
                for i in range(3)
            ]
        )
        assert summary.splitlines()[0] == "3 registered, 3 tested here, none fired"
        assert _rows(summary) == [("CORROBORATED", "NOT TRIGGERED", 3)]

    def test_pairs_with_no_checks_are_omitted(self):
        # The single row rules out all fourteen other pairs, where two `not in` checks
        # rule out the two they name.
        summary = check_summary(
            [self._report("a", Warrant.CERTIFIED, Outcome.NOT_TRIGGERED)]
        )
        assert _rows(summary) == [("CERTIFIED", "NOT TRIGGERED", 1)]

    def test_a_firing_falsifier_is_visible(self):
        summary = check_summary(
            [
                self._report("a", Warrant.CORROBORATED, Outcome.NOT_TRIGGERED),
                self._report("b", Warrant.CORROBORATED, Outcome.FIRED),
            ]
        )
        assert summary.splitlines()[0] == "2 registered, 2 tested here, 1 fired"
        assert _rows(summary) == [
            ("CORROBORATED", "NOT TRIGGERED", 1),
            ("CORROBORATED", "FIRED", 1),
        ]

    def test_all_five_outcomes_render_in_declaration_order(self):
        # The whole vocabulary in one block: each outcome once, the warrants in enum
        # order rather than input order, the two that never ran under the dash, and a
        # header separating five registered from the three that were tested here.
        summary = check_summary(
            [
                self._report("e", None, Outcome.NOT_RUN_HERE),
                self._report("c", Warrant.CORROBORATED, Outcome.FIRED),
                self._report(
                    "a", Warrant.PROVED, Outcome.NOT_TRIGGERED, (_certificate(),)
                ),
                self._report("d", None, Outcome.NOT_APPLICABLE),
                self._report("b", Warrant.CERTIFIED, Outcome.NOT_RESOLVED),
            ]
        )
        assert summary.splitlines()[0] == "5 registered, 3 tested here, 1 fired"
        assert _rows(summary) == [
            ("PROVED", "NOT TRIGGERED", 1),
            ("CERTIFIED", "NOT RESOLVED", 1),
            ("CORROBORATED", "FIRED", 1),
            ("—", "NOT APPLICABLE", 1),
            ("—", "NOT RUN HERE", 1),
        ]

    def test_the_header_separates_registered_from_tested(self):
        # The accounting ADR-029 required and a single count cannot carry: four
        # registered, two of which this run actually tested.
        summary = check_summary(
            [
                self._report(
                    "a", Warrant.PROVED, Outcome.NOT_TRIGGERED, (_certificate(),)
                ),
                self._report("b", Warrant.CORROBORATED, Outcome.NOT_RESOLVED),
                self._report("c", None, Outcome.NOT_APPLICABLE),
                self._report("d", None, Outcome.NOT_RUN_HERE),
            ]
        )
        assert summary.splitlines()[0] == "4 registered, 2 tested here, none fired"

    def test_checks_with_no_warrant_sort_under_a_dash(self):
        summary = check_summary(
            [
                self._report("a", None, Outcome.NOT_APPLICABLE),
                self._report("b", Warrant.CORROBORATED, Outcome.NOT_TRIGGERED),
            ]
        )
        assert _rows(summary) == [
            ("CORROBORATED", "NOT TRIGGERED", 1),
            ("—", "NOT APPLICABLE", 1),
        ]

    def test_a_void_falsifier_is_not_counted_as_tested(self):
        # ADR-029's gloss: evidence for nothing, and not a survivor.
        summary = check_summary([self._report("a", None, Outcome.NOT_APPLICABLE)])
        assert summary.splitlines()[0] == "1 registered, 0 tested here, none fired"

    def test_unresolved_is_not_folded_into_survival(self):
        # The ADR-029 rule that survives the vocabulary change: a check that decided
        # neither way is not counted among the ones that did, and keeps its own row
        # rather than being read off the header's tested count.
        summary = check_summary(
            [
                self._report("a", Warrant.CORROBORATED, Outcome.NOT_TRIGGERED),
                self._report("b", Warrant.CORROBORATED, Outcome.NOT_RESOLVED),
            ]
        )
        assert _rows(summary) == [
            ("CORROBORATED", "NOT TRIGGERED", 1),
            ("CORROBORATED", "NOT RESOLVED", 1),
        ]

    def test_counts_are_stable_under_input_order(self):
        reports = [
            self._report("a", Warrant.CERTIFIED, Outcome.NOT_TRIGGERED),
            self._report("b", Warrant.CORROBORATED, Outcome.FIRED),
        ]
        assert check_summary(reports) == check_summary(list(reversed(reports)))

    def test_no_checks_is_not_a_blank_line(self):
        # An empty suite is a finding, not a silence.
        assert check_summary([]).strip() != ""


class TestPublicSurface:
    def test_warrant_is_re_exported_from_the_package_root(self):
        import cpomdp

        assert cpomdp.Warrant is Warrant
        assert "Warrant" in cpomdp.__all__

    def test_the_rest_of_the_vocabulary_is_warrantlibs_alone(self):
        # ADR-069: one home for the vocabulary, so cpomdp does not promise it twice.
        import cpomdp

        for name in ("Outcome", "Tier", "CheckReport", "SymbolicReduction"):
            assert name not in cpomdp.__all__
            assert not hasattr(cpomdp, name)


class TestExistingLabelsUnchanged:
    def test_enumerated_search_is_still_proved(self):
        search = EnumeratedEfeSearch(_model(), _finite_set([-1.0, 1.0]), horizon=2)
        assert search.warrant is Warrant.PROVED

    def test_grid_selector_is_still_corroborated(self):
        selector = EFESelector(_model(), n_candidates=5, action_bounds=(-1.0, 1.0))
        assert selector.warrant is Warrant.CORROBORATED

    @pytest.mark.parametrize("driver", [RecedingHorizonSelector, OpenLoopSelector])
    def test_enumerated_drivers_are_still_proved(self, driver):
        search = EnumeratedEfeSearch(_model(), _finite_set([-1.0, 1.0]), horizon=2)
        assert driver(search).warrant is Warrant.PROVED

    def test_certificate_names_the_set_it_decided_over(self):
        # This string reaches the write-up, so it is pinned character for character.
        # It moved once, here: "finite set" became the set's own version, and the
        # count gained the base and exponent that produced it. Without them the
        # rendered evidence cannot tell two enumerations apart, since 81 is 9^2 and
        # 3^4 alike. The warrant word and both counts are where they were.
        cert = CompletenessCertificate(
            expected=4,
            visited=4,
            warrant=Warrant.PROVED,
            action_set_size=2,
            horizon=2,
            action_set_version="test-v1",
        )
        assert str(cert) == "PROVED (set test-v1, |A|^H = 2^2 = 4, visited 4)"

    def test_search_still_scores(self):
        # The label is a property, so a broken search would still report PROVED.
        search = EnumeratedEfeSearch(_model(), _finite_set([-1.0, 1.0]), horizon=2)
        result = search.evaluate(
            Belief([0.0, 0.0], [[1.0, 0.0], [0.0, 1.0]]),
            Preference(goal=[0.0], precision=[[1.0]]),
        )
        assert result.g.shape == (4,)
        assert search.warrant is Warrant.PROVED
