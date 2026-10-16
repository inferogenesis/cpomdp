"""How well a claim is warranted, kept separate from whether it held.

Warrant is a property of the check, not of the number it produced. A grid sample and an
exhaustive enumeration can both come back clean. Only one of them decided anything, and
printing both as ``PASS`` loses that from the record.

Every check in a suite labels itself from this enum. It started as ``SearchWarrant``
inside cpomdp's enumerated search, covering searches alone.

``CheckReport`` is what a check emits. It pairs the warrant with an ``Outcome`` and a
``Tier``, which vary independently: a run can be green throughout and have decided
nothing, and the summary says so.

``CompletenessEvidence`` and ``SymbolicReduction`` are the two evidence families a
``PROVED`` report carries, one per decisive prover. The first has a leaf per domain
shape: ``CompletenessCertificate`` over a tree, ``ProductCompletenessCertificate`` over
a cross of ``AxisDeclaration``. ``Provenance`` is the other thing it
carries: which ref registered the claim, and which one measured it.

A report carries two names. ``name`` is prose and is reworded whenever the wording
improves. ``check_id`` is the key a manifest declares and two runs are joined on, so it
is not.

``report_to_dict`` and ``report_from_dict`` are the wire form, at ``SCHEMA_VERSION``.
Reading goes through the constructor rather than around it, and refuses a record it
cannot read rather than guessing what moved. A JSON Schema for the record ships beside
the code.

The standard library is the only dependency.
"""

from warrantlib._serialise import (
    SCHEMA_VERSION,
    report_from_dict,
    report_to_dict,
)
from warrantlib._vocabulary import (
    AxisDeclaration,
    CheckReport,
    CompletenessCertificate,
    CompletenessEvidence,
    Evidence,
    Outcome,
    ProductCompletenessCertificate,
    Provenance,
    SymbolicReduction,
    Tier,
    Warrant,
    check_summary,
)

__all__ = [
    "SCHEMA_VERSION",
    "AxisDeclaration",
    "CheckReport",
    "CompletenessCertificate",
    "CompletenessEvidence",
    "Evidence",
    "Outcome",
    "ProductCompletenessCertificate",
    "Provenance",
    "SymbolicReduction",
    "Tier",
    "Warrant",
    "check_summary",
    "report_from_dict",
    "report_to_dict",
]
