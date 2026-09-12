# warrantlib

A column of `PASS` cannot say whether anything was decided. A grid sample over a
continuous range and an exhaustive enumeration over a declared finite set can both come
back clean. Only the second settled the question. warrantlib is a small vocabulary for
keeping that difference in a check suite's output instead of losing it there, and a
pytest plugin that reports it.

```bash
pip install "warrantlib[pytest]"   # the vocabulary and the plugin
pip install warrantlib             # the vocabulary alone
```

Python 3.11 and up. The standard library is the only dependency. The extra adds pytest
and nothing else, and `import warrantlib` pulls in no pytest either way.

## Running checks under pytest

The plugin loads through pytest's `pytest11` entry point. Nothing to configure and no
`-p` flag.

A test hands its findings over with the `record_check` fixture.

```python
def test_the_coefficient_is_closed_form(record_check):
    record_check(measure_c2())
```

```text
test_the_coefficient_is_closed_form NOT TRIGGERED                        [100%]

=============================== warrant summary ================================
1 registered, 1 tested here, none fired
   PROVED        NOT TRIGGERED   1
```

A check's outcome decides its test's, in pytest's own three:

| Outcome | The test | Letter | `-v` reads |
| --- | --- | --- | --- |
| `NOT_TRIGGERED` | passes | `.` | `NOT TRIGGERED` |
| `FIRED` | fails | `F` | `FIRED` |
| `NOT_RESOLVED` | fails | `?` | `NOT RESOLVED` |
| `NOT_APPLICABLE` | skips | `v` | `NOT APPLICABLE` |
| `NOT_RUN_HERE` | skips | `e` | `NOT RUN HERE` |

`NOT_RESOLVED` fails because a falsifier that ran and could not decide has not left the
claim standing. The two that never ran skip, and the letters keep them apart where the
skip does not.

The counting category stays pytest's own, so `N passed` counts what it always counted
and `assert_outcomes` still sees it. A fired check reads like any other failing test: a
`FAILURES` block carrying the check's reason, a row in the short summary, and the run's
accounting underneath with the fired count in it.

### Flags

| Flag | What it does |
| --- | --- |
| *(none)* | progress letters, then the registered / tested here / fired accounting |
| `-v` | one line per check, naming it and its outcome |
| `--warrant-detail` | each check's own line: outcome, warrant, tier, the reason it gives, the refs it was registered at |
| `-vv` | the same as `--warrant-detail`, which is pytest's own spelling for more detail |

```bash
pytest --warrant-detail
```

```text
================================ warrant checks ================================
T3 gain: NOT TRIGGERED (PROVED, tier exact). PASS — K = σ²/R̄ − σ⁴/R̄² + O(σ⁶).
got: K = sigma**2/Rbar - sigma**4/Rbar**2 provenance: Step 3, the expansion in
prior spread (registered at 99e3c34, measured at 23f0c47)
```

The row a run prints without it carries the verdict and not the reason, which is the
half a reader acts on.

## Declaring what a suite is registered to report

A count of checks says a suite got shorter. It cannot say which check left, and a check
renamed or swapped for another leaves the count untouched, so the gate passes on a suite
that is now measuring something else.

A manifest declares them instead. It names each suite's entry point and every id it
reported when the file was written.

```toml
# Generated. Do not edit by hand.
schema_version = "1.0"

[suites.series_kernel]
entry_point = "research.checks.series_kernel:run_checks"
checks = [
  "series_kernel.first_cumulant_is_the_mean",
]
```

Point pytest at it and give it the path to collect:

```toml
# pyproject.toml
[tool.pytest.ini_options]
warrant_manifest = "research/registered_checks.toml"
```

```bash
pytest research/registered_checks.toml
```

Every declared check becomes an item, and the item exists because the manifest declares
it rather than because the suite reported it. A check that stops reporting still has a
row, and the row fails naming the check and the entry point it went missing from. One
further item per suite fails on any id the run reported that the manifest does not
carry, so a rename reports as one drop and one addition, which is what it is.

The suite runs once per session however many checks it declares, so a suite costing half
a minute costs that once rather than once per row.

### Keeping it current

| Command | What it does |
| --- | --- |
| `python -m warrantlib.manifest <path>` | rewrite it from the suites it names |
| `python -m warrantlib.manifest --check <path>` | say whether it is current, change nothing, exit non-zero if not |
| `--layout-only`, with either | act on the layout alone, running no suite |

`--check` compares the file as text, so a layout the writer no longer produces counts as
stale too. It runs every suite to read their ids, which on a project whose suites are
expensive is a heavy way to ask about whitespace.

`--layout-only` restricts either form to that half, answering from the ids the file
already declares. It runs nothing, it says in its own output that it did not ask whether
the ids are current, and its write form prints `relaid` rather than `rewritten`. Where a
runner reconciles the declared ids as it runs them, this is the only question left for a
separate command, and it costs milliseconds.

Adding a suite means adding its
`[suites.<name>]` table by hand, with no checks, and then rewriting. The new ids land in
the diff, which is where they get reviewed.

A check whose registered result is a refutation is listed under `refuted`, by hand and
after the result is on record. The pytest plugin then passes it by firing, as
`REFUTED`, and fails it by name as `NOT REFUTED` otherwise. A rewrite keeps the list
for every id the run still reports.

## Warrant

`Warrant` says how well a claim is warranted, by the prover class behind it.

| Prover | What it does | Label |
| --- | --- | --- |
| 1 | pen-and-paper theorem, within stated hypotheses | `PROVED` |
| 2 | symbolic computation: closed-form identities, algebraic non-existence | `PROVED` |
| 3 · enumeration | exhaustive enumeration over a finite domain | `PROVED`, with a completeness certificate |
| 3 · validated | validated numerics over a compact domain | `CERTIFIED` |
| 3 · sample | sampling a continuum | `CORROBORATED` |

`CERTIFIED` sits between the other two. Validated numerics prove a universal over a
compact domain, and the proof carries the bound it was computed with. Borrowing `PROVED`
overclaims. Borrowing `CORROBORATED` throws the bound away.

An action sweep over a continuous range is a finite grid over an infinite domain, so it
samples. A policy enumeration over a declared finite set enumerates. The warrant follows
from which of those the check did, not from how clean the answer looked.

## Outcome

`Outcome` says what a registered falsifier did. A falsifier does not pass. It fires or it
does not, and `PASS` is absent from the vocabulary rather than disambiguated by a column
beside it.

| Value | What happened |
| --- | --- |
| `NOT_TRIGGERED` | it ran, the condition did not obtain, the claim survives it |
| `FIRED` | the condition obtained. The claim is refuted, and that is the result |
| `NOT_RESOLVED` | it ran and the ordering is genuinely undetermined, because the two quantities' intervals overlap |
| `NOT_APPLICABLE` | void by construction, so it is evidence for nothing and is not a survivor |
| `NOT_RUN_HERE` | measured elsewhere, or not yet. The detail says where |

Collapsing the last three loses the survivor accounting, and burns the word a real tie
needs. The last two never ran, so they carry no warrant, and `CheckReport` enforces that.

## The rest

`Tier` says what the check was measured against. `EXACT` against a closed form,
`BOUNDED` against a stated bar, `COMPUTED` where there is no bar to state. It cuts across
the other two rather than ranking them.

`CheckReport` is what a check emits. Frozen, because editing a report after the check ran
is editing the finding. It refuses `PROVED` with nothing behind it.

Evidence comes in two families, one per decisive prover. `CompletenessEvidence` backs an
exhaustive enumeration, recording the domain it covered against the count it visited. Its
leaves differ only in the shape of that domain: `CompletenessCertificate` over a tree,
`|A|^H` on one versioned action set, and `ProductCompletenessCertificate` over a cross,
the product of declared `AxisDeclaration` axes. A bare count separates neither, since 81
is `9**2` and `3**4` and 12 is `3 x 4` and `2 x 6`, so both carry their factors and their
versions.
`SymbolicReduction` backs a theorem or a symbolic identity, and names where the symbolic
setup was checked by hand against the analytic problem it stands for. A CAS establishes
that one expression equals another. Whether those are the right expressions is a human
obligation, and this is where it is discharged rather than assumed.

`check_summary` prints a run as counts per `(warrant, outcome)`.

```python
from warrantlib import (
    CheckReport, Outcome, Provenance, SymbolicReduction, Tier, Warrant, check_summary,
)

report = CheckReport(
    name="second gap coefficient",
    check_id="gap_series.c2_closed_form",
    warrant=Warrant.PROVED,
    outcome=Outcome.NOT_TRIGGERED,
    tier=Tier.EXACT,
    detail="the CAS reduces the integral to the quoted constant",
    evidence=(
        SymbolicReduction(
            claim="the second gap coefficient equals the quoted constant",
            correspondence="hand derivation, section 3",
            assumptions=("the expansion is formal, not convergent",),
        ),
    ),
    provenance=(
        Provenance(
            registered_at="a76cf1b",
            measured_at="9baaa22",
            registered="the coefficient's closed form, registered 2026-08-07",
        ),
    ),
)

print(check_summary([report]))
```

```text
1 registered, 1 tested here, none fired
   PROVED        NOT TRIGGERED   1
```

Registering four falsifiers and testing two is a different claim from testing four, and
one number cannot carry both. The header separates them.

## Two names per check

A report carries prose and a key, and they do different jobs.

`name` reads in a summary line, so it is reworded whenever the wording improves.
`check_id` is what a manifest declares before a run and what joins one run's report to
the next: dot-separated segments of letters, digits and underscores, refused at
construction if anything else appears in it.

Deriving the key from the prose ties the two together, and the first reworded name then
reads as one check dropped and one added, which is the reading a comparison exists to
rule out.

## Provenance

Evidence says a claim was decided. It does not say when the bar was set, and a bar chosen
after the number is visible decides nothing. `Provenance` is the pointer a reviewer
follows: the ref where the prediction or the derivation was registered, the ref whose
tree produced the number, and one line saying what they will find at the first.

A ref is a git commit SHA, an http(s) URL or a DOI. A path, a branch, a tag and `HEAD`
are refused, because each resolves to a different tree every time it is read. A `PROVED`
report requires one, on the same terms as it requires evidence.

Where the two refs name one commit, the render says the ordering is not established by
history. That is not a failure. It is what happens whenever a check and the derivation
behind it land together, and the marker keeps it from reading as something a reviewer
could verify.

The type compares refs. It cannot order them, so a registration written after the fact
still renders without a marker. That is a `git merge-base --is-ancestor` away, which is
why the refs are refs.

## Writing a run down

A run that prints and exits leaves nothing to compare against. `report_to_dict` writes
one report as a JSON-ready mapping, and `report_from_dict` reads one back.

```python
from warrantlib import SCHEMA_VERSION, report_from_dict, report_to_dict

record = report_to_dict(report)
assert report_from_dict(record) == report
```

Key order is fixed rather than taken from the input, so two runs of one suite produce
the same bytes and a diff shows what changed rather than what moved. Enums travel as
their values, which is why those values are words.

Reading goes through the constructor rather than around it, so every precondition still
applies. A record naming `PROVED` with its evidence stripped does not construct. A
record from a schema version this one does not know, an evidence kind with no class
behind it, or an enum value no member carries is refused rather than resolved to
whatever fits.

Nothing here touches a filesystem. The caller decides where the bytes go. A JSON Schema
for the record ships beside the code at `warrantlib/report.schema.json`, for a consumer
reading a ledger without Python.

## Where it comes from

warrantlib was factored out of [cpomdp](https://github.com/inferogenesis/cpomdp), where
it labels a research programme's falsification battery. It is developed in that
repository and released separately. The API reference is at
[cpomdp.inferogenesis.com/api/warrant](https://cpomdp.inferogenesis.com/api/warrant/).
