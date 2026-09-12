# Warrant

Warrant is a property of the check, not of the number. A grid sample over a continuous action range and an exhaustive enumeration over a declared finite set can both report that no policy flips. Only the second decided it.

| Prover | What it does | Label |
| --- | --- | --- |
| 1 | pen-and-paper theorem, within stated hypotheses | `PROVED` |
| 2 | symbolic computation: closed-form identities, algebraic non-existence | `PROVED` |
| 3 · enumeration | exhaustive enumeration over a finite domain | `PROVED`, with a completeness certificate |
| 3 · validated | validated numerics over a compact domain | `CERTIFIED` |
| 3 · sample | sampling a continuum | `CORROBORATED` |

`research/warrant_ledger.md` carries the canonical version of this table, with the evidence each warrant requires and the tier a number is known to. An action sweep over a continuous range is a finite grid over an infinite domain, so it samples. A policy enumeration over a declared finite set enumerates. `EFESelector` reports `CORROBORATED` and `EnumeratedEfeSearch` reports `PROVED` for that reason.

`CERTIFIED` sits between the two. Validated numerics prove a universal over a compact domain, and the proof carries the bound it was computed with. Borrowing `PROVED` overclaims. Borrowing `CORROBORATED` throws the bound away.

::: warrantlib.Warrant

## What a check emits

A registered falsifier does not pass. It fires or it does not, and `PASS` is absent from the vocabulary rather than disambiguated by a column beside it.

`Outcome` has five values because five things can happen to a falsifier, and they are not interchangeable. It ran and did not fire, so the claim survives it. It fired, and the refutation is the result. It ran and the ordering came out genuinely undetermined, because the two quantities' intervals overlap. It was void by construction and could not have fired here, so it is evidence for nothing and is not a survivor. Or it was measured elsewhere and did not run here at all. Collapsing the last three loses the survivor accounting, and burns the word a real tie needs.

`Tier` says what the check was measured against, and cuts across the other two rather than ranking them. An `EXACT` closed-form reference can be sampled, and an exhaustive enumeration can produce a `COMPUTED` number.

A report carries two names. `name` is prose and reads in a summary line, so it is reworded whenever the wording improves. `check_id` is the key: dot-separated segments of letters, digits and underscores, refused at construction if anything else appears in it. The separation is what lets a manifest declare a check before the run and a later run be joined to an earlier one. Deriving the key from the prose would tie the two together, and the first reworded name would read as one check dropped and one added.

A check that never ran carries no warrant. `CORROBORATED` means sampling-grade evidence was obtained, so attributing it to a falsifier that sampled nothing claims evidence that does not exist. The warrant is `None` there and prints as `—`, enforced at construction.

A `PROVED` report needs evidence, enforced at construction. There are two families, one per decisive prover. `CompletenessEvidence` backs an exhaustive enumeration over a finite domain, and its leaves differ only in the shape of that domain. `SymbolicReduction` backs a theorem or a symbolic identity (Provers 1 and 2), which decide by argument and enumerate nothing, so a certificate is the wrong evidence for them rather than a missing one. The weaker levels need none, because a bound and a sample carry their story in `detail`. Report `PROVED` with nothing behind it and the constructor raises.

Those two families are the only things the evidence tuple accepts. A path naming where the proof lives is the plausible substitute, and it satisfies a presence check exactly as well as a certificate does. So the constructor checks every item's kind. Checking only the first would let a claim over several enumerations carry one certificate and three references to a write-up. The weaker levels are held to the same rule. They need no evidence, so a tuple on one of them is something the report says it is carrying.

::: warrantlib.Outcome

::: warrantlib.Tier

A completeness claim has two predicates and several possible domains. `CompletenessEvidence` holds the predicates: `expected` is the declared set's own cardinality, and `visited` reaches it. Each leaf under it says what its domain means and nothing else, so the `PROVED` gate is written once and a leaf cannot weaken it by forgetting to run it.

`CompletenessCertificate` is a tree, `|A|^H` over one versioned action set. `ProductCompletenessCertificate` is a cross, the product of declared axes. A bare count distinguishes neither: 81 is `9**2` and `3**4`, and 12 is `3 x 4` and `2 x 6`. Both carry their factors and their versions, so an addition after results are seen shows up in the diff.

::: warrantlib.CompletenessEvidence

::: warrantlib.CompletenessCertificate

::: warrantlib.AxisDeclaration

::: warrantlib.ProductCompletenessCertificate

::: warrantlib.CheckReport

## Evidence a symbolic claim carries

A CAS is a checker, not a witness. It establishes that one expression equals another, and it has nothing to say about whether those expressions are the ones the analytic claim is about. The warrant ledger records that step as a human obligation, which is the condition on Prover 2 being theorem-grade at all.

`SymbolicReduction` is where the obligation is discharged rather than assumed. `correspondence` names where the setup was analytically checked against the problem: a hand derivation by file and line, or a dated registration result. A field that cannot be filled honestly is the signal to report `CORROBORATED` and say why, so the type is not a formality. Blank fields do not construct.

Blank means blank to a reader rather than empty to `str.strip()`, which strips the whitespace and leaves the zero-width formatting characters behind. A field that is not text, one holding only whitespace, one holding only zero-width characters, and one carrying a line break into a one-line render are all refused. `assumptions` is checked entry by entry, and the message names which entry.

`assumptions` carries the scope. An identity contingent on smoothness, on positivity, or on an expansion being formal rather than convergent is a different claim from one that is not, and the difference belongs beside the evidence instead of in the algebra a reader would have to redo.

::: warrantlib.SymbolicReduction

## Registration, and the ordering it claims

Evidence says a claim was decided. It does not say when the bar was set, and a bar chosen after the number is visible decides nothing at all. `Provenance` is the pointer a reviewer follows to check: the ref where the prediction, the bar or the derivation was registered, the ref whose tree produced the number, and one line saying what they will find at the first of them.

A ref is a git commit SHA, an http(s) URL or a DOI. A path, a branch, a tag and `HEAD` are refused. Each of them satisfies a presence check exactly as well as a commit does, and each resolves to a different tree every time it is read. A URL is taken to be a permalink; one that tracks a branch has the same defect and the type cannot tell the two apart.

Where the two refs name one commit, the render says the ordering is not established by history. Registering and measuring together is not refused. It is what happens whenever a check and the derivation behind it land in one go, and the honest reading is that the ordering rests on the surrounding prose rather than on anything a reviewer can verify. An abbreviated ref counts as the same commit, or lengthening one of the two hashes would walk away from the marker while naming the same thing.

What the type cannot do is order two refs. Equality is checkable in a string and ordering is not, so a registration written after the fact renders exactly like one written before. That is a `git merge-base --is-ancestor` away, which is a reviewer's job or a test's, and is the reason the refs are refs.

::: warrantlib.Provenance

## The evidence union

`Evidence` names the completeness base and the symbolic reduction together. `CheckReport.evidence` is annotated as a tuple of it, so the admissible types are declared once, and a caller building reports of its own has a name to annotate against.

The union and the runtime guard are separate declarations. A kind outside the completeness family needs a member added to both. A type added to one alone annotates as evidence and then refuses to construct, or constructs and reads as evidence of a kind nothing declared. A new completeness leaf needs neither, since both already name the base, and needs a serialiser entry instead.

::: warrantlib.Evidence

## Reading a run

Registering four falsifiers and testing two is a different claim from testing four, and one number cannot carry both. The header separates them and names how many fired. The rows underneath say what warrant the tested ones carried, so a run that survived everything without deciding anything reads as exactly that.

```text
4 registered, 2 tested here, none fired
   PROVED        NOT TRIGGERED   2
   —             NOT APPLICABLE  1
   —             NOT RUN HERE    1
```

::: warrantlib.check_summary

## Writing a run down

A run that prints and exits leaves nothing to compare against. Two questions need the
record on disk: whether a registered check quietly stopped reporting, and whether one
changed status between two versions of the work. Both are joins on `check_id`, and
both fail if the form the record takes moves underneath them.

`report_to_dict` writes one report as a JSON-ready mapping. Every value is a string, an
integer, a list, a mapping or `None`, and the key order is fixed rather than taken from
the input, so two runs of one suite produce the same bytes and a diff shows what
changed rather than what moved. Enums travel as their values, which is why those values
are words.

`report_from_dict` reads one back through the constructor rather than around it. Every
precondition still applies: a record naming `PROVED` with its evidence stripped does not
construct. A wire form that could bypass the guard would make the guard optional.

Reading refuses what it cannot read. A record from a schema version this one does not
know, an evidence kind with no class behind it, an enum value that has since been
renamed: each raises rather than resolving to the nearest thing that fits. `Tier.A`
became `Tier.EXACT` once already, and a reader that had guessed its way through that
rename would have reported a status change nobody made.

Nothing here touches a filesystem. The caller decides where the bytes go.

::: warrantlib.SCHEMA_VERSION

::: warrantlib.report_to_dict

::: warrantlib.report_from_dict

A JSON Schema for the record ships beside the code, at `warrantlib/report.schema.json`.
It is there for a consumer reading a ledger without Python, so it states what it can of
the preconditions the constructor enforces: the key's shape, that a `PROVED` record
carries evidence and a registration, and that a check which never ran carries no
warrant. The test suite validates the writer's own output against it, since a schema
nothing checks drifts from the writer without saying so.

## Running checks under pytest

`pip install "warrantlib[pytest]"` adds a plugin, loaded through pytest's `pytest11`
entry point. It needs no configuration to do the first half of its job.

A test hands its findings over with the `record_check` fixture, and the run reports them
in the vocabulary the check used rather than as a column of dots.

```python
def test_the_coefficient_is_closed_form(record_check):
    record_check(measure_c2())
```

A fired check fails its test. So does an unresolved one, because a falsifier that ran and
could not decide has not left the claim standing. The two that never ran here skip, and
the progress letters keep them apart: `v` for void by construction, `e` for measured
elsewhere. Under `-v` the row reads `NOT TRIGGERED` where it would have read `PASSED`,
and the run closes with the registered / tested here / fired accounting.

The pytest outcome underneath is one of pytest's own three, and so is the counting
category. `junitxml` branches on the outcome and reads nothing else, so a fourth value
writes the row as no row at all. A fresh category would move every check out of
`N passed` into a name no existing tool reads. The vocabulary is worth carrying. A broken
tally is not the price to pay for it.

## Declaring what a suite is registered to report

The other half needs a manifest. A count of checks says a suite got shorter. It cannot
say which check left, and a check renamed or swapped for another leaves the count
untouched, so the gate passes on a suite that is now measuring something else.

`warrantlib.manifest` declares them instead. The file names each suite's entry point and
every id it reported when the file was written.

```toml
schema_version = "1.1"

[suites.series_kernel]
entry_point = "research.checks.series_kernel:run_checks"
checks = [
  "series_kernel.first_cumulant_is_the_mean",
]

[suites.example]
entry_point = "research.checks.example:run_checks"
checks = [
  "example.holds",
  "example.refuted_on_record",
]
refuted = [
  "example.refuted_on_record",
]
```

Point pytest at it with the `warrant_manifest` ini option and give it the path to
collect. Every declared check becomes an item, and the item exists because the manifest
declares it rather than because the suite reported it. A check that stops reporting still
has a row, and the row fails naming the check and the entry point it went missing from. A
second item per suite fails on any id the run reported that the manifest does not carry,
so a rename reports as one drop and one addition, which is what it is.

The suite runs once per session however many checks it declares, so a suite costing half
a minute costs that once rather than once per row.

`--warrant-detail` prints every check's own line: its outcome, its warrant, its tier,
the reason it gives and the refs it was registered at. `-vv` does the same, which is
pytest's own spelling for more detail than `-v` and is why the plugin claims no short
flag of its own. The row a run prints without either carries the verdict and not the
reason, which is the half a reader acts on.

A check that fires reads like any other failing test. It gets a `FAILURES` block naming
the check and carrying its reason, a row in the short summary, and the run's accounting
prints underneath with the fired count in it. No flag is needed for that.

A check whose registered result is a refutation is listed under `refuted`. Firing is
then what reconciles, rendered `REFUTED`, and anything else fails by name as
`NOT REFUTED`, since a refutation that stopped firing is a change on the same terms as
a check that stopped reporting. The check's own report is untouched: the accounting
still counts it as fired, and a line beneath names what was held. The list is written
by hand, after the result is on record elsewhere, and a rewrite keeps it for every id
the run still reports.

pytest's own total counts those reconciliation items and the warrant accounting does not,
because they are not checks and carry no warrant. Seventy declared checks across three
suites collect as seventy-three items, and the summary says which three so the difference
is not left as arithmetic.

Regenerate the file with `python -m warrantlib.manifest <path>` after a suite changes, and
ask whether it is current with `--check`, which returns non-zero on a stale one. It
compares as text, so a layout the writer no longer produces counts as stale too.

Reading a suite's ids means running it, so `--check` costs a full run. `--layout-only`
restricts either form to the layout question, answers it from the ids the file already
declares, and runs nothing. It is the form to put on a fast job where the ids are
reconciled elsewhere.
