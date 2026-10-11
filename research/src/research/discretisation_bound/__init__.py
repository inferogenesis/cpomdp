"""Verification code for `research/discretisation_bound_hand_derivation.md`.

The derivation bounds how far the reference engine's plug-in gap on the declared lattice
can sit from the exact one. It was worked on paper first. The scan is
`research/discretisation_bound_hand_derivation_part1.pdf`, and the markdown file types
it up. This package checks what the derivation claims: its identities in SymPy, its
inequalities as identities with a manifestly non-negative remainder where it can, and
its constants as exact numbers. Where a bound is a claim about every point of a strip,
the bound is corroborated numerically on a grid and is not proved there.

Nothing in it is the bound. No module evaluates `δ_ref` or reports a warrant. The suite
that does is written against this derivation's commit and emits `CERTIFIED`.

Running a module is checking it: every claim it prints is asserted inside it.
"""
