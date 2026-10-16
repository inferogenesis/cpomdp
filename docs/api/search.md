# Exhaustive search

An exhaustive search scores every length-H sequence of a declared, finite action set and returns the best one. It also returns a [`CompletenessCertificate`][warrantlib.CompletenessCertificate] saying every sequence was scored. `EFESelector` samples a continuous range and reports `CORROBORATED`. A search decides over its declared set and reports `PROVED`. The names here are imported from `cpomdp.enumeration`.

::: cpomdp.enumeration.FiniteActionSet

## Two searches, one interface

The two searches take the same model, set and horizon, and answer the same `evaluate`. Swapping one for the other changes the memory a plan needs and no line that reads the result. `EnumeratedEFESearch` holds every policy and every score, and is the one to reach for first. `ChunkedEFESearch` scores in blocks at a footprint its block sets. Use it past the front-loaded search's budget, which is `2^21` policies unless set.

??? note "In plain terms"
    Give the search a menu of moves and a number of steps. It writes down every way of
    chaining that many moves from the menu, scores each chain by expected free energy
    from the current belief, and hands back the lowest. Nothing is sampled. Every chain
    on the menu is tried, so "this is the best plan on the menu" is a decided fact
    rather than a likely one, and the certificate records the count that makes it so.

    The number of chains is the menu's size raised to the number of steps, so it grows
    fast. The front-loaded search keeps every chain and every score in memory at once,
    which is quick and lets you look at all the scores afterwards. Past a budget it
    refuses at construction instead of running the machine out of memory, and its error
    says how big the set was and what to use instead. The chunked search works through
    the chains a block at a time and keeps only the best so far. It returns the same
    plan, bit for bit, and never holds the whole set.

    One convention to know. A plan's cost is reported per plan, as `cost_per_plan`.
    What an agent step costs depends on how often the agent plans, so only the
    selectors below report `cost_per_cycle`.

::: cpomdp.enumeration.ExhaustiveEFESearch

::: cpomdp.enumeration.SearchResult

::: cpomdp.enumeration.EnumeratedEFESearch
    options:
      inherited_members: true
      show_bases: false

::: cpomdp.enumeration.EnumeratedSearchResult

::: cpomdp.enumeration.ChunkedEFESearch
    options:
      inherited_members: true
      show_bases: false

::: cpomdp.enumeration.PolicyBudgetError

::: cpomdp.enumeration.IncompleteEnumerationError

## Driving an agent

A search returns a whole plan. The two selectors below turn one into an action per step for `Agent(selector=...)`, and each takes any `ExhaustiveEFESearch`. `RecedingHorizonSelector` plans every step and applies the first action. `OpenLoopSelector` plans once and applies the whole sequence. Their `cost_per_cycle` differs by a factor of H for that reason.

::: cpomdp.enumeration.RecedingHorizonSelector
    options:
      inherited_members: true
      show_bases: false

::: cpomdp.enumeration.OpenLoopSelector
    options:
      inherited_members: true
      show_bases: false
