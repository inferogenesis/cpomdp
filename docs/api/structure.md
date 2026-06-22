# Model structure

An optional, static declaration of a model's factorisation — `factors`, Markov-blanket `roles`, and observation `channels` — that you can inspect and `validate()` against the matrices. Declared but not yet exploited by the engine (ADR-010): the data and inspection surface is stable; `validate()` is experimental.

::: cpomdp.ModelStructure
