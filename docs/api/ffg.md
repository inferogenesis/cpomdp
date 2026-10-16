# Factor graph models

A `CouplingGraph` declares a rooted tree of Gaussian-coupled variables. It holds the shape a chain cannot: a node with three or more neighbours, inferred through the ones around it. `Coupling` edges carry the within-slice drive `child = W·parent + noise`, and each node's dynamics is a `GaussianTransition`. A node's observation is the sensor a flat model takes, a [`FixedSensor`][cpomdp.FixedSensor] or a state-dependent [`CallableSensor`][cpomdp.CallableSensor]. Moving a model onto a graph keeps its sensors. Run the graph through a [`CouplingGraphBackend`][cpomdp.CouplingGraphBackend].

::: cpomdp.CouplingGraph

::: cpomdp.Coupling

::: cpomdp.GaussianTransition
