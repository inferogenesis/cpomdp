"""The exact reference filter's substrate: a quadrature grid and densities on it.

Nothing here knows about a state-space model, a backend or an action. A density is
values on a declared lattice, and integration is a weighted sum over those values.
The filter that builds such densities from a likelihood and a transition kernel sits
above this, and is what makes the two independent objects a comparison needs.

Reached via ``cpomdp._lab.reference.quadrature`` and not re-exported at the package top
level, since nothing outside is meant to depend on it yet.
"""
