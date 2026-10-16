"""The E. coli chemotaxis network as a native FFG — built *with* cpomdp (issue #32).

A receptor-driven CheA hub feeds a fast CheY -> motor branch and a slow CheR/CheB
methylation branch. Each node is an Ornstein-Uhlenbeck process discretised on a single
clock via ``GaussianTransition.from_ou`` (ADR-017); the structural couplings are the
within-slice signal transduction ``child = W * parent + noise``. Topology matches the
RxInfer oracle ``cpomdp_chemotaxis_tree``.

This is a model built on the toolbox, not part of it: the Phase-3 acceptance tests and
the end-to-end demo (#27) import ``chemotaxis_ffg`` from here.
"""

from cpomdp.ffg.factors.linear_gaussian import GaussianTransition
from cpomdp.ffg.graph import Coupling, CouplingGraph
from cpomdp.observation import FixedSensor

CHEA = 0  # receptor kinase hub — the root
CHEY = 1  # fast response regulator, drives the motors
CHEB = 2  # slow methylation - the adaptation memory
MOTOR_A = 3
MOTOR_B = 4
_NODES = 5

# Per-node relaxation timescale tau (seconds). Fast kinase/motors ~0.05s; slow
# methylation ~9.9s - the 200x split the methylation carry cut exploits.
TAU = {CHEA: 0.05, CHEY: 0.05, CHEB: 9.9, MOTOR_A: 0.05, MOTOR_B: 0.05}

STATIONARY_VAR = dict.fromkeys(range(_NODES), 1.0)

_EDGES = [
    (CHEA, CHEY, 0.8, 0.05),
    (CHEA, CHEB, 0.6, 0.05),
    (CHEY, MOTOR_A, 1.0, 0.03),
    (CHEY, MOTOR_B, 1.0, 0.03),
]

_OBSERVED = (CHEB, MOTOR_A, MOTOR_B)
_OBS_NOISE = 0.1


def chemotaxis_ffg(dt):
    """Build the chemotaxis network for a timestep ``dt`` (seconds).

    Returns ``(graph, transitions)``: the structural ``CouplingGraph`` (couplings +
    sensors) and the node-indexed per-node OU ``GaussianTransition``s, to hand
    to ``CouplingGraphBackend(graph, transitions, prior=...)``.
    """
    transitions = tuple(
        GaussianTransition.from_ou(
            TAU[node], stationary_var=STATIONARY_VAR[node], dt=dt
        )
        for node in range(_NODES)
    )

    couplings = tuple(
        Coupling(
            parent,
            child,
            [[gain]],
            coupling_noise=[[noise]],
        )
        for parent, child, gain, noise in _EDGES
    )

    observation_models = {
        node: FixedSensor([[1.0]], observation_noise=[[_OBS_NOISE]])
        for node in _OBSERVED
    }
    graph = CouplingGraph(
        root=CHEA,
        dims=(1,) * _NODES,  # every node is scalar (1-D)
        couplings=couplings,
        observation_models=observation_models,
    )
    return graph, transitions
