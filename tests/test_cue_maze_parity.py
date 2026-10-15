"""`cue_maze.build_maze(1)` is the model the crossover was measured on, not a lookalike.

`cue_maze.py` is the dimension-agnostic rewrite of the corridor scene. Its comment
claims that at one spatial dimension it reproduces
`epistemic_dissociation_figure.build_backend(cue_x=CUE_DETOUR_X)` element for element.
The claim was true when it was written and nothing held it there: two model definitions
in two files, kept in sync by whoever remembered both.

That matters because the registered crossover number is measured on the dissociation
model. If the rewrite drifts, every result quoted off `cue_maze` silently stops being
about the model the number belongs to, and the coincidence of shapes hides it.

The frozen twins are excluded on purpose. They disagree, `cue_maze.py` says so and says
why, and pinning them equal here would assert the opposite.
"""

import cue_maze
import epistemic_dissociation_figure as dissociation
import jax.numpy as jnp
import numpy as np
import pytest

# The one-dimensional arena the two builds are supposed to agree on.
N_DIMS = 1
# Positions spanning the corridor, over the cue and both goals. R(x) is smooth, so a
# drift in the well's centre, width or floor cannot hide between them.
PROBE_POSITIONS = np.linspace(-4.0, 4.0, 17)


@pytest.fixture(scope="module")
def pair():
    """The two live builds, `(cue_maze, dissociation)`."""
    return (
        cue_maze.build_maze(N_DIMS),
        dissociation.build_backend(
            epistemic_alive=True, cue_x=dissociation.CUE_DETOUR_X
        ),
    )


def _joint_mean(position: float) -> jnp.ndarray:
    """The joint state `[context, position, goal belief]` with only position moving."""
    return jnp.array([dissociation.PRIOR_ARM, float(position), dissociation.PRIOR_ARM])


def test_node_shapes_and_wiring_agree(pair):
    """dims, root, and which node carries the observation."""
    maze, registered = pair
    assert maze.dims == registered.dims
    assert maze.graph.root == registered.graph.root
    assert sorted(maze.graph.observations) == sorted(registered.graph.observations)


def test_dynamics_and_control_agree(pair):
    """A and Q per node, and the control B that drives position."""
    maze, registered = pair
    pairs = zip(maze.transitions, registered.transitions, strict=True)
    for node, (mine, theirs) in enumerate(pairs):
        np.testing.assert_allclose(
            mine.dynamics_matrix,
            theirs.dynamics_matrix,
            err_msg=f"A differs at node {node}",
        )
        np.testing.assert_allclose(
            mine.dynamics_noise,
            theirs.dynamics_noise,
            err_msg=f"Q differs at node {node}",
        )
    # `_control` rather than `to_flat_model().control`: flattening raises on an R(x)
    # model with couplings, which is exactly what both of these are.
    np.testing.assert_allclose(maze._control, registered._control)


def test_the_coupling_and_its_noise_agree(pair):
    """W and the coupling noise on the single context -> arena edge."""
    maze, registered = pair
    (mine,), (theirs,) = maze.graph.couplings, registered.graph.couplings
    assert (mine.parent, mine.child) == (theirs.parent, theirs.child)
    assert mine.efe_relevant == theirs.efe_relevant
    np.testing.assert_allclose(
        mine.factor.coupling_matrix, theirs.factor.coupling_matrix
    )
    np.testing.assert_allclose(mine.factor.coupling_noise, theirs.factor.coupling_noise)


def test_the_whole_rx_surface_agrees(pair):
    """C, and R(x) sampled the length of the corridor rather than at one point.

    A well that had moved, widened or changed floor would still match at a single
    sample. The sweep is what makes this a check on the surface. Moving the cue to
    2.5 takes the info channel from 0.02 to 108 at the cue's own position, so the
    sweep has room to see a drift far smaller than that.
    """
    maze, registered = pair
    np.testing.assert_allclose(
        maze.observation_model[0], registered.observation_model[0]
    )
    for position in PROBE_POSITIONS:
        np.testing.assert_allclose(
            maze.observation_noise_at(_joint_mean(position)),
            registered.observation_noise_at(_joint_mean(position)),
            err_msg=f"R(x) differs at position {position}",
        )
