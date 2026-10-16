import contextlib
import inspect
from collections.abc import Mapping

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from cpomdp._lab.degraded import WrongFixedRBackend
from cpomdp._lab.harness import ScoredAgent, World
from cpomdp.backends.kalman import KalmanBackend
from cpomdp.dynamics import CallableProcessNoise
from cpomdp.observation import CallableSensor, FixedSensor
from cpomdp.types import Belief, LinearGaussianModel

# The double integrator from test_agent: state = [position, velocity], a force moves
# the velocity, velocity moves the position, position is observed. Two *separately
# constructed* instances stand in for p* and p — the harness never derives one from
# the other, so the tests never do either.
DT = 0.1
DYNAMICS = np.array([[1.0, DT], [0.0, 1.0]])
CONTROL = np.array([[0.0], [DT]])
SENSOR = np.array([[1.0, 0.0]])


def _model(*, damping: float = 1.0) -> LinearGaussianModel:
    """A double integrator whose velocity decays at ``damping`` per step.

    ``damping=1.0`` is the undamped truth. Anything else is a misspecified model, and
    the misspecification lives in one entry so a cell can name what it perturbed.
    """
    dynamics = np.array([[1.0, DT], [0.0, damping]])
    return LinearGaussianModel(
        dynamics_matrix=dynamics,
        observation_matrix=SENSOR,
        dynamics_noise=[[1e-4, 0.0], [0.0, 1e-4]],
        observation_noise=[[1e-2]],
        prior=Belief(mean=[0.0, 0.0], cov=[[1.0, 0.0], [0.0, 1.0]]),
        control_matrix=CONTROL,
    )


def _reachable(root: object, *, limit: int = 20_000) -> set[int]:
    """Every object id reachable from ``root`` by attribute and container traversal.

    Follows closure cells, argument defaults, a bound method's ``__self__`` and the
    attributes of an object's class, as well as its own, so a captured reference counts
    as reaching. Arrays are terminal: they hold numbers, not references onward, and
    iterating a traced array is a mistake in its own right. The limit is a runaway
    guard. A cycle is already handled by ``seen``.
    """
    seen: set[int] = set()
    queue = [root]
    while queue and len(seen) < limit:
        obj = queue.pop()
        if id(obj) in seen:
            continue
        seen.add(id(obj))
        if isinstance(obj, jax.Array | np.ndarray | str | bytes | int | float | bool):
            continue
        if isinstance(obj, dict):
            queue.extend(obj.keys())
            queue.extend(obj.values())
            continue
        if isinstance(obj, list | tuple | set | frozenset):
            queue.extend(obj)
            continue
        # getattr rather than vars(): a descriptor found on a class has a __dict__ that
        # is not a mapping, and vars() raises on it.
        namespace = getattr(obj, "__dict__", None)
        if isinstance(namespace, Mapping):
            queue.extend(namespace.values())
        slots = getattr(obj, "__slots__", ())
        if isinstance(slots, str):
            slots = (slots,)
        if isinstance(slots, list | tuple):
            queue.extend(
                getattr(obj, name)
                for name in slots
                if isinstance(name, str) and hasattr(obj, name)
            )
        if (bound := getattr(obj, "__self__", None)) is not None:
            queue.append(bound)
        # Every dunder below is type-checked before use. Reached through a class rather
        # than an instance each resolves to the descriptor implementing it, and a
        # descriptor is not the tuple or dict the name suggests.
        if isinstance(defaults := getattr(obj, "__defaults__", None), tuple):
            queue.extend(defaults)
        if isinstance(kwdefaults := getattr(obj, "__kwdefaults__", None), dict):
            queue.extend(kwdefaults.values())
        if not isinstance(obj, type):
            # Class attributes are not in vars(obj), and a reference parked on the class
            # is held just as firmly as one parked on the instance.
            queue.extend(vars(type(obj)).values())
        if isinstance(closure := getattr(obj, "__closure__", None), tuple):
            for cell in closure:
                # An empty cell comes from a recursive definition and holds nothing yet.
                with contextlib.suppress(ValueError, AttributeError):
                    queue.append(cell.cell_contents)
    return seen


def _defaulted(world):
    """A function holding the world in a positional default, not in a closure."""

    def held(w=world):
        return w

    return held


def _kwdefaulted(world):
    """The same, in a keyword-only default."""

    def held(*, w=world):
        return w

    return held


# --- the detector, before the seam it is pointed at ---------------------------------
#
# The seam tests below assert a negative, so they also pass when the walk finds
# nothing at all. These plant each way a reference can be held and require it found.


@pytest.mark.parametrize(
    "leak",
    [
        pytest.param(lambda world: world, id="attribute"),
        pytest.param(lambda world: [world], id="inside a list"),
        pytest.param(lambda world: {"w": world}, id="inside a dict"),
        pytest.param(lambda world: world.step, id="bound method"),
        pytest.param(lambda world: lambda: world, id="closure cell"),
        pytest.param(_defaulted, id="argument default"),
        pytest.param(_kwdefaulted, id="keyword-only default"),
    ],
)
def test_the_walk_finds_a_planted_reference(leak):
    world = World(_model())
    agent = ScoredAgent(_model())
    agent._leak = leak(world)  # ty: ignore[unresolved-attribute]
    assert id(world) in _reachable(agent)


def test_the_walk_finds_a_reference_parked_on_the_class():
    world = World(_model())
    agent = ScoredAgent(_model())

    class _Leaky(ScoredAgent):
        held = world

    assert id(world) in _reachable(_Leaky(_model()))
    assert id(world) not in _reachable(agent)


def test_the_walk_finds_a_planted_parameter():
    truth = _model()
    agent = ScoredAgent(_model())
    agent._leak = truth.dynamics_matrix  # ty: ignore[unresolved-attribute]
    assert id(truth.dynamics_matrix) in _reachable(agent)


def test_the_walk_terminates_on_a_cycle():
    left: dict = {}
    right = {"back": left}
    left["on"] = right
    assert id(right) in _reachable(left)


# --- the seam: no path from the agent to the world's parameters -------------------


def test_world_exposes_no_generative_model():
    world = World(_model())
    exposed = [getattr(world, name) for name in dir(world) if not name.startswith("_")]
    assert not [x for x in exposed if isinstance(x, LinearGaussianModel)]


def test_world_exposes_no_generative_parameter():
    truth = _model()
    world = World(truth)
    # One step first. An unstepped world's `state` *is* the prior's mean, so leaving the
    # prior out of the set below would be the only way to pass, and a `prior_cov`
    # accessor would then go unnoticed.
    world.step([1.0], jax.random.PRNGKey(0))
    parameters = {
        id(truth.dynamics_matrix),
        id(truth.observation_matrix),
        id(truth.dynamics_noise),
        id(truth.observation_noise),
        id(truth.control_matrix),
        id(truth.prior),
        id(truth.prior.mean),
        id(truth.prior.cov),
    }
    exposed = {
        id(getattr(world, name)) for name in dir(world) if not name.startswith("_")
    }
    assert not parameters & exposed


@pytest.mark.parametrize(
    "constructor", [ScoredAgent.__init__, ScoredAgent.from_backend]
)
def test_scored_agent_accepts_no_world(constructor):
    parameters = inspect.signature(constructor).parameters
    annotations = {str(p.annotation) for p in parameters.values()}
    assert "world" not in parameters
    assert not [a for a in annotations if "World" in a]


def test_the_world_is_not_reachable_from_the_agent():
    world = World(_model())
    agent = ScoredAgent(_model(damping=0.8))
    key = jax.random.PRNGKey(0)
    for _ in range(3):
        key, subkey = jax.random.split(key)
        agent.driven_step(world.step([1.0], subkey), [1.0])

    assert id(world) not in _reachable(agent)


def test_the_worlds_parameters_are_not_reachable_from_the_agent():
    truth = _model()
    world = World(truth)
    agent = ScoredAgent(_model(damping=0.8))
    agent.driven_step(world.step([1.0], jax.random.PRNGKey(0)), [1.0])

    reached = _reachable(agent)
    assert id(truth) not in reached
    assert id(truth.dynamics_matrix) not in reached


def test_the_agent_keeps_its_own_model():
    agent = ScoredAgent(_model(damping=0.8))
    assert float(agent.model.dynamics_matrix[1, 1]) == pytest.approx(0.8)


# --- exogenous action: the agent never chooses -------------------------------------


def test_scored_agent_cannot_select_actions():
    agent = ScoredAgent(_model())
    assert not hasattr(agent, "sample_action")
    assert not hasattr(agent, "infer_policies")


def test_a_driven_step_requires_an_action():
    agent = ScoredAgent(_model())
    with pytest.raises(TypeError):
        agent.driven_step([0.0])  # ty: ignore[missing-argument]


def test_the_driven_action_reaches_the_predict_step():
    left, right = ScoredAgent(_model()), ScoredAgent(_model())
    observation = [0.5]
    pushed = left.driven_step(observation, [+1.0])
    pulled = right.driven_step(observation, [-1.0])
    assert not np.allclose(np.asarray(pushed.mean), np.asarray(pulled.mean))


def test_the_belief_advances_and_is_replaced_not_mutated():
    agent = ScoredAgent(_model())
    before = agent.belief
    after = agent.driven_step([0.5], [0.0])
    assert after is agent.belief
    assert before is not after
    assert float(before.cov[0, 0]) > float(after.cov[0, 0])


def test_the_agent_starts_at_its_own_prior():
    agent = ScoredAgent(_model())
    assert np.allclose(np.asarray(agent.belief.mean), [0.0, 0.0])


def test_a_backend_is_accepted_and_carries_its_own_model():
    model = _model(damping=0.8)
    agent = ScoredAgent.from_backend(KalmanBackend(model))
    assert agent.model is model


def test_a_degraded_backend_is_scored_under_the_model_it_degrades():
    model = _model(damping=0.8)
    agent = ScoredAgent.from_backend(WrongFixedRBackend(model, magnitude=0.5))
    assert agent.model is model


def test_a_model_and_a_backend_cannot_both_be_passed():
    # Each front door takes one input, so there is nothing to disagree.
    model = _model()
    with pytest.raises(TypeError):
        ScoredAgent(model, backend=KalmanBackend(model))  # ty: ignore[unknown-argument]


# --- the world simulates p*, and only p* --------------------------------------------


def test_world_step_returns_one_observation():
    world = World(_model())
    observation = world.step([0.0], jax.random.PRNGKey(0))
    assert observation.shape == (1,)


def test_world_step_advances_the_true_state():
    world = World(_model(), initial_state=[0.0, 1.0])
    key = jax.random.PRNGKey(0)
    first = world.step([0.0], key)
    second = world.step([0.0], key)  # same key: the difference is the state, not noise
    assert float(second[0]) > float(first[0])


def test_the_same_key_gives_the_same_observation():
    left = World(_model(), initial_state=[0.0, 1.0])
    right = World(_model(), initial_state=[0.0, 1.0])
    key = jax.random.PRNGKey(7)
    assert float(left.step([0.5], key)[0]) == float(right.step([0.5], key)[0])


def test_the_initial_state_defaults_to_the_prior_mean():
    world = World(_model())
    assert np.allclose(np.asarray(world.state), [0.0, 0.0])


def test_the_true_state_is_readable_because_the_world_is_not_the_agent():
    world = World(_model(), initial_state=[1.0, 0.0])
    assert np.allclose(np.asarray(world.state), [1.0, 0.0])


def test_an_initial_state_of_the_wrong_shape_does_not_construct():
    with pytest.raises(ValueError, match="1-D vector of length 2"):
        World(_model(), initial_state=[1.0, 0.0, 0.0])


# --- state-dependent noise known to the world -----------------------------
#
# The filter reads R(x) and Q(x) at its predicted mean, an estimate of the state the
# step arrives at. A world knows that state, so the two do not read the same point, and
# each test below pins which point the world used rather than only that it varied.


def _sensor_noise_at(mean, params):
    return jnp.array([[1e-2]]) * (1.0 + 4.0 * jnp.sum(jnp.asarray(mean) ** 2))


def _process_noise_at(x, params):
    return jnp.eye(2) * 1e-4 * (1.0 + 4.0 * jnp.sum(jnp.asarray(x) ** 2))


def _sensed_model(*, sensor=True, process=False) -> LinearGaussianModel:
    return LinearGaussianModel(
        dynamics_matrix=DYNAMICS,
        observation_matrix=None if sensor else SENSOR,
        dynamics_noise=None if process else [[1e-4, 0.0], [0.0, 1e-4]],
        observation_noise=None if sensor else [[1e-2]],
        prior=Belief(mean=[1.0, 1.0], cov=np.eye(2)),
        control_matrix=CONTROL,
        observation_model=(
            CallableSensor(SENSOR, _sensor_noise_at, ()) if sensor else None
        ),
        dynamics_noise_model=(
            CallableProcessNoise(_process_noise_at, ()) if process else None
        ),
    )


def test_a_state_dependent_sensor_constructs():
    assert World(_sensed_model()).n_observations == 1


def test_state_dependent_process_noise_constructs():
    assert World(_sensed_model(sensor=False, process=True)).n_states == 2


def test_the_sensor_noise_is_read_at_the_state_the_step_arrived_at():
    model = _sensed_model()
    world = World(model, initial_state=[1.0, 1.0])
    key = jax.random.PRNGKey(5)
    observation = world.step([1.0], key)

    _, key_observation = jax.random.split(key)
    arrived = world.state  # drawn before the reading, and what the sensor measured
    expected = jax.random.multivariate_normal(
        key_observation,
        jnp.asarray(SENSOR) @ arrived,
        _sensor_noise_at(arrived, ()),
    )
    assert float(observation[0]) == float(expected[0])


def test_the_sensor_noise_is_not_read_at_the_state_the_step_left():
    model = _sensed_model()
    world = World(model, initial_state=[1.0, 1.0])
    key = jax.random.PRNGKey(5)
    observation = world.step([1.0], key)

    _, key_observation = jax.random.split(key)
    departed = jnp.asarray([1.0, 1.0])
    wrong = jax.random.multivariate_normal(
        key_observation,
        jnp.asarray(SENSOR) @ world.state,
        _sensor_noise_at(departed, ()),
    )
    assert float(observation[0]) != float(wrong[0])


def test_the_process_noise_is_read_at_the_pushed_forward_mean():
    model = _sensed_model(sensor=False, process=True)
    start = jnp.asarray([1.0, 1.0])
    world = World(model, initial_state=start)
    key = jax.random.PRNGKey(5)
    world.step([1.0], key)

    key_dynamics, _ = jax.random.split(key)
    pushed = jnp.asarray(DYNAMICS) @ start + jnp.asarray(CONTROL) @ jnp.asarray([1.0])
    expected = jax.random.multivariate_normal(
        key_dynamics, pushed, _process_noise_at(pushed, ()), method="svd"
    )
    assert np.array_equal(np.asarray(world.state), np.asarray(expected))


def test_the_process_noise_is_not_read_at_the_state_the_step_left():
    model = _sensed_model(sensor=False, process=True)
    start = jnp.asarray([1.0, 1.0])
    world = World(model, initial_state=start)
    key = jax.random.PRNGKey(5)
    world.step([1.0], key)

    key_dynamics, _ = jax.random.split(key)
    pushed = jnp.asarray(DYNAMICS) @ start + jnp.asarray(CONTROL) @ jnp.asarray([1.0])
    wrong = jax.random.multivariate_normal(
        key_dynamics, pushed, _process_noise_at(start, ()), method="svd"
    )
    assert not np.array_equal(np.asarray(world.state), np.asarray(wrong))


def test_a_fixed_sensor_carried_as_an_observation_model_reads_its_constant():
    fixed = LinearGaussianModel(
        dynamics_matrix=DYNAMICS,
        dynamics_noise=[[1e-4, 0.0], [0.0, 1e-4]],
        prior=Belief(mean=[1.0, 1.0], cov=np.eye(2)),
        control_matrix=CONTROL,
        observation_model=FixedSensor(SENSOR, observation_noise=[[1e-2]]),
    )
    plain = World(_model(), initial_state=[1.0, 1.0])
    carried = World(fixed, initial_state=[1.0, 1.0])
    key = jax.random.PRNGKey(5)
    assert float(plain.step([1.0], key)[0]) == float(carried.step([1.0], key)[0])


def test_the_fixed_path_is_unmoved_by_the_state_dependent_one():
    # The regression guard. The number was measured on the commit before state-dependent
    # noise was admitted, in a worktree of that commit, not read off this branch.
    world = World(_model(), initial_state=[1.0, 1.0])
    observation = world.step([1.0], jax.random.PRNGKey(5))
    assert float(observation[0]) == 1.0105588758439044


# --- a sensor the world cannot draw from is refused, not turned into NaN ------------


def _disagreeing_sensor_model() -> LinearGaussianModel:
    """A model whose sensor reads a different observation dimension from its matrix."""
    wide = np.array([[1.0, 0.0], [0.0, 1.0]])

    def noise_at(mean, params):
        return jnp.eye(2) * 1e-2

    return LinearGaussianModel(
        dynamics_matrix=DYNAMICS,
        observation_matrix=SENSOR,  # (1, 2): the model says one observation
        dynamics_noise=[[1e-4, 0.0], [0.0, 1e-4]],
        observation_noise=[[1e-2]],
        prior=Belief(mean=[0.0, 0.0], cov=np.eye(2)),
        control_matrix=CONTROL,
        observation_model=CallableSensor(wide, noise_at, ()),  # the sensor says two
    )


def test_a_sensor_disagreeing_with_the_model_does_not_construct():
    # The refusal is the model's, not the world's, and it fires while the argument
    # to World is still being built. Asserting it around the World call read as a
    # check on World and would have passed with no World in the line at all.
    with pytest.raises(ValueError, match="observation_matrix"):
        _disagreeing_sensor_model()


def test_a_sensor_noise_that_is_not_positive_definite_is_refused_not_drawn():
    # A silent NaN observation would reach every agent's belief and every score. The
    # dynamics draw uses svd because a noiseless direction is legal there; a sensor
    # noise that loses definiteness is not legal anywhere.
    def vanishing(mean, params):
        return jnp.array([[1e-2]]) * (1.0 - jnp.asarray(mean)[0])

    model = LinearGaussianModel(
        dynamics_matrix=DYNAMICS,
        dynamics_noise=[[1e-8, 0.0], [0.0, 1e-8]],
        prior=Belief(mean=[0.0, 0.0], cov=np.eye(2)),
        control_matrix=CONTROL,
        observation_model=CallableSensor(SENSOR, vanishing, ()),
    )
    world = World(model, initial_state=[0.0, 10.0])

    def walk_until_the_noise_vanishes():
        for _ in range(4):
            world.step([0.0], jax.random.PRNGKey(0))

    with pytest.raises(ValueError, match="positive-definite because") as refusal:
        walk_until_the_noise_vanishes()
    assert "Cholesky" in str(refusal.value)
    assert "inverted" not in str(refusal.value)


def test_a_process_noise_that_goes_indefinite_is_refused_not_drawn():
    # The sensor's failure is a NaN. This one is not: the svd draw factors through the
    # singular values, so a negative eigenvalue is sampled as |λ| and the step returns
    # a finite, plausible state drawn from the wrong Q. Nothing downstream can tell.
    def loses_a_direction(mean, params):
        return jnp.diag(jnp.array([1e-4, 1e-4 * (1.0 - jnp.asarray(mean)[0])]))

    model = LinearGaussianModel(
        dynamics_matrix=DYNAMICS,
        observation_matrix=SENSOR,
        observation_noise=[[1e-2]],
        prior=Belief(mean=[0.0, 0.0], cov=np.eye(2)),
        control_matrix=CONTROL,
        dynamics_noise_model=CallableProcessNoise(loses_a_direction, ()),
    )
    # It probes positive-semi-definite at the origin, so it constructs.
    world = World(model, initial_state=[2.0, 0.0])

    with pytest.raises(ValueError, match=r"dynamics_noise_model\.noise_at"):
        world.step([0.0], jax.random.PRNGKey(0))


def test_a_noiseless_dynamics_direction_is_still_legal():
    # The guard above must not reach past indefinite into semi-definite: a state
    # dimension with no process noise is a deterministic direction, not an error.
    def noiseless_second_direction(mean, params):
        return jnp.diag(jnp.array([1e-4, 0.0]))

    model = LinearGaussianModel(
        dynamics_matrix=DYNAMICS,
        observation_matrix=SENSOR,
        observation_noise=[[1e-2]],
        prior=Belief(mean=[0.0, 0.0], cov=np.eye(2)),
        control_matrix=CONTROL,
        dynamics_noise_model=CallableProcessNoise(noiseless_second_direction, ()),
    )
    world = World(model, initial_state=[1.0, 1.0])

    assert world.step([1.0], jax.random.PRNGKey(0)).shape == (1,)


def test_a_well_behaved_state_dependent_sensor_still_draws():
    world = World(_sensed_model(), initial_state=[1.0, 1.0])
    assert world.step([1.0], jax.random.PRNGKey(0)).shape == (1,)
