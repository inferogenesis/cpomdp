"""The world and the agent as separate objects, with no reference between them.

``World`` runs the process that produces observations. ``ScoredAgent`` runs the model
it filters with. Neither holds the other: ``World`` has no accessor returning its model
or any parameter of it, and ``ScoredAgent`` has no constructor slot for a ``World``.
Nothing here carries a reference from either one to the other.

Actions arrive from outside. ``driven_step`` takes the action to predict with, so one
sequence can drive several agents. There is no selector here and no ``sample_action``.

``drive`` runs one world and any number of agents over an
``ExogenousActionSequence``: the world advances once per step, and every agent folds
the same reading. What comes back is a ``DrivenRun``, which carries the declaration
that the control loop was cut alongside the trajectories, so no number leaves here
without it.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float64
from numpy.typing import ArrayLike

from cpomdp._validation import state_dependent, validate_covariance
from cpomdp.backends.base import InferenceBackend
from cpomdp.backends.kalman import KalmanBackend
from cpomdp.types import Belief, LinearGaussianModel

__all__ = [
    "SEVERED_CONTROL_LOOP",
    "DrivenRun",
    "ExogenousActionSequence",
    "ModellingChoice",
    "ScoredAgent",
    "World",
    "drive",
]


class World:
    """The process an agent is scored against: true state in, observations out.

    Holds a ``LinearGaussianModel`` privately and advances one true state under it.
    Each ``step`` applies an action, draws the process noise, and returns a noisy
    observation of the state it arrived at. There is no ``model`` attribute and no
    accessor returning a parameter of it.

    ``state`` is readable, so a run can score against the true trajectory.

    State-dependent noise is read at the state it belongs to. ``R(x)`` is the sensor's,
    so it is read at the state the step arrived at, which is the state being measured.
    ``Q(x)`` is the diffusion of the arrived-at state, which is not known until the
    diffusion has been drawn, so it is read at the mean the step pushes forward to. That
    mean is what a filter's ``μ⁻`` estimates, so a world reads where a filter reads once
    the filter's belief sits on the truth.
    """

    def __init__(
        self,
        model: LinearGaussianModel,
        *,
        initial_state: ArrayLike | None = None,
    ) -> None:
        """Build a world running ``model``, starting at ``initial_state``.

        Args:
            model: The process that generates observations — p*. Kept private; the
                world exposes no route back to it.
            initial_state: The true state to start from, shape ``(n,)``. Defaults to
                the model prior's mean.

        Raises:
            ValueError: If ``initial_state`` is not a 1-D vector of length ``n``.
        """
        self._model = model
        if initial_state is None:
            state = model.prior.mean
        else:
            state = jnp.asarray(initial_state, dtype=float)
            if state.shape != (model.n_states,):
                raise ValueError(
                    f"initial_state must be a 1-D vector of length "
                    f"{model.n_states} (the state dimension), got shape {state.shape}"
                )
        self._state = state

    @property
    def state(self) -> Float64[Array, "n"]:
        """The current true state, shape ``(n,)``."""
        return self._state

    @property
    def n_states(self) -> int:
        """Dimension of the true state (n)."""
        return self._model.n_states

    @property
    def n_observations(self) -> int:
        """Dimension of an observation (m)."""
        return self._model.n_observations

    @property
    def n_controls(self) -> int:
        """Dimension of an action (p); 0 if the process takes no action."""
        return self._model.n_controls

    def step(self, action: ArrayLike | None, key: Array) -> Float64[Array, "m"]:
        """Advance the true state by one step and observe it.

        The action drives the transition out of the current state; the observation
        reads the state it arrives at. Both noise draws come from ``key``, so two
        worlds given the same key and the same start produce the same reading.

        A state-dependent ``Q(x)`` is read at the mean this step pushes forward to, and
        a state-dependent ``R(x)`` at the state drawn around it.

        Args:
            action: The action applied over this step, shape ``(p,)``. ``None`` for a
                process with no control matrix.
            key: A JAX PRNG key, split internally into a process draw and a sensor
                draw. Pass a fresh key per step or the trajectory repeats.

        Returns:
            One observation, shape ``(m,)``.

        Raises:
            ValueError: If the process has a control matrix and ``action`` is ``None``
                or not shape ``(p,)``.
        """
        model = self._model
        key_dynamics, key_observation = jax.random.split(key)

        mean_next = model.dynamics_matrix @ self._state
        if model.control_matrix is not None:
            if action is None:
                raise ValueError(
                    "this world has a control matrix; step requires an action"
                )
            action = jnp.asarray(action, dtype=float)
            if action.shape != (model.n_controls,):
                raise ValueError(
                    f"action must be a 1-D vector of length {model.n_controls} "
                    f"(the action dimension), got shape {action.shape}"
                )
            mean_next = mean_next + model.control_matrix @ action

        process = state_dependent(model.dynamics_noise_model)
        if process is None:
            dynamics_noise = model.dynamics_noise
        else:
            dynamics_noise = process.noise_at(mean_next)
            # A process noise is only probed at construction, at one state. The svd
            # draw below factors through the *singular* values, so a direction that
            # has gone negative by the state the walk reaches is drawn as |λ| rather
            # than as NaN: a finite, plausible state sampled from the wrong Q, with
            # nothing raised. Semi-definite stays legal here, indefinite does not.
            validate_covariance(dynamics_noise, "dynamics_noise_model.noise_at(x)")
        # svd rather than the cholesky default: dynamics_noise is only required
        # positive *semi*-definite, so a noiseless state dimension is legal here.
        self._state = jax.random.multivariate_normal(
            key_dynamics, mean_next, dynamics_noise, method="svd"
        )

        sensor = state_dependent(model.observation_model)
        if sensor is None:
            observation_matrix = model.observation_matrix
            observation_noise = model.observation_noise
        else:
            # linearize is exact here rather than approximate: the observation mean is
            # linear in this regime, so the Jacobian it returns is the map itself. The
            # model probes the sensor's Jacobian against its declared
            # observation_matrix once, at prior.mean, so a nonlinear-mean sensor
            # (issue #21) is kept out at construction rather than at every state.
            observation_matrix, observation_noise = sensor.linearize(self._state)
            # A sensor noise is only probed at construction, at one state. Where it
            # loses definiteness at a state the walk reaches, the cholesky draw below
            # returns NaN, and a NaN reading reaches every agent's belief and every
            # score with nothing raised. The dynamics draw above takes svd because a
            # noiseless direction is legal there. This one is not legal anywhere.
            validate_covariance(
                observation_noise,
                "observation_model.linearize(x)[1]",
                require_definite=True,
            )
        return jax.random.multivariate_normal(
            key_observation,
            observation_matrix @ self._state,
            observation_noise,
        )


class ScoredAgent:
    """An agent that filters under its own model, driven by actions from outside.

    Owns a ``LinearGaussianModel`` and the belief it carries forward. Each
    ``driven_step`` folds one observation in, predicting with the action it is handed.
    The agent neither chooses nor remembers actions, and has no selector and no
    ``sample_action``.

    ``ScoredAgent(model)`` filters with a per-step ``KalmanBackend``.
    ``ScoredAgent.from_backend(backend)`` filters with any other engine, a degraded one
    included, under the backend's model.
    """

    def __init__(self, model: LinearGaussianModel) -> None:
        """Build an agent filtering under ``model`` with a per-step ``KalmanBackend``.

        Args:
            model: The agent's own generative model — p.
        """
        self._setup(KalmanBackend(model))

    @classmethod
    def from_backend(cls, backend: InferenceBackend) -> "ScoredAgent":
        """Build an agent that filters with ``backend`` under its model.

        For a degraded or replaced filter. The agent's model is ``backend.model``, so
        the model it is scored under and the one it filters with cannot differ.

        Args:
            backend: The inference engine, any ``InferenceBackend``.

        Returns:
            The agent.
        """
        agent = cls.__new__(cls)
        agent._setup(backend)
        return agent

    def _setup(self, backend: InferenceBackend) -> None:
        """Hold the backend and start from its model's prior."""
        self._backend = backend
        self.belief = backend.model.prior

    @property
    def model(self) -> LinearGaussianModel:
        """The model the agent filters and is scored under, read-only.

        It is the backend's model, fixed when the backend was built.
        """
        return self._backend.model

    def driven_step(self, observation: ArrayLike, action: ArrayLike) -> Belief:
        """Fold one observation in, predicting with the action supplied.

        The current belief goes in as the prior and the posterior comes back out and is
        stored, which is the recursive filter advanced one step. The belief is replaced
        rather than mutated.

        Args:
            observation: The reading for this step, shape ``(m,)``.
            action: The action applied since the previous observation, shape ``(p,)``.

        Returns:
            The updated belief, also stored on ``self.belief``.

        Raises:
            ValueError: On a shape mismatch in ``observation`` or ``action``, enforced
                by the backend's ``_validate_step_inputs``.
        """
        self.belief = self._backend.infer_states(observation, self.belief, action)
        return self.belief


@dataclass(frozen=True)
class ModellingChoice:
    """A choice made in building a run, stated where the numbers are read.

    Args:
        name: What the choice is called.
        statement: What was chosen, and what it costs.
        contested_by: The reading under which the choice changes the answer.
    """

    name: str
    statement: str
    contested_by: str


SEVERED_CONTROL_LOOP = ModellingChoice(
    name="exogenous action",
    statement=(
        "Actions were imposed from a declared sequence rather than chosen by the "
        "agents, which is what makes the entropy of the true process a shared constant "
        "and lets it cancel between agents. It also cuts the control loop."
    ),
    contested_by=(
        "Under a state-dependent R(x) an agent choosing its own actions steers toward "
        "low-noise regions and so changes its own inference gap. A gap measured under "
        "an imposed sequence can misrepresent the closed-loop one."
    ),
)
"""The choice every run through ``drive`` makes, carried on each ``DrivenRun``."""


@dataclass(frozen=True)
class ExogenousActionSequence:
    """The control sequence driven into every agent under comparison.

    One sequence for the whole run, declared ahead of it and versioned, so an action
    changed later shows up in the diff rather than in the prose.

    Args:
        actions: The actions in order, shape ``(k, p)`` — ``k`` steps of dimension
            ``p``. At least one.
        version: A non-empty tag.
    """

    actions: Float64[Array, "k p"]
    version: str

    def __init__(self, actions: ArrayLike, *, version: str) -> None:
        object.__setattr__(self, "actions", jnp.asarray(actions, dtype=float))
        object.__setattr__(self, "version", version)
        if not isinstance(version, str) or not version:
            raise ValueError(
                "version must be a non-empty string — the sequence is declared and "
                "versioned"
            )
        if self.actions.ndim != 2:
            raise ValueError(
                f"actions must be a 2-D (k, p) array, got shape {self.actions.shape}"
            )
        if self.actions.shape[0] < 1:
            raise ValueError("a sequence needs at least one action to drive")

    @property
    def horizon(self) -> int:
        """The step count ``k``."""
        return int(self.actions.shape[0])

    @property
    def n_controls(self) -> int:
        """The action dimension ``p``, named as the model names it."""
        return int(self.actions.shape[1])


@dataclass(frozen=True)
class DrivenRun:
    """What one world and its agents produced, and the choice that produced it.

    Args:
        observations: The readings the world emitted, shape ``(k, m)``. Every agent
            folded these, in this order.
        states: The true states the world passed through, shape ``(k, n)``.
        beliefs: Each agent's belief after each step, keyed by the agent's name.
        action_sequence: The sequence that drove the run. A score reads its actions
            from here, so it cannot pair the readings with another sequence.
        control_loop: The declaration that the loop was cut. Required, so a run cannot
            hand back numbers without it.
    """

    observations: Float64[Array, "k m"]
    states: Float64[Array, "k n"]
    beliefs: Mapping[str, tuple[Belief, ...]]
    action_sequence: ExogenousActionSequence
    control_loop: ModellingChoice

    @property
    def action_sequence_version(self) -> str:
        """The version of the sequence that drove the run."""
        return self.action_sequence.version

    @property
    def final_beliefs(self) -> Mapping[str, Belief]:
        """Each agent's last belief, keyed by name."""
        return MappingProxyType(
            {name: trajectory[-1] for name, trajectory in self.beliefs.items()}
        )


def drive(
    world: World,
    agents: Mapping[str, ScoredAgent],
    sequence: ExogenousActionSequence,
    key: Array,
) -> DrivenRun:
    """Run ``world`` over ``sequence``, folding every reading into every agent.

    The world advances once per step, not once per agent, so all the agents face one
    trajectory and one stream of readings. That is what leaves their beliefs comparable.

    Args:
        world: The process the run is against. Advanced in place.
        agents: The agents to drive, keyed by the name their results are reported under.
            Advanced in place too, so each holds its final belief when this returns.
            May be empty, which produces the trajectory alone.
        sequence: The actions to drive, one per step.
        key: A JAX PRNG key, split once per step.

    Returns:
        A ``DrivenRun`` carrying the trajectory, the beliefs, and the declaration.

    Raises:
        ValueError: If the sequence's action dimension is not the world's.
    """
    if sequence.n_controls != world.n_controls:
        raise ValueError(
            f"the sequence drives p={sequence.n_controls} but the world takes "
            f"p={world.n_controls}; the declared actions must match the action "
            f"dimension."
        )
    keys = jax.random.split(key, sequence.horizon)
    observations, states = [], []
    trajectories: dict[str, list[Belief]] = {name: [] for name in agents}
    for action, step_key in zip(sequence.actions, keys, strict=True):
        observation = world.step(action, step_key)
        observations.append(observation)
        states.append(world.state)
        for name, agent in agents.items():
            trajectories[name].append(agent.driven_step(observation, action))
    return DrivenRun(
        observations=jnp.stack(observations),
        states=jnp.stack(states),
        beliefs=MappingProxyType(
            {name: tuple(beliefs) for name, beliefs in trajectories.items()}
        ),
        action_sequence=sequence,
        control_loop=SEVERED_CONTROL_LOOP,
    )
