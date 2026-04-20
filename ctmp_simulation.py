import logging
import warnings
from typing import Callable, ParamSpec, Concatenate

import numpy as np
from scipy.interpolate import interp1d

P = ParamSpec("P")


class EvolutionResult:
    def __init__(self, ts, history):
        self.ts: np.ndarray = ts
        self.history: np.ndarray = history

    def __iter__(self):
        """Allow unpacking (backwards compatibility)."""
        # warnings.warn(
        #     f"Unpacking a {self.__class__} is no longer recommended",
        #     category=DeprecationWarning,
        #     stacklevel=2,
        # )
        return iter([self.ts, self.history])


def evolution[S](
    ratefunc: Callable[Concatenate[float, S, P], dict[S, float]],
    tspan: tuple[float, float],
    state0: S,
    *,
    t_eval: np.ndarray | None = None,
    maxrate: float = 1e6,
    args: P.args = (),  # ty:ignore[invalid-parameter-default, invalid-paramspec]
    kwargs: P.kwargs = None,  # ty:ignore[invalid-parameter-default, invalid-paramspec]
) -> EvolutionResult:
    """Event-driven simulation of a continuous-time Markov process.

    The transition function `ratefunc` is a callable of the form
    f(t, state, *args) that should return a dictionary whose keys are
    the possible new states and whose values are the transition rates
    (probability per unit time).

    @param ratefunc transition function
    @param tspan tuple of (tmin, tmax)
    @param state0 initial state
    @param t_eval Array of times at which the output should be given
    @param maxrate Maximum allowed transition rate, abort the process if exceeded
    @param args additional args passed to ratefunc
    @param kwargs additional kwargs passed to ratefunc
    """

    if kwargs is None:
        kwargs: P.kwargs = {}

    tmin, tmax = tspan

    if t_eval is not None:
        if np.any(t_eval[:-1] >= t_eval[1:]):
            raise ValueError("t_eval must be sorted")
        if t_eval[0] < tmin:
            raise ValueError(f"{t_eval[0] = } but {tmin = }")
        if t_eval[-1] > tmax:
            raise ValueError(f"{t_eval[-1] = } but {tmax = }")

    state = state0
    t = tmin
    ts = [t]
    history = [state0]

    while t < tmax:
        transitions = ratefunc(t, state, *args, **kwargs)
        transitions = {k: transitions[k] for k in transitions if transitions[k] > 0}
        if not transitions:
            # No way to leave this state, staying here forever
            ts.append(tmax)
            history.append(state)
            break

        if maxrate:
            transitions = {
                k: transitions[k] for k in transitions if transitions[k] <= maxrate
            }
            if not transitions:
                warnings.warn(
                    f"Maximum rate {maxrate} exceeded by all possible transitions at time {t}.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                break

        logging.info(transitions)

        times_to_move = {
            k: np.random.exponential(1 / transitions[k]) for k in transitions
        }
        logging.info(times_to_move)

        new_state = min(times_to_move, key=lambda k: times_to_move[k])
        time_elapsed = times_to_move[new_state]
        if t + time_elapsed < tmax:
            # move to the next state
            t += time_elapsed
            state = new_state
            ts.append(t)
            history.append(state)
        else:
            # had to wait too long before going to next state
            # stay here instead
            ts.append(tmax)
            history.append(state)
            break

    ts = np.array(ts)
    history = np.array(history)

    if t_eval is None:
        return EvolutionResult(ts, history)

    interp = interp1d(ts, history, kind="previous", axis=0)
    return EvolutionResult(np.array(t_eval), np.array([interp(t) for t in t_eval]))
