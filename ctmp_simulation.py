import logging
import warnings
from typing import Any, Callable, TypeVar

import numpy as np
from scipy.interpolate import interp1d

S = TypeVar("S")


def evolution(
    ratefunc: Callable[[float, S], dict[S, float]],
    tspan: tuple[float, float],
    state0: S,
    *,
    t_eval=None,
    maxrate: float = 1e6,
    args: tuple = (),
    kwargs: dict[str, Any] | None = None,
):
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
        kwargs = {}

    tmin, tmax = tspan

    state = state0
    t = tmin
    ts = [t]
    history = [state0]

    while t < tmax:
        transitions = ratefunc(t, state, *args, **kwargs)
        transitions = {k: transitions[k] for k in transitions if transitions[k] > 0}
        if not transitions:
            break

        if maxrate:
            transitions = {
                k: transitions[k] for k in transitions if transitions[k] < maxrate
            }
            if not transitions:
                warnings.warn(
                    f"Maximum rate {maxrate} exceeded by all possible transitions at time {t}.",
                    RuntimeWarning,
                )
                break

        logging.info(transitions)

        times_to_move = {
            k: np.random.exponential(1 / transitions[k]) for k in transitions
        }
        logging.info(times_to_move)

        new_state = min(times_to_move, key=lambda k: times_to_move[k])
        time_elapsed = times_to_move[new_state]
        t += time_elapsed
        state = new_state
        ts.append(t)
        history.append(state)

    ts = np.array(ts)
    history = np.array(history)

    if t_eval is None:
        return ts, history

    interp = interp1d(ts, history, kind="previous", axis=0)
    return np.array(t_eval), np.array([interp(t) for t in t_eval])
