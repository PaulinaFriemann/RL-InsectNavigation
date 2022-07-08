import itertools
from typing import Literal, Union
from functools import partialmethod, partial
import numpy as np


Action = tuple[int, int]

CARDINALS = ((1, 0), (0, 1), (-1, 0), (0, -1))
INTERCARDINALS = ((1, 0), (0, 1), (-1, 0), (0, -1), (-1,-1), (-1,1), (1,-1), (1,1))


def int_to_point(i: int, width: int) -> tuple[int, int]:
    """
    Convert a state int into the corresponding coordinate.

    i: State int.
    -> (x, y) int tuple.
    """

    return (i % width, i // width)


def point_to_int(p: tuple[int, int], width: int) -> int:
    """
    Convert a coordinate into the corresponding state int.

    p: (x, y) tuple.
    -> State int.
    """

    return p[0] + p[1]*width


def neighbouring(i: tuple[int, int], k: tuple[int, int], actions: list[tuple[int, int]]) -> bool:
    """
    Get whether two points neighbour each other with the current action set.

    i: (x, y) int tuple.
    k: (x, y) int tuple.
    -> bool.
    """
    return any(
        (i[0] + a[0], i[1] + a[1])==k for a in actions
    )


def feature_vector(i: int, width: int, height: int,
    feature_map: Literal["ident", "coord", "proxi"] ="ident") -> list[Union[int, float]]:
    """
    Get the feature vector associated with a state integer.

    i: State int.
    feature_map: Which feature map to use (default ident). String in {ident,
        coord, proxi}.
    -> Feature vector.
    """
    n_states = width * height
    if feature_map == "coord":
        if width != height:
            raise NotImplementedError("only implemented for quadradict environments")
        f = np.zeros(width) #np.zeros(self.grid_size) # TODO figure out how to use for non-quadr.
        x, y = int_to_point(i, width) #i % width, i // width
        f[x] += 1
        f[y] += 1
        return f
    if feature_map == "proxi":
        f = np.zeros(n_states)
        x, y = int_to_point(i, width)
        for b, a in itertools.product(range(height), range(width)):
            dist = abs(x - a) + abs(y - b)
            f[point_to_int((a, b), width)] = dist
        return f
    # Assume identity map.
    f = np.zeros(n_states)
    f[i] = 1
    return f


def feature_matrix(width: int, height: int,
    feature_map : Literal["ident", "coord", "proxi"] ="ident") -> np.array:
    """
    Get the feature matrix for this gridworld.

    feature_map: Which feature map to use (default ident). String in {ident,
        coord, proxi}.
    -> NumPy array with shape (n_states, d_states).
    """
    n_states = width * height
    features = []
    for n in range(n_states):
        f = feature_vector(n, width, height, feature_map)
        features.append(f)
    return np.array(features)


def get_action_probabilities(trajectories: np.array, nS: int, nA: int) -> np.array:
    """
    Arguments
    ---------
    trajectories: np.array
        shape (n_trajectories, trajectory_len, 4) with (state int, action int, reward float, next state int)
    nS: int
        Numer of states
    nA: int
        Number of actions
    
    Returns
    -------
    np.array of shape (nS, nA)
    """
    action_probabilities = np.zeros((nS, nA))
    for traj in trajectories:
        for (s, a, _, _) in traj:
            action_probabilities[s][a] += 1
    action_probabilities[action_probabilities.sum(axis=1) == 0] = 1e-5
    action_probabilities /= action_probabilities.sum(axis=1).reshape(nS, 1)
    return action_probabilities



def transition_probability(i: int, j: int, k: int, width: int, height: int, actions: tuple, wind: float) -> float:
    """
    Get the probability of transitioning from state i to state k given
    action j.

    i: State int.
    j: Action int.
    k: State int.
    -> p(s_k | s_i, a_j)
    """
    xi, yi = int_to_point(i, width)
    xj, yj = actions[j]
    xk, yk = int_to_point(k, width)

    if not neighbouring((xi, yi), (xk, yk), actions):
        return 0.0
    
    # TODO check that this works
    if (xj, yj) not in actions:
        assert isinstance(actions[0], tuple)
        return 0.0

    n_actions = len(actions)
    # Is k the intended state to move to?
    if (xi + xj, yi + yj) == (xk, yk):
        return 1 - wind + wind/n_actions

    # If these are not the same point, then we can move there by wind.
    if (xi, yi) != (xk, yk):
        return wind/n_actions

    # If these are the same point, we can only move here by either moving
    # off the grid or being blown off the grid. Are we on a corner or not?
    if (xi, yi) in {(0, 0), (width-1, height-1),
                    (0, height-1), (width-1, 0)}:
        # Corner.
        # Can move off the edge in two directions.
        # Did we intend to move off the grid?
        if not (0 <= xi + xj < width and
                0 <= yi + yj < height):
            # We intended to move off the grid, so we have the regular
            # success chance of staying here plus an extra chance of blowing
            # onto the *other* off-grid square.
            return 1 - wind + 2*wind/n_actions
        else:
            # We can blow off the grid in either direction only by wind.
            return 2*wind/n_actions

    # Not a corner. Is it an edge?
    if (xi not in {0, width-1} and
        yi not in {0, height-1}):
        # Not an edge.
        return 0.0

    # Edge.
    # Can only move off the edge in one direction.
    # Did we intend to move off the grid?
    if not (0 <= xi + xj < width and
            0 <= yi + yj < height):
        # We intended to move off the grid, so we have the regular
        # success chance of staying here.
        return 1 - wind + wind/n_actions
    else:
        # We can blow off the grid only by wind.
        return wind/n_actions
