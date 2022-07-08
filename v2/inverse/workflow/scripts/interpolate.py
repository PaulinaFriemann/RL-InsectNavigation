import numpy as np
import pandas as pd

import sys
import os
#sys.path.insert(0, os.path.abspath("src/"))
from insect_rl.mdp.utils import grid_math, algebra, policies
# from mdp.utils import algebra




def combine_utilities(actions, utilities):
    utilities = {
                    action: sum(pressure * utility[action] for pressure, utility in utilities)
                        for action in actions
                }
    return utilities


def calculate_utilities(actions, delta):
    # utility of the actions according to how similar they are to the homing vector
    similarity_to_delta = {action: algebra.similarity(action, delta) for action in actions}
    #move_cost = {action: np.linalg.norm(action) for action in actions}
    # mix utility from homing vector with utility from memory
    return similarity_to_delta


def _interpolate(s1, s2, actions):
    delta = s2 - s1
    utils = calculate_utilities(actions, delta)
    boltzmann = policies.boltzmann(actions, utils)
    rng = np.random.default_rng()
    return rng.choice(actions, p=boltzmann)


def interpolate(trajectory, actions):
    traj_iter = iter(trajectory)
    s1 = next(traj_iter)
    s2 = next(traj_iter)
    interpolated = [s1]
    while s2 is not None:
        s1 = np.array(s1)
        s2 = np.array(s2)
        if grid_math.neighbouring(s1, s2, actions):
            interpolated.append(tuple(s2))
            s1 = s2
            s2 = next(traj_iter, None)
        else:
            inter = _interpolate(s1, s2, actions)
            s1 = s1 + inter
            interpolated.append(tuple(s1))

    return interpolated


if __name__ == "__main__":
    dat = pd.read_csv(snakemake.input[0])
    trajectories = []
    for ant in pd.unique(dat["ant_nb"]):
        traj = dat[dat.ant_nb == ant]
        traj = list(zip(traj.path_x, traj.path_y))
        trajectories.append(interpolate(traj, grid_math.__dict__[snakemake.config["actions"]]))
    # TODO put them back in the df
