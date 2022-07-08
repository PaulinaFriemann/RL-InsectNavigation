from datetime import datetime
import pandas as pd
from icecream import install
install()
from tqdm import tqdm
import itertools
import os
import sys
import numpy as np

from mdp.grid_math import GridMath, INTERCARDINALS, CARDINALS, Action, transition_probability, neighbouring
import simulation

ACTIONS = CARDINALS


def angle(action, homing_vector):
    """Angle between an action and the homing vector, in radians [0,pi].
    For action (0,0), return pi
    """
    if not any(action): # (0,0) action
        return np.pi
    a = np.array(action, dtype=float)
    b = np.array(homing_vector, dtype=float)
    return np.arctan2(np.linalg.norm(np.cross(a,b)), np.dot(a,b))


def similarity(action, homing_vector):
    # inversely scale vector angle to [-1,1]
    return 2* (1 - angle(action, homing_vector) / np.pi) - 1


def combine_utilities(actions, utilities):
    utilities = {
                    action: sum(pressure * utility[action] for pressure, utility in utilities)
                        for action in actions
                }
    return utilities


def calculate_utilities(actions, delta):
    # utility of the actions according to how similar they are to the homing vector
    similarity_to_delta = {action: similarity(action, delta) for action in actions}
    #move_cost = {action: np.linalg.norm(action) for action in actions}
    # mix utility from homing vector with utility from memory
    return similarity_to_delta


def calc_boltzmann(actions, utilities):
    # calculate Boltzmann distribution.
    boltzman_distribution = []
    for a in actions:
        boltzman_distribution.append(np.exp(utilities[a]))
    boltzman_distribution = np.array(boltzman_distribution)
    boltzman_distribution /= np.sum(boltzman_distribution)
    return boltzman_distribution


def _interpolate(s1, s2, actions):
    #actions = list(actions)
    delta = s2 - s1
    utils = calculate_utilities(actions, delta)
    boltzmann = calc_boltzmann(actions, utils)
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
        if neighbouring(s1, s2, actions):
            interpolated.append(tuple(s2))
            s1 = s2
            s2 = next(traj_iter, None)
        else:
            inter = _interpolate(s1, s2, actions)
            s1 = s1 + inter
            interpolated.append(tuple(s1))

    return interpolated


data_file = "./data/deriveddata/Melophorus_Individual_Data_Paths_Discretized.csv"

dat = pd.read_csv(data_file)

trial_nb = 1 # 0 = no trap, 1 = trap

print(dat)

# before trap setting
trajectories = []
for ant in pd.unique(dat[dat["trial_nb"] == trial_nb]["ant_nb"]):
    traj = dat[(dat.trial_nb == trial_nb) & (dat.ant_nb == ant)]
    traj = list(zip(traj.pathX, traj.pathY))
    trajectories.append(interpolate(traj, ACTIONS))

rewards = [10, 1, 100]
winds = [0.5, 0.8]
trap_costs = [-10]

iterations = 10000

env = simulation.WYSTRACH2020_ENVIRONMENT
#env["traps"] = []
#env["trap_exits"] = []
env["goals"] = [trajectory[-1] for trajectory in trajectories]
envc = simulation.EnvironmentConfig.from_kwargs(**env)

dir_name = f"./results/MelophorusIndivDiscrete/{envc.width}x{envc.height}/"
# dd/mm/YY H:M:S
dir_name += datetime.now().strftime("%d_%m_%Y_%H_%M") + "/"
os.makedirs(dir_name + "gridworld_imgs/")


with open(dir_name + 'readme.txt', 'w') as f:
    f.write(f'{envc.start=}\n{envc.goals=}\n{envc.traps=}\n{envc.trap_exits=}\n{ACTIONS=}\n{iterations=}')

for reward, wind, trap_cost in tqdm(itertools.product(rewards, winds, trap_costs)):
    print("run simulation")
    res = simulation.run_with_data(trajectories, envc, ACTIONS, reward, wind, trap_cost, res_dir=dir_name, save_plots=True)
    for (name, arr) in res.items():
        np.savetxt(f"{dir_name}{name}.csv", arr, delimiter=",")
