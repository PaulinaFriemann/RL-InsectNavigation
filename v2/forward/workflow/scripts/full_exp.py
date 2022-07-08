from functools import partial
import pickle
import os
import sys
import itertools as it
import numpy as np
import matplotlib.pyplot as plt
from insect_rl.mdp.utils import grid_math
from insect_rl.mdp.mrl import MyGridWorld, IAVICallback

from mushroom_rl.core import Core
import mushroom_rl.utils.dataset as mrl_dataset
from mushroom_rl.utils.callbacks import CollectDataset
from mushroom_rl.utils.parameters import Parameter
from mushroom_rl.policy.td_policy import EpsGreedy
import exp_funs


def mkdir_if_not_existant(dir_):
    try:
        # Create target Directory
        os.makedirs(dir_)
        print("Directory " , dir_ ,  " Created ") 
    except FileExistsError:
        print("Directory " , dir_ ,  " already exists")
        pass


#CONDITIONS = ["NOTRAP","TRAP"]
REWARD = int(sys.argv[2])# 10
TRAP_COST = int(sys.argv[3])#  -100

NUM_AGENTS = 5


with open(os.path.abspath("temp/agent.pickle"), 'rb') as agent_i:
    agent_cons = pickle.load(agent_i)
with open("Wystrach2020/env.pickle", 'rb') as env_i:
    env_settings = pickle.load(env_i)
learning_rate = float(sys.argv[4])
agent_cons.keywords["learning_rate"] = Parameter(learning_rate)

#agent_cons.keywords["policy"] = EpsGreedy(.3)

sim_settings = {
    'reward':REWARD,
    'trap_cost':TRAP_COST,
    'actions':vars(grid_math)["INTERCARDINALS"],
    'gamma':.99
}


res_dir = f"RESULTS/FULLEXPERIMENT/td/nomovecost/lr{learning_rate}/r{sim_settings['reward']}tc{sim_settings['trap_cost']}aINTERg{sim_settings['gamma']}"


mkdir_if_not_existant(res_dir)


def centroid(data):
    x, y = zip(*data)
    l = len(x)
    return int(round(sum(x) / l)), int(round(sum(y) / l))

def environment(configs, sim_settings, r):
    goals = configs.pop("goals")
    width = configs["width"]
    
    configs["width"] = configs["height"] # TODO WHYYY MUSHROOM_RL???
    configs["height"] = width

    #configs["goal"] = [(g[1], g[0]) for g in goals] # TODO only one possible
    configs["goal"] = centroid(goals)

    return MyGridWorld(**(configs | sim_settings), r=r)

r = {"0-no-trap": np.load("../insect_rl/mdp/fromIAVI/Rnotrap.npy").reshape((60,23)),
     "1-trap": np.load("../insect_rl/mdp/fromIAVI/Rtrap.npy").reshape((60,23))}


mdp = environment(env_settings, sim_settings, None)# r["0-no-trap"])
conditions = {"0-no-trap": mdp.close_trap,
              "1-trap": mdp.open_trap,
              "2-no-trap": mdp.close_trap,
}

from pathlib import Path
LOAD_AGENTS = False
#NUM_AGENTS = 5
if LOAD_AGENTS:
    agents = {}
    p = Path(res_dir) / '0-no-trap'
    for agent_dir in p.iterdir():
        idx = int(agent_dir.name[1])
        agents[idx] = agent_cons.func.load(agent_dir / f'agent_{idx}')

else:
    num = sys.argv[1]
    #agents = {i: agent_cons(mdp.info) for i in range(NUM_AGENTS)}
    agents = {num: agent_cons(mdp.info)}

its = [1000,2]

for i, (condition, startup_fun) in enumerate(conditions.items()):
    print("starting", condition)
    if condition == "2-no-trap":
        continue
    mkdir_if_not_existant(dir_:=f"{res_dir}/{condition}")
    startup_fun()
    for i_agent, agent in agents.items():
        print(condition, i_agent)
        os.mkdir(agent_dir:=f"{dir_}/a{i_agent}")
        td, ss = exp_funs.run_td_experiment(mdp, agent, i_agent, its[i], agent_dir)

        with open(f"{agent_dir}/tds.pickle", 'wb') as o:
            pickle.dump(td, o)

        with open(f"{agent_dir}/states.pickle", 'wb') as o:
            pickle.dump(ss, o)

exit()

ITERATIONS = 1000
print(vars(mdp))

for i, (condition, startup_fun) in enumerate(conditions.items()):
    print("starting", condition)
    mkdir_if_not_existant(dir_:=f"{res_dir}/{condition}")
    startup_fun()
    print(vars(mdp))
    print()
    #if i == 1:
    #    mdp.r = r["1-trap"]
    #if i == 2:
    #    mdp.r = r["0-no-trap"]

    for i_agent, agent in agents.items():
        print(condition, i_agent)
        os.mkdir(agent_dir:=f"{dir_}/a{i_agent}")
        exp_funs.run_experiment(mdp, agent, i_agent, ITERATIONS, agent_dir)
