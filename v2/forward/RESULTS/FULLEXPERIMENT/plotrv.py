import math
import pickle
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from functools import partial
from matplotlib.patches import Circle, Rectangle
from matplotlib.colors import SymLogNorm, LogNorm
import random
from pathlib import Path
import sys
import os
from insect_rl.mdp.mrl import INTERCARDINALS

# don't hate me
import warnings
import matplotlib.cbook
warnings.filterwarnings("ignore",category=matplotlib.cbook.mplDeprecation)
warnings.filterwarnings("ignore",category=FutureWarning)

if "../workflow/scripts" not in sys.path:
    sys.path.insert(0, os.path.abspath("workflow/scripts"))
from plotting import plots


# RESULTS/RESULTSNOTRAP/r100tc-100aINTERg0.99/a0/J_0.pickle
#df = pd.read_csv("a0/df_0.csv")

def centroid(data):
    x, y = zip(*data)
    l = len(x)
    return int(round(sum(x) / l)), int(round(sum(y) / l))

with open("Wystrach2020/env.pickle", 'rb') as env_pickled:
    env = pickle.load(env_pickled)
env['goal'] = centroid(env.pop('goals'))
traps = env["traps"]
trap_exits = env["trap_exits"]


REWARD = int(sys.argv[1])# 10
TRAP_COST = int(sys.argv[2])#  -100
LR = float(sys.argv[3])
MC = sys.argv[4]
settings = f"{MC}/lr{LR}/r{REWARD}tc{TRAP_COST}aINTERg0.99"

RESULTS_DIR = Path(f"RESULTS/FULLEXPERIMENT/{settings}")

PLOT_DIR= Path(f"/home/paulina/Documents/thesis/figures/experiments/forward/{settings}")
CONDITIONS = ["0-no-trap", "1-trap", "2-no-trap"]

if not os.path.exists(PLOT_DIR):
    os.makedirs(PLOT_DIR)
    print("Directory " , PLOT_DIR ,  " Created ")
else:    
    print("Directory " , PLOT_DIR ,  " already exists")
traps = env["traps"]
trap_exits = env["trap_exits"]

V = np.zeros((60,23))

for condition in CONDITIONS:
    dir_ = Path(f"RESULTS/FULLEXPERIMENT/{settings}/{condition}")

    AGENT_NUMBER = len(os.listdir(dir_))

    agent_ids = [i for i in range(0,AGENT_NUMBER)]
    #sample_ids = random.sample(agent_ids, min(AGENT_NUMBER, 5))

    SAVE_PLOTS = True


    value_list = []
    for i in agent_ids:
        with open(f"{dir_}/a{i}/value_fun_{i}.npy",'rb') as f_in:
            value_list.append(np.load(f_in))
    # creating mean over all agents
    all_agent_mean_value = np.array(value_list).mean(axis=0)

    for i in range(60):
        for j in range(23):
            if all_agent_mean_value[i,j] != 0:
                V[i,j] = all_agent_mean_value[i,j]

    plt.figure(figsize=(10,10))
    plt.imshow(V[::-1], origin='lower', norm=SymLogNorm(10)) # cmap="gnuplot",

    patches = [
        Rectangle((env["goal"][0]-.5, env["goal"][1]-.5), width=1, height=1, edgecolor='red', linewidth=3.0, fill=False),
        Rectangle((env["start"][0]-.5,env["start"][1]-.5), width=1, height=1, edgecolor='green', linewidth=3.0, fill=False)
    ]
    if condition=="1-trap":
        patches.extend([Rectangle((traps[0][0]-.5, traps[0][1]-.5), width=18, height=1, edgecolor='dimgray', linewidth=3.0, fill=False),
                        Rectangle((env["trap_exits"][0][0][0]-.5,env["trap_exits"][0][0][1]-.5), width=2.0, height=1, edgecolor='gray', linewidth=3.0, fill=False)
        ])
    else:
        patches.extend([Rectangle((trap[0]-.5, trap[1]-.5), width=1, height=1, edgecolor='dimgray', linewidth=3.0, fill=False) for trap in traps[::2]])

    for patch in patches:
        plt.gca().add_patch(patch, )

    plt.gca().set_xticks([])
    plt.gca().set_yticks([])

    #plt.title(f"Value function - {TITLES[condition]}")
    clb = plt.colorbar(ticks=[0,10,-10,20,30,-20,-30,-70,70,100,50,-50,-100,-200,-300,200], format = "%d")
    clb.ax.set_title('V(s)')
    #clb.set_ticks([round(t,1) for t in clb.get_ticks()])
    plt.show()
    #if SAVE_PLOTS:
    #    plt.savefig(f"{plot_dir}/Value_Func_Mean.svg",bbox_inches='tight')
    #plt.clf()


TITLES = {
"0-no-trap": "trap covered",
"1-trap": "trap uncovered",
"2-no-trap": "trap covered again"
}

# start with 0-no-trap


