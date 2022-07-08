import pickle
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from functools import partial
from matplotlib.colors import SymLogNorm, LogNorm

from matplotlib.patches import Circle, Rectangle
import random
import sys
from pathlib import Path
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




def plot_condition(condition, dir_, plot_dir, env, traps, trap_exits):
    AGENT_NUMBER = len(os.listdir(dir_))

    agent_ids = [i for i in range(0,AGENT_NUMBER)]
    #sample_ids = random.sample(agent_ids, min(AGENT_NUMBER, 5))

    SAVE_PLOTS = False

    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
        print("Directory " , plot_dir ,  " Created ")
    else:    
        print("Directory " , plot_dir ,  " already exists")

    tds_ns_list = []
    for i in agent_ids:
        with open(f"{dir_}/a{i}/tds_ns_{i}.npy",'rb') as f_in:
            tds_ns_list.append(np.load(f_in))
    # creating mean over all agents
    all_agent_mean_tds_ns = np.array(tds_ns_list).mean(axis=0)

    plt.figure(figsize=(10,10))
    plt.imshow(all_agent_mean_tds_ns[::-1], origin='lower', cmap="Greys", norm=LogNorm(vmin=0.01, vmax=all_agent_mean_tds_ns.max()/2))
    #plt.title("Mean TD error over all agents and all trials")
    #plt.title(f"mean cumulative TD error over all trials - {TITLES[condition]}")
    plt.colorbar()
    if SAVE_PLOTS:
        plt.savefig(f"{plot_dir}/td_error_allagents_alltrials.svg",bbox_inches='tight')
    else:
        plt.show()



TITLES = {
"0-no-trap": "trap covered",
"1-trap": "trap uncovered",
"2-no-trap": "trap covered again"
}

# start with 0-no-trap
CONDITIONS = ["0-no-trap", "1-trap", "2-no-trap"]


def run(dir_, plot_dir):
    with open("Wystrach2020/env.pickle", 'rb') as env_pickled:
        env = pickle.load(env_pickled)
    env['goal'] = centroid(env.pop('goals'))
    traps = env["traps"]
    trap_exits = env["trap_exits"]

    for condition in CONDITIONS:
        plot_condition(condition, dir_ / condition, plot_dir / condition, env, traps, trap_exits)


if __name__=="__main__":
    REWARD = 100
    TRAP_COST = -100
    LR = 0.5# float(sys.argv[3])
    MC = "nomovecost"#sys.argv[4]
    settings = f"{MC}/lr{LR}/r{REWARD}tc{TRAP_COST}aINTERg0.99"

    RESULTS_DIR = Path(f"RESULTS/FULLEXPERIMENT/{settings}")

    PLOT_DIR= Path(f"/home/paulina/Documents/thesis/figures/experiments/forward/td/{settings}")
    #PLOT_DIR = f"PLOTS/{settings}/{condition}"
    run(RESULTS_DIR, PLOT_DIR)
