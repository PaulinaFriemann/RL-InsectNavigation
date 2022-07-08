import itertools
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



def plot_line(dir_, ids, data_name, xlabel, ylabel, line_label):
  SMOOTHING_FACTOR = 20

  data = []
  for i in ids:
    with open(f"{dir_}/a{i}/{data_name}_{i}.pickle",'rb') as f_in:
      data.append(pickle.load(f_in))
  # Smoothing by Factor  
  data_tupls = []
  N = SMOOTHING_FACTOR
  for D in data:
    D = np.convolve(D, np.ones(N)/N, mode='valid')
    for idx, d in enumerate(D):
      data_tupls.append((idx,d))
  df_data = pd.DataFrame(data_tupls, columns = ['x', 'y'])

  sns.lineplot(data=df_data, x="x", y="y", ci=None, legend='brief', label=line_label)
  ax = plt.gca()
  #ax.set_ylim(df_J.y.min(),df_J.y.max() +2)
  ax.set_xlabel(xlabel)
  ax.set_ylabel(ylabel)
  #ax.set_title(title)


def read_last_data(data_name, dir_):
    data = []
    for i in range(len(os.listdir(dir_))):
        with open(f"{dir_}/a{i}/{data_name}_{i}.pickle",'rb') as f_in:
            data.append(pickle.load(f_in)[-1])
    return data

def read_data(data_name, dir_):
    data = []
    for i in range(len(os.listdir(dir_))):
        with open(f"{dir_}/a{i}/{data_name}_{i}.pickle",'rb') as f_in:
            data.append(pickle.load(f_in))
    return data


if __name__=="__main__":
    REWARD = [100]
    TRAP_COST = [-100]
    LR = [0.5,0.2,0.8]
    MC = ["nomovecost"]

    conditions = ["0-no-trap", "1-trap", "2-no-trap"]

    for condition in conditions:
        print(condition)

        G_dict = {}
        len_dict = {}
        len_dict_all = {}

        for mc,lr,r,tc in itertools.product(MC, LR, REWARD, TRAP_COST):
            settings = f"{mc}/lr{lr}/r{r}tc{tc}aINTERg0.99"
            RESULTS_DIR = Path(f"RESULTS/FULLEXPERIMENT/{settings}/{condition}")
            if RESULTS_DIR.exists():
                print(RESULTS_DIR)
                mc_title = "no move cost" if mc=="nomovecost" else "move cost"
                lr_title = r'$\alpha=$' + str(lr)
                agent_ids = range(len(os.listdir(RESULTS_DIR)))
                plot_line(RESULTS_DIR, agent_ids, data_name='episode_lens' , xlabel='episodes', ylabel='steps', line_label=f"{lr_title}, r={r}, tc={tc}")
        plt.show()
    

        for mc,lr,r,tc in itertools.product(MC, LR, REWARD, TRAP_COST):
            settings = f"{mc}/lr{lr}/r{r}tc{tc}aINTERg0.99"
            RESULTS_DIR = Path(f"RESULTS/FULLEXPERIMENT/{settings}/{condition}")
            if RESULTS_DIR.exists():
                print(RESULTS_DIR)
                mc_title = "no move cost" if mc=="nomovecost" else "move cost"
                lr_title = r'$\alpha=$' + str(lr)
                agent_ids = range(len(os.listdir(RESULTS_DIR)))
                plot_line(RESULTS_DIR, agent_ids, data_name='J' , xlabel='episodes', ylabel='G', line_label=f"{lr_title}, r={r}, tc={tc}")
        plt.show()


        #############
        continue

        for mc,lr,r,tc in itertools.product(MC, LR, REWARD, TRAP_COST):
            settings = f"{mc}/lr{lr}/r{r}tc{tc}aINTERg0.99"
            RESULTS_DIR = Path(f"RESULTS/FULLEXPERIMENT/{settings}/{condition}")
            if RESULTS_DIR.exists():
                print(RESULTS_DIR)
                G_dict[(mc,lr,r,tc)] = read_last_data("J", RESULTS_DIR)
                len_dict[(mc,lr,r,tc)] = read_last_data("episode_lens", RESULTS_DIR)
                len_dict_all[(mc,lr,r,tc)] = read_data("episode_lens", RESULTS_DIR)

        print("G:",G_dict)
        print()
        print({s: str(np.round(np.mean(v),1)) + " (" + str(np.round(np.std(v),1)) + ")" for s, v in G_dict.items()})
        print()
        print("lens:",len_dict)
        print()
        print({s: str(np.round(np.mean(v),1)) + " (" + str(np.round(np.std(v),1)) + ")" for s, v in len_dict.items()})

        df = pd.DataFrame.from_dict(G_dict, orient='index')
        df.index.rename('Parameters', inplace=True)

        stacked = df.stack().reset_index()
        stacked.rename(columns={'level_1': 'Ant', 0: 'G'}, inplace=True)

        sns.swarmplot(data=stacked, x='Parameters', y='G')
        plt.xticks(rotation=30, ha='right')
        #plt.show()
        plt.clf()

        df = pd.DataFrame.from_dict(len_dict, orient='index')
        df.index.rename('Parameters', inplace=True)

        stacked = df.stack().reset_index()
        stacked.rename(columns={'level_1': 'Ant', 0: 'episode_lens'}, inplace=True)

        sns.swarmplot(data=stacked, x='Parameters', y='episode_lens')
        plt.xticks(rotation=30, ha='right')
        plt.show()
        sns.boxplot(data=stacked, x='Parameters', y='episode_lens')
        plt.xticks(rotation=30, ha='right')
        plt.show()
        sns.boxenplot(data=stacked, x='Parameters', y='episode_lens')
        plt.xticks(rotation=30, ha='right')
        plt.show()

        #####

        # df = pd.DataFrame.from_dict(len_dict_all, orient='index')
        # df.index.rename('Parameters', inplace=True)

        # stacked = df.stack().reset_index()
        # stacked.rename(columns={'level_1': 'Ant', 0: 'episode_lens'}, inplace=True)

        # sns.lineplot(data=stacked, x='Parameters', y='episode_lens')
        # #plt.xticks(rotation=30, ha='right')
        # plt.show()


    exit()

    PLOT_DIR= Path(f"/home/paulina/Documents/thesis/figures/experiments/forward/{settings}")
    #PLOT_DIR = f"PLOTS/{settings}/{condition}"
    run(RESULTS_DIR, PLOT_DIR)
