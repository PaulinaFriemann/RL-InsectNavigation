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





def plot_line(dir_, ids, data_name, title, xlabel, ylabel):
  SMOOTHING_FACTOR = 10

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

  sns.lineplot(data=df_data, x="x", y="y", ci='sd')
  ax = plt.gca()
  #ax.set_ylim(df_J.y.min(),df_J.y.max() +2)
  ax.set_xlabel(xlabel)
  ax.set_ylabel(ylabel)
  #ax.set_title(title)


def plot_condition(condition, dir_, plot_dir, env, traps, trap_exits):
    AGENT_NUMBER = 7#len(os.listdir(dir_))

    agent_ids = [i for i in range(0,AGENT_NUMBER)]
    #sample_ids = random.sample(agent_ids, min(AGENT_NUMBER, 5))

    SAVE_PLOTS = True

    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
        print("Directory " , plot_dir ,  " Created ")
    else:    
        print("Directory " , plot_dir ,  " already exists")
    plt.clf()
    plot_line(dir_, agent_ids, data_name='J', title=TITLES[condition] , xlabel='episodes', ylabel='cum. discounted rewards   G')

    if SAVE_PLOTS:
        plt.savefig(f"{plot_dir}/J.svg",bbox_inches='tight')
    plt.clf()

    # plot J for all agents
    for i in agent_ids:
        plot_line(dir_, [i], data_name='J', title=TITLES[condition] , xlabel='episodes', ylabel='cum. discounted rewards   G')
        if SAVE_PLOTS:
            plt.savefig(f"{plot_dir}/J_{i}.svg",bbox_inches='tight')
        plt.clf()


    plot_line(dir_, agent_ids, data_name='episode_lens', title=TITLES[condition] , xlabel='episodes', ylabel='average number of steps per episode')

    if SAVE_PLOTS:
        plt.savefig(f"{plot_dir}/episode_lens.svg",bbox_inches='tight')

    samples = []
    for idx, i in enumerate(agent_ids):
        df = pd.read_csv(f"{dir_}/a{i}/df_{i}.csv")
        samples.append(df)


    last_trials = [ant_df[ant_df.trial_nb == ant_df.trial_nb.max()] for ant_df in samples]
    #last_trials[0]

    plt.clf()

    ax = plt.subplot()
    ax.set_xlim(0,env["width"])
    ax.set_ylim(0,env["height"])
    ax.set_aspect('equal')
    ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))
    ax.set_yticks([])
    #ax.yaxis.set_major_locator(ticker.MultipleLocator(5))
    #ax.yaxis.set_minor_locator(ticker.MultipleLocator(1))

    #ax.grid(which='both')

    traps = env["traps"]
    trap_exits = env["trap_exits"]

    patches = [
        Rectangle(env["goal"], width=1, height=1, color='red'),
        Rectangle(env["start"], width=1, height=1, color='green'),
        *(Rectangle(trap, width=1, height=1, color='gray') for trap in traps),
        Rectangle(trap_exits[0][0], width=.5, height=2, color='gray'),
        Rectangle(trap_exits[1][0], width=.5, height=2, color='gray')
    ]

    for patch in patches:
        ax.add_patch(patch)

    ax.annotate('Feeder', (env['start'][0] + 2, env['start'][1]))
    ax.annotate('Nest', (env['goal'][0] + 2, env['goal'][1]))
    if SAVE_PLOTS:
        plt.savefig(f"{plot_dir}/environment.svg",bbox_inches='tight')
        #plt.savefig(f"{PLOT_DIR}/environment_trap_squiqqly.svg")
    plt.clf()


    ax = plt.subplot()
    ax.set_xlim(0,env["width"])
    ax.set_ylim(0,env["height"])
    ax.set_aspect('equal')

    #traps = env["traps"]
    #trap_exits = env["trap_exits"]

    patches = [
        
        Rectangle(env["goal"], width=1, height=1, color='red'),
        #Rectangle((1,11), width=1, height=1, color='green')    
        Rectangle(env["start"], width=1, height=1, color='green') # wrong in simulation
        #Rectangle(trap_exits[0][0], width=.5, height=2, color='black'),
        #Rectangle(trap_exits[1][0], width=.5, height=2, color='black')
    ]

    patches.extend([Rectangle(trap, width=1, height=1, color='lightgray') for trap in traps])

    for patch in patches:
        ax.add_patch(patch)

    #ax.set_title(TITLES[condition])
    for ant_df in last_trials:
        plots.draw_trajectories_phil(ant_df, ax)
    ax.annotate('Feeder', (env['start'][0] + 2, env['start'][1]))
    ax.annotate('Nest', (env['goal'][0] + 2, env['goal'][1]))
    ax.set_yticks([])
    ax.set_xticks([])
    print("MANUALLY CHANGES CODE HERE TO ACCOUNT FOR SIMULATION ERROR OF WRONG STARTING POINT")
    if SAVE_PLOTS:
        plt.savefig(f"{plot_dir}/all_last_trajectories.svg",bbox_inches='tight')

    plt.clf()
    fig, ax = plt.subplots(nrows=1, ncols=len(agent_ids), sharey=True) # sharey funktioniert nicht, frag mich nicht wieso - alles mögliche probiert
    #figsize=(45, 5),
    for idx, i in enumerate(agent_ids):
        df = pd.read_csv(f"{dir_}/a{i}/df_{i}.csv")
        df = df[df.trial_nb == df.trial_nb.max()]

        if idx == 0:
            ax = plt.subplot(1,len(agent_ids),idx+1)
            ax.set_yticks([])  # Diesen Befehl benutzen um alle Axen zu entfernen
            ax.set_xticks([])  # Diesen Befehl benutzen um alle Axen zu entfernen
            ax1 = ax
        else:
            ax = plt.subplot(1,len(agent_ids),idx+1)
            ax.set_yticks([])  # Diesen Befehl benutzen um alle Axen zu entfernen
            ax.set_xticks([])  # Diesen Befehl benutzen um alle Axen zu entfernen
        
        ax.set_xlim(0,env["width"])
        ax.set_ylim(0,env["height"])
        ax.set_aspect('equal')


        patches = [
        Rectangle(env["goal"], width=1, height=1, color='red'),
        Rectangle(env["start"], width=1, height=1, color='green')    
        ]
        if condition=="1-trap":
            patches.extend([Rectangle(trap, width=1, height=1, color='lightgray') for trap in traps])
        else:
            patches.extend([Rectangle(trap, width=1, height=1, color='lightgray') for trap in traps[::2]])
        for patch in patches:
            ax.add_patch(patch)

        plots.draw_trajectories_phil(df, ax)
        ax.annotate('Feeder', (env['start'][0] + 2, env['start'][1]))
        ax.annotate('Nest', (env['goal'][0] + 2, env['goal'][1]))

    #plt.suptitle(TITLES[condition])
    if SAVE_PLOTS:
        plt.savefig(f"{plot_dir}/last_trajectories_sample.svg",bbox_inches='tight')
    plt.clf()

    tds_ns_list = []
    for i in agent_ids:
        with open(f"{dir_}/a{i}/tds_ns_{i}.npy",'rb') as f_in:
            tds_ns_list.append(np.load(f_in))
    # creating mean over all agents
    all_agent_mean_tds_ns = np.array(tds_ns_list).mean(axis=0)

    plt.figure(figsize=(10,10))
    plt.imshow(all_agent_mean_tds_ns[::-1], origin='lower', cmap="Greys")#, norm=LogNorm(vmin=0.01, vmax=all_agent_mean_tds_ns.max()/2))
    #plt.title("Mean TD error over all agents and all trials")
    #plt.title(f"mean cumulative TD error over all trials - {TITLES[condition]}")
    plt.colorbar()
    if SAVE_PLOTS:
        plt.savefig(f"{plot_dir}/td_error_allagents_alltrialsnormal.svg",bbox_inches='tight')
    plt.clf()

    trial_samples = [1,2,5,250, 500, 750, 1000]
    fig, axs = plt.subplots(nrows=1, ncols=len(trial_samples), sharey=True)#, figsize=(25, 7))
    #cs = iter(cm.rainbow(np.linspace(0, 1, len(trial_samples))))

    for df_ant in samples:
        # per ant
        for idx,trial_sample in enumerate(trial_samples):

            df_subsample = df_ant[df_ant.trial_nb == trial_sample-1]
            ax = axs[idx]#plt.subplot(1,len(trial_samples),idx+1)
            plt.sca(ax)
            plots.draw_trajectories_phil(df_subsample, ax)

            ax.set_xlim(0,env["width"])
            ax.set_ylim(0,env["height"])
            ax.set_aspect('equal')
            ax.set_title(f"Episode {trial_sample}",fontsize=7)
            ax.set_xticks([])
            ax.set_yticks([])

            patches = [
            Rectangle(env["goal"], width=1, height=1, color='red'),
            Rectangle(env["start"], width=1, height=1, color='green')    
            ]
            if condition=="1-trap":
                patches.extend([Rectangle(trap, width=1, height=1, color='lightgray') for trap in traps])
            else:
                patches.extend([Rectangle(trap, width=1, height=1, color='lightgray') for trap in traps[::2]])

            for patch in patches:
                ax.add_patch(patch)    
            ax.annotate('F', (env['start'][0] + 2, env['start'][1]))
            ax.annotate('N', (env['goal'][0] + 2, env['goal'][1]))
    #plt.tight_layout()
    #plt.suptitle(TITLES[condition])

    if SAVE_PLOTS:
        plt.savefig(f"{plot_dir}/Traj_all_trials.svg",bbox_inches='tight')

    plt.clf()

    # prep heatmap
    max_x = env["height"]
    max_y = env["width"]

    #sub_sample = [samples[i] for i in agent_ids]
    #trial_samples = [1, 100, 200, 300, 400, 500, 600, 700, 800, 1000]
    fig, axs = plt.subplots(nrows=1, ncols=len(trial_samples), sharey=True)#, figsize=(35, 7))
    #cs = iter(cm.rainbow(np.linspace(0, 1, len(trial_samples))))

    for idx,trial_sample in enumerate(trial_samples):
        # each ant's path is drawn into all plots
        ax = axs[idx]
        plt.sca(ax)
        sur_map = np.zeros((max_x, max_y))
        for ix, df_ant in enumerate(samples):
            df_subsample = df_ant[df_ant.trial_nb == trial_sample-1]

            #ax = plt.subplot(len(sub_sample),len(trial_samples),ix*10+idx+1)
            ax.set_title(f"Episode {trial_samples[idx]}",fontsize=7)
            #ax.set_xlabel(f"Ant #{agent_ids[ix] + 1}")
            ax.set_xticks([])
            ax.set_yticks([])


            traj = df_subsample
            y_vals = list(traj['path_x'])
            x_vals = list(traj['path_y'])

            for i in range(0, len(x_vals)):
                sur_map[x_vals[i]][y_vals[i]] += 1

            plt.imshow(sur_map[::-1], origin='upper', cmap="gnuplot")
    #plt.suptitle(TITLES[condition])
    if SAVE_PLOTS:  
        plt.savefig(f"{plot_dir}/Heatmap-Ants_Sampled-trials.svg",bbox_inches='tight')
    
    plt.clf()
    
    value_list = []
    for i in agent_ids:
        with open(f"{dir_}/a{i}/value_fun_{i}.npy",'rb') as f_in:
            value_list.append(np.load(f_in))
    # creating mean over all agents
    all_agent_mean_value = np.array(value_list).mean(axis=0)

    plt.figure(figsize=(10,10))
    plt.imshow(all_agent_mean_value[::-1], origin='lower', norm=SymLogNorm(100, vmin=-200, vmax=200)) # cmap="gnuplot",

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
    clb = plt.colorbar()#ticks=list(range(-200,200,10)))
    clb.ax.set_title('V(s)')
    #clb.set_ticks([round(t,1) for t in clb.get_ticks()])

    if SAVE_PLOTS:
        plt.savefig(f"{plot_dir}/Value_Func_Mean.svg",bbox_inches='tight')
    plt.clf()

    # Q-Values
    i = 0
    with open(f"{dir_}/a{i}/q_{i}.npy",'rb') as f_in:
        qs = np.load(f_in)

    def max_direction(state):
        if all([math.isclose(a,state[0]) for a in state]):
            return np.array((0.0,0.0))
        #print(state)
        max_dir = np.array(INTERCARDINALS[np.argmax(state)])
        return (max_dir/np.linalg.norm(max_dir)) * np.max(state)

    qs = np.apply_along_axis(max_direction, 1, qs)
    qs_resh = np.reshape(qs, (60,23,2))
    U_quiv = qs_resh[:,:,0]
    V_quiv = qs_resh[:,:,1]


    fig, ax = plt.subplots()
    quiv = ax.quiver(U_quiv, V_quiv, pivot='mid')

    ax.set_aspect('equal')
    patches = [
        Rectangle(env["goal"], width=1, height=1, color='red'),
        Rectangle(env["start"], width=1, height=1, color='green')    
    ]
    if condition=="1-trap":
        patches.extend([Rectangle(trap, width=1, height=1, color='lightgray') for trap in traps])
    else:
        patches.extend([Rectangle(trap, width=1, height=1, color='lightgray') for trap in traps[::2]])

    for patch in patches:
        ax.add_patch(patch)    
    ax.annotate('F', (env['start'][0] + 2, env['start'][1]))
    ax.annotate('N', (env['goal'][0] + 2, env['goal'][1]))
    if SAVE_PLOTS:
        plt.savefig(f"{plot_dir}/Qs_0.svg",bbox_inches='tight')
    plt.clf()


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
    REWARD = int(sys.argv[1])# 10
    TRAP_COST = int(sys.argv[2])#  -100
    LR = float(sys.argv[3])
    MC = sys.argv[4]
    settings = f"{MC}/lr{LR}/r{REWARD}tc{TRAP_COST}aINTERg0.99"

    RESULTS_DIR = Path(f"RESULTS/FULLEXPERIMENT/{settings}")

    PLOT_DIR= Path(f"/home/paulina/Documents/thesis/figures/experiments/forward/{settings}")
    #PLOT_DIR = f"PLOTS/{settings}/{condition}"
    run(RESULTS_DIR, PLOT_DIR)
