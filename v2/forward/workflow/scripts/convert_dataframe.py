import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from insect_rl.mdp.utils import grid_math
from icecream import ic


#actions = list(actions)
#ic(config, actions)

def convert(df, actions, reward, trap_cost, traps=False):

    df["state_int"] = df[["path_x", "path_y"]].apply(grid_math.point_to_int, args=(config['width'],), axis=1)
    df["next_state_int"] = df.state_int.shift(-1)
    df["next_x"] = df.path_x.shift(-1)
    df["next_y"] = df.path_y.shift(-1)

    df = df.dropna()
    df["next_state_int"] = df.next_state_int.astype('int32')
    df["next_x"] = df.next_x.astype('int32')
    df["next_y"] = df.next_y.astype('int32')

    # remove the steps that stay at the same point?
    df = df[df['state_int'] != df['next_state_int']]

    def get_action(row):
        return tuple(np.array([row.next_x, row.next_y]) - np.array([row.path_x, row.path_y]))


    def is_goal(row):
        return (row.next_x, row.next_y) in config['goals']

    def is_trap(row):
        return (row.next_x, row.next_y) in config['traps']

    new_dfs = []
    actions = list(actions)

    by_ant = df.groupby(["ant_nb", "trial_nb"])
    for (ant,trial), frame in by_ant:
        f = frame.drop(frame.tail(1).index)
        f.next_state_int = f.next_state_int.astype(int)
        f[["next_x", "next_y"]] = f[["next_x", "next_y"]].astype(int)
        f["action"] = f.apply(get_action, axis=1)
        f["action_int"] = f.action.apply(actions.index)
        f["reached_goal"] = f.apply(is_goal, axis=1)
        f["in_trap"] = f.apply(is_trap, axis=1)
        f["reward"] = 0.0
        f.reward = f.reward.where(~f.reached_goal, reward)
        if traps:
            f.reward = f.reward.where(~f.in_trap, trap_cost)

        f.drop(["path_x", "path_y", "next_x", "next_y"], axis=1, inplace=True)
        f.reset_index(drop=True, inplace=True)
        reached_goal_idx = f[f.reached_goal].iloc[0].name
        f = f[:reached_goal_idx + 1]

        new_dfs.append(f)

    df = pd.concat(new_dfs, ignore_index=True)

    return df


def compute_J(dataset, gamma=1.):
    """
    Compute the cumulative discounted reward of each episode in the dataset.

    Args:
        dataset (list): the dataset to consider;
        gamma (float, 1.): discount factor.

    Returns:
        The cumulative discounted reward of each episode in the dataset.

    """

    J_trials = []
    n_steps = []
    for trial in pd.unique(dataset["trial_nb"]):
        trial_data = dataset[dataset["trial_nb"] == trial]
        discounted = []
        
        i = len(trial_data.index)
        n_steps.append(len(trial_data.index))
        for step in reversed(trial_data.index):
            i -= 1
            J = gamma ** i * trial_data.reward[step]
            #print(i, step)
            discounted.append(J)
        J_trials.append(sum(discounted))
    return J_trials[0], n_steps



with open(snakemake.input.env[0], 'rb') as config_file:
    config = pickle.load(config_file)
actions = vars(grid_math)[snakemake.config['actions']]
reward = float(snakemake.wildcards.reward)
trap_cost = float(snakemake.wildcards.trapcost)

df_nt = pd.read_csv(snakemake.input[0])
df_nt = convert(df_nt, actions, reward, trap_cost)
df_nt = df_nt.reset_index(drop=True)
#df.to_csv(snakemake.output[0], index=False)

Js_no_trap = []
n_steps_no_trap = []
for ant in pd.unique(df_nt["ant_nb"]):
    ant_df = df_nt[df_nt["ant_nb"] == ant]
    J, n_steps = compute_J(ant_df, gamma=snakemake.config["simulation_settings"]["discount_factor"])
    Js_no_trap.append(J)
    n_steps_no_trap.append(n_steps)
#np.save(snakemake.output[0], Js_no_trap)
plt.plot(Js_no_trap)
plt.title("Cumulative discounted rewards without trap")
plt.savefig(snakemake.output[0])
plt.clf()

####### trap
df_t = pd.read_csv(snakemake.input[1])
df_t = convert(df_t, actions, reward, trap_cost, traps=True)
df_t = df_t.reset_index(drop=True)

Js_trap = []
n_steps_trap = []
for ant in pd.unique(df_t["ant_nb"]):
    ant_df = df_t[df_t["ant_nb"] == ant]
    J, n_steps = compute_J(ant_df, gamma=snakemake.config["simulation_settings"]["discount_factor"])
    Js_trap.append(J)
    n_steps_trap.append(n_steps)

plt.plot(Js_trap)
plt.title("Cumulative discounted rewards with trap")
plt.savefig(snakemake.output[1])
plt.clf()
#plt.show()

n_steps_no_trap = [np.mean(ant) for ant in n_steps_no_trap]
n_steps_trap = [np.mean(ant) for ant in n_steps_trap]
#plt.plot(n_steps_trap)
plt.title("Number of steps per trial")
plt.boxplot([n_steps_no_trap, n_steps_trap], labels=["no trap", "trap"])
#plt.show()
plt.savefig(snakemake.output[2])
