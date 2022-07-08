import pickle
import pandas as pd
import numpy as np

from insect_rl.mdp.utils import grid_math


df = pd.read_csv(snakemake.input[0])

with open(snakemake.input[1], 'rb') as config_file:
    config = pickle.load(config_file)

actions = vars(grid_math)[snakemake.config['actions']]

df["state_int"] = df[["path_x", "path_y"]].apply(grid_math.point_to_int, args=(config['width'],), axis=1)
df["next_state_int"] = df.state_int.shift(-1)
df["next_x"] = df.path_x.shift(-1)
df["next_y"] = df.path_y.shift(-1)


def get_action(row):
    return tuple(np.array([row.next_x, row.next_y]) - np.array([row.path_x, row.path_y]))


def is_goal(row):
    return (row.next_x, row.next_y) in config['goals']


new_dfs = []

by_ant = df.groupby(["ant_nb"])
for ant, frame in by_ant:
    f = frame[frame["trial_nb"]==frame.max().trial_nb]
    f = f.drop(f.tail(1).index)

    f.next_state_int = f.next_state_int.astype(int)
    f[["next_x", "next_y"]] = f[["next_x", "next_y"]].astype(int)
    f["action"] = f.apply(get_action, axis=1)
    f["action_int"] = f.action.apply(actions.index)
    f["reached_goal"] = f.apply(is_goal, axis=1)
    f.drop(["path_x", "path_y", "next_x", "next_y"], axis=1, inplace=True)
    f.reset_index(drop=True, inplace=True)
    #reached_goal_idx = f[f.reached_goal].iloc[0].name
    #f = f[:reached_goal_idx + 1]
    
    new_dfs.append(f)

# by_ant_trial = df.groupby(["ant_nb", "trial_nb"])
# for ant_trial, frame in by_ant_trial:

#     #f = frame[frame["trial_nb"]==frame.max().trial_nb]
#     f = frame.drop(frame.tail(1).index)

#     f.next_state_int = f.next_state_int.astype(int)
#     f[["next_x", "next_y"]] = f[["next_x", "next_y"]].astype(int)
#     f["action"] = f.apply(get_action, axis=1)
#     f["action_int"] = f.action.apply(actions.index)
#     f["reached_goal"] = f.apply(is_goal, axis=1)
#     f.drop(["path_x", "path_y", "next_x", "next_y"], axis=1, inplace=True)
#     f.reset_index(drop=True, inplace=True)
#     #reached_goal_idx = f[f.reached_goal].iloc[0].name
#     #f = f[:reached_goal_idx + 1]
    
#     new_dfs.append(f)

df = pd.concat(new_dfs, ignore_index=True)
df.to_csv(snakemake.output[0], index=False)


# by_ant = df.groupby(["ant_nb", "trial_nb"])
# for (ant,trial), frame in by_ant:
#     f = frame.drop(frame.tail(1).index)
#     f.next_state_int = f.next_state_int.astype(int)
#     f[["next_x", "next_y"]] = f[["next_x", "next_y"]].astype(int)
#     f["action"] = f.apply(get_action, axis=1)
#     f["action_int"] = f.action.apply(actions.index)
#     f["reached_goal"] = f.apply(is_goal, axis=1)
#     f.drop(["path_x", "path_y", "next_x", "next_y"], axis=1, inplace=True)
#     f.reset_index(drop=True, inplace=True)
#     reached_goal_idx = f[f.reached_goal].iloc[0].name
#     f = f[:reached_goal_idx + 1]
    
#     new_dfs.append(f)

# df = pd.concat(new_dfs, ignore_index=True)
# df.to_csv(snakemake.output[0], index=False)
