import sys
import os
import pickle
from dataclasses import asdict
import pandas as pd
import numpy as np

from insect_rl.simulation import Simulation, EnvironmentConfig, AgentConfig, CustomGridworld
from insect_rl.mdp.utils import grid_math



def run(df, transition_probs, envc, actions):

    _agentc = AgentConfig(actions=actions)
    env = CustomGridworld(
        **asdict(envc),
        **asdict(_agentc),
        transition_probability=transition_probs
    )

    df["reward"] = df.next_state_int.apply(env.reward)
    by_ant_trial = df.groupby(["ant_nb", "trial_nb"], sort=False)
    trajectories = []
    for _, frame in by_ant_trial:
        trajectories.append(frame[["state_int", "action_int", "reward", "next_state_int"]].to_numpy())
    
    action_probs = grid_math.get_action_probabilities(trajectories, env.n_states, env.n_actions)
    return action_probs, env.ground_r


if __name__ == "__main__":
    df = pd.read_csv(snakemake.input[0]) # converted dataframe (state int, action int, reward, next state int)

    settings = snakemake.params["settings"]
    with open(snakemake.input[1], 'rb') as envc_file:
        envc = pickle.load(envc_file)

    transition_probs = np.load(snakemake.input[2])

    action_probs, ground_r = run(
        df,
        transition_probs,
        envc = envc,
        actions = vars(grid_math)[snakemake.config['actions']],
    )

    np.save(snakemake.output[0], action_probs)
    np.save(snakemake.output[1], ground_r)
