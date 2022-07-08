import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from mushroom_rl.utils import spaces
from mushroom_rl.environments.grid_world import GridWorld, AbstractGridWorld, GridWorldVanHasselt
from mushroom_rl.core import MDPInfo
from mushroom_rl.core import Core
from mushroom_rl.utils.callbacks import CollectDataset, CollectMaxQ

from insect_rl.mdp.mrl import MyGridWorld, IAVICallback
from insect_rl.mdp.utils import grid_math

from mushroom_rl.policy import EpsGreedy, Boltzmann
from mushroom_rl.core import Core
from mushroom_rl.utils.parameters import Parameter, ExponentialParameter
from mushroom_rl.utils.dataset import compute_J, parse_dataset, compute_metrics
from mushroom_rl.algorithms.value import (
    QLearning,
    DoubleQLearning,
    WeightedQLearning,
    SpeedyQLearning,
    SARSA,
)


def environment(configs):
    return GridWorld(**configs)
    return MyGridWorld(width=4, height=4, goal=(2, 3), reward=reward,
                  start=(0, 0), traps = [(1,3)], trap_cost=trap_cost, actions=actions)


def get_env_configs():
    return {
        "width": 20,
        "height": 20,
        "goal": (15,12),
        "start": (0,0)
    }


def experiment(algorithm_class, exp):
    np.random.seed()

    # MDP
    mdp = environment(get_env_configs())

    # Policy
    epsilon = Parameter(value=.3)
    pi = EpsGreedy(epsilon=epsilon)

    # Agent
    learning_rate = ExponentialParameter(value=1, exp=exp, size=mdp.info.size)
    algorithm_params = dict(learning_rate=learning_rate)
    agent = algorithm_class(mdp.info, pi, **algorithm_params)

    # Algorithm
    start = mdp.convert_to_int(mdp._start, mdp._width)
    collect_max_Q = CollectMaxQ(agent.Q, start)
    collect_dataset = CollectDataset()
    #iavi_dataconverter = IAVICallback(mdp)

    callbacks = [collect_dataset, collect_max_Q]
    core = Core(agent, mdp, callbacks)

    # Train
    core.learn(n_episodes=1000, n_steps_per_fit=1, quiet=True)

    #states, actions, reward, _, _, _ = parse_dataset(collect_dataset.get())
    #max_Qs = collect_max_Q.get()

    data = collect_dataset.get()

    shape = agent.Q.shape
    q = np.zeros(shape)
    for i in range(shape[0]):
        for j in range(shape[1]):
            state = np.array([i])
            action = np.array([j])
            q[i, j] = agent.Q.predict(state, action)

    # state, action, reward, next_state, absorbing, last

    return data,\
           collect_max_Q.get(),\
           q,\
           compute_J(data, gamma=mdp.info.gamma),\
           compute_metrics(data, gamma=mdp.info.gamma)

           #iavi_dataconverter.get()


result = experiment(SARSA, .8)

data = result[0]
max_Qs = result[1]
q = result[2]
J = result[3] # cumulative discounted reward of each episode in the dataset
#iavi_data = result[2]
min_J, max_J, mean_J, len_J = result[4]

df = pd.DataFrame(data, columns=["state_int", "action_int", "reward", "next_state_int", "absorbing", "last"])
df.state_int = df.state_int.astype(int)
df.action_int = df.action_int.astype(int)
df.next_state_int = df.next_state_int.astype(int)
df.to_csv(snakemake.output[0], index=False)

np.save(snakemake.output[1], max_Qs)
np.save(snakemake.output[2], q)
np.save(snakemake.output[3], J)
exit()

#plt.subplot(2, 1, 1)
#print(J)
r = np.convolve(df.reward.to_numpy(), np.ones(1000) / 1000., 'valid')
print(min_J, max_J, mean_J, len_J)
plt.plot(r)
plt.show()
exit()
r = df.reward.to_numpy()
r = np.convolve(r, np.ones(100) / 100., 'valid')
print(r)
x = np.arange(0,len(r))
print(x.shape)
print(len(r))
print(r.shape)
plt.plot(x, r)
#plt.subplot(2, 1, 2)
#plt.plot(max_Qs)
plt.show()
exit()




exit()

#import viewer
reward = snakemake.config["simulation_settings"]["rewards"]
actions = grid_math.__dict__[snakemake.config["actions"]]
trap_cost = snakemake.config["simulation_settings"]["trapcosts"]
wind = snakemake.config["simulation_settings"]["winds"]

# width: int, height : int, goal : tuple, reward: float, start : tuple, actions: tuple[Action]=INTERCARDINALS, traps=None, trap_cost=-10, **kwargs):
width = 5
height = 5

epsilon = Parameter(value=wind)
policy = Boltzmann(beta=0.0) #EpsGreedy(epsilon=epsilon)  # 
learning_rate = Parameter(value=0.6)
ant = SARSA(env.info, policy, learning_rate)

iavi_dataconverter = IAVICallback(env)
collect_dataset = CollectDataset()
callbacks = [collect_dataset]

iterations = 10000#snakemake.config["iterations"]

core = Core(ant, env, callbacks_fit=[], callback_step=iavi_dataconverter, preprocessors=[])
core.learn(n_episodes=iterations, n_steps_per_fit=1, render=False)
print("\n\n\n-----------------------")
#initial_states=initial_states
dataset = core.evaluate(n_episodes=100, render=True)

shape = ant.Q.shape
q = np.zeros(shape)
for i in range(shape[0]):
    for j in range(shape[1]):
        state = np.array([i])
        action = np.array([j])
        q[i, j] = ant.Q.predict(state, action)
#print(q)

print()
print(compute_J(dataset, gamma=env.info.gamma))
print([env.convert_to_grid(s[0], width) for s in dataset])
