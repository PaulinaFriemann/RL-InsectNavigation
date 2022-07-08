from collections import Counter
import pickle
import functools
import itertools as it
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from mushroom_rl.environments.grid_world import GridWorld, AbstractGridWorld, GridWorldVanHasselt
from mushroom_rl.core import Core
from mushroom_rl.utils.callbacks import CollectDataset, CollectMaxQ, PlotDataset, CollectQ, CollectParameters

from insect_rl.mdp.mrl import MyGridWorld, IAVICallback, TDCallback
from insect_rl.mdp.utils import grid_math

from mushroom_rl.core import Core, Logger
import mushroom_rl.utils.dataset as mrl_dataset# compute_J, parse_dataset, compute_metrics, episodes_length
from icecream import ic
from tqdm import tqdm


def centroid(data):
    x, y = zip(*data)
    l = len(x)
    return int(round(sum(x) / l)), int(round(sum(y) / l))


def environment(configs, sim_settings):
    #configs.pop("traps")
    #configs.pop("trap_exits")
    goals = configs.pop("goals")
    # TODO WHYYY MUSHROOM_RL???
    width = configs["width"]
    configs["width"] = configs["height"]
    configs["height"] = width

    configs["goal"] = [(g[1], g[0]) for g in goals]
    # TODO only one possible
    configs["goal"] = centroid(goals)
    ic(configs)
    
    #configs["start"] = configs["start"]

    ic(sim_settings)

    #print("FYI I removed traps and trap exits")

    return MyGridWorld(**(configs | sim_settings))

def run_td_experiment(mdp, agent, i_agent, iterations, res_dir):
    collect_dataset = CollectDataset()
    iavi_dataconverter = IAVICallback(mdp, agent, statement="fit")
    tdcallback = TDCallback(mdp, agent)

    callbacks = [tdcallback]#collect_dataset, iavi_dataconverter]
    core = Core(agent, mdp)
    core.callbacks_fit = callbacks

    #len_batch = min(iterations, 10)

    #its = list(range(0, iterations, len_batch))
    tds = []#{s:[] for s in range(mdp._mdp_info.observation_space.n)}
    ss = []
    for i in tqdm(range(iterations)):
        tdcallback.clean()
        core.learn(n_episodes=1, n_steps_per_fit=1, quiet=True)
        ep_td, states = tdcallback.get()
        tds.append(ep_td)
        if i in [0,1,999]:
            ss.append(states)
    #fds.append()
    return tds, ss


def run_experiment(mdp, agent, i_agent, iterations, res_dir):
    collect_dataset = CollectDataset()
    iavi_dataconverter = IAVICallback(mdp, agent, statement="fit")
    tdcallback = TDCallback(mdp, agent)

    callbacks = [collect_dataset, iavi_dataconverter]
    core = Core(agent, mdp)
    core.callbacks_fit = callbacks

    len_batch = min(iterations, 10)

    its = list(range(0, iterations, len_batch))
    data = []
    Js = []
    episode_lens = []

    tds = np.zeros(iavi_dataconverter.cum_td.shape)
    tds_ns = np.zeros(iavi_dataconverter.cum_td.shape)
    for i in range(len(its)):
        print(f"batch {i}, {len(data)}")
        core.learn(n_episodes=len_batch, n_steps_per_fit=1, render=False, quiet=False)
        training_dataset = collect_dataset.get()
        data.extend(training_dataset)
        collect_dataset.clean()
        Js.append(compute_metrics(training_dataset, mdp.info.gamma))
        episode_lens.extend(mrl_dataset.episodes_length(training_dataset))

        for i in range(iavi_dataconverter.cum_td.shape[0]):
            # TODO maybe not sum but average?
            tds[i] += iavi_dataconverter.cum_td[np.array([i])]
            tds_ns[i] += iavi_dataconverter.cum_td_ns[np.array([i])]

    Js = list(it.chain.from_iterable(Js))
    plt.clf()
    plt.plot(list(range(len(Js))), Js, label='J')
    plt.title("Cumulative discounted reward")
    plt.legend(loc='best')
    plt.savefig(f"{res_dir}/J_{i_agent}.svg")
    with open(f"{res_dir}/J_{i_agent}.pickle", 'wb') as o:
        pickle.dump(Js, o)
    
    plt.plot(list(range(len(episode_lens))), episode_lens, label='episode length')
    plt.title("Episode lengths")
    plt.legend(loc='best')
    plt.savefig(f"{res_dir}/episode_lens_{i_agent}.svg")
    with open(f"{res_dir}/episode_lens_{i_agent}.pickle", 'wb') as o:
        pickle.dump(episode_lens, o)

    shape = iavi_dataconverter.V.shape
    v = np.zeros(shape)
    for i in range(shape[0]):
        v[i] = iavi_dataconverter.V[np.array([i])]

    np.save(f"{res_dir}/value_fun_{i_agent}.npy", np.rot90(v.reshape(mdp._height, mdp._width)))
    np.save(f"{res_dir}/tds_{i_agent}.npy", np.rot90(tds.reshape(mdp._height, mdp._width)))
    np.save(f"{res_dir}/tds_ns_{i_agent}.npy", np.rot90(tds_ns.reshape(mdp._height, mdp._width)))

    shape = agent.Q.shape
    q = np.zeros(shape)
    for i in range(shape[0]):
        for j in range(shape[1]):
            state = np.array([i])
            action = np.array([j])
            q[i, j] = agent.Q.predict(state, action)
    np.save(f"{res_dir}/q_{i_agent}.npy", q)

    with open(f"{res_dir}/data_{i_agent}.pickle", 'wb') as o:
        pickle.dump(data, o)
    df = convert_trajectories(data, mdp)
    df.to_csv(f"{res_dir}/df_{i_agent}.csv")
    agent.save(f"{res_dir}/agent_{i_agent}", full_save=True)


def _experiment(algorithm_class, mdp, ant, iterations, res_dir):
    print(f"ANT {ant}")
    res_dir += f"/a{ant}"
    os.mkdir(res_dir)
    agent = algorithm_class(mdp.info)
    # Algorithm
    collect_dataset = CollectDataset()
    iavi_dataconverter = IAVICallback(mdp, agent, statement="fit")

    callbacks = [collect_dataset, iavi_dataconverter] #collect_q
    core = Core(agent, mdp)

    len_batch = 10

    its = list(range(0, iterations, len_batch))
    core.callbacks_fit = callbacks
    data = []
    Js = []
    episode_lens = []

    tds = np.zeros(iavi_dataconverter.cum_td.shape)
    tds_ns = np.zeros(iavi_dataconverter.cum_td.shape)
    for i in range(len(its)):
        print(f"batch {i}, {len(data)}")
        core.learn(n_episodes=len_batch, n_steps_per_fit=1, quiet=False)
        training_dataset = collect_dataset.get()
        data.extend(training_dataset)
        collect_dataset.clean()
        Js.append(compute_metrics(training_dataset, mdp.info.gamma))
        episode_lens.extend(mrl_dataset.episodes_length(training_dataset))

        for i in range(iavi_dataconverter.cum_td.shape[0]):
            # TODO maybe not sum but average?
            tds[i] += iavi_dataconverter.cum_td[np.array([i])]
            tds_ns[i] += iavi_dataconverter.cum_td_ns[np.array([i])]

    Js = list(it.chain.from_iterable(Js))
    plt.clf()
    plt.plot(list(range(len(Js))), Js, label='J')
    plt.title("Cumulative discounted reward")
    plt.legend(loc='best')
    plt.savefig(f"{res_dir}/J_{ant}.svg")
    with open(f"{res_dir}/J_{ant}.pickle", 'wb') as o:
        pickle.dump(Js, o)
    
    plt.clf()
    plt.plot(list(range(len(episode_lens))), episode_lens, label='J')
    plt.title("Episode lengths")
    plt.legend(loc='best')
    plt.savefig(f"{res_dir}/episode_lens_{ant}.svg")
    with open(f"{res_dir}/episode_lens_{ant}.pickle", 'wb') as o:
        pickle.dump(episode_lens, o)

    shape = iavi_dataconverter.V.shape
    v = np.zeros(shape)
    for i in range(shape[0]):
        v[i] = iavi_dataconverter.V[np.array([i])]
    
    plt.clf()
    plt.imshow(np.rot90(v.reshape(mdp._height, mdp._width)))
    plt.title("Value function")
    plt.savefig(f"{res_dir}/value_fun_{ant}.svg")
    np.save(f"{res_dir}/value_fun_{ant}.npy", np.rot90(v.reshape(mdp._height, mdp._width)))
    
    plt.clf()
    plt.imshow(np.rot90(tds.reshape(mdp._height, mdp._width)))
    plt.title("Cumulative TD Error")
    plt.savefig(f"{res_dir}/tds_{ant}.svg")
    np.save(f"{res_dir}/tds_{ant}.npy", np.rot90(tds.reshape(mdp._height, mdp._width)))
    
    plt.clf()
    plt.imshow(np.rot90(tds_ns.reshape(mdp._height, mdp._width)))
    plt.title("Cumulative TD Error for the next state")
    plt.savefig(f"{res_dir}/tds_ns_{ant}.svg")
    np.save(f"{res_dir}/tds_ns_{ant}.npy", np.rot90(tds_ns.reshape(mdp._height, mdp._width)))
    plt.clf()

    shape = agent.Q.shape
    q = np.zeros(shape)
    for i in range(shape[0]):
        for j in range(shape[1]):
            state = np.array([i])
            action = np.array([j])
            q[i, j] = agent.Q.predict(state, action)
    np.save(f"{res_dir}/q_{ant}.npy", q)

    with open(f"{res_dir}/data_{ant}.pickle", 'wb') as o:
        pickle.dump(data, o)
    df = convert_trajectories(data, mdp)
    df.to_csv(f"{res_dir}/df_{ant}.csv")
    agent.save(f"{res_dir}/agent_{ant}", full_save=True)
    return


def convert_trajectories(data, mdp):
    episodes_lens = mrl_dataset.episodes_length(data)
    episodes = np.split(data, np.cumsum(episodes_lens)[:-1])
    dfs = []
    for i, episode in enumerate(episodes):
        state, action, reward, next_state, _, _ = mrl_dataset.parse_dataset(episode)

        to_grid = functools.partial(mdp.convert_to_grid, width=mdp._width)
        state = state.T[0]
        next_state = next_state.T[0]
        action = action.T[0]

        df = pd.DataFrame({"trial_nb":i, "state":state, "action":action, "reward":reward, "next_state":next_state})
        state_grid = df.state.apply(lambda s: to_grid([s])).tolist()
        df[['path_x', 'path_y']] = pd.DataFrame(state_grid, index=df.index)

        next_state_grid = df.next_state.apply(lambda s: to_grid([s])).tolist()
        df[['path_x_next', 'path_y_next']] = pd.DataFrame(next_state_grid, index=df.index)

        # conert everything but the rewards and J to int
        discrete = ["state", "next_state", "path_x", "path_y", "path_x_next", "path_y_next", "action"]
        df = df.astype({col: 'int32' for col in discrete})
        dfs.append(df)
    return pd.concat(dfs)


def compute_metrics(data, gamma):
    episode_lens = mrl_dataset.episodes_length(data)
    episodes = np.split(data, np.cumsum(episode_lens)[:-1])
    Js = list(it.chain.from_iterable([mrl_dataset.compute_J(e, gamma) for  e in episodes]))

    return Js


def experiment(algorithm_class, env_settings, n_ants, iterations, sim_settings):
    print("doing exp")
    np.random.seed()

    results_dir = f"RESULTS/RESULTSTRAP/r{sim_settings['reward']}tc{sim_settings['trap_cost']}aINTERg{sim_settings['gamma']}"

    try:
        # Create target Directory
        os.mkdir(results_dir)
        print("Directory " , results_dir ,  " Created ") 
    except FileExistsError:
        print("Directory " , results_dir ,  " already exists")
    
    mdp = environment(env_settings, sim_settings)

    for ant in tqdm(range(1,n_ants)):
        _experiment(algorithm_class, mdp, ant, iterations, results_dir)


if __name__=="__main__":
    with open("temp/agent.pickle", 'rb') as agent_i:
        agent = pickle.load(agent_i)
    with open("Wystrach2020/env.pickle", 'rb') as env_i:
        env = pickle.load(env_i)


    n_ants = 18#snakemake.config["ants"]
    iterations = 1000#snakemake.config["iterations"]

    sim_settings = {
        'reward':100,#eval(snakemake.wildcards["reward"]),
        'trap_cost':-100,#eval(snakemake.wildcards["trapcost"]),
        'actions':vars(grid_math)["INTERCARDINALS"],
        'gamma':.99#snakemake.config['simulation_settings']['discount_factor']
    }
    
    experiment(agent, env, n_ants, iterations, sim_settings)
