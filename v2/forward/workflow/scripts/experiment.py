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

from insect_rl.mdp.mrl import MyGridWorld, IAVICallback
from insect_rl.mdp.utils import grid_math

from mushroom_rl.core import Core, Logger
import mushroom_rl.utils.dataset as mrl_dataset# compute_J, parse_dataset, compute_metrics, episodes_length
from icecream import ic
from tqdm import tqdm


def centroid(data):
    x, y = zip(*data)
    l = len(x)
    return sum(x) / l, sum(y) / l


def environment(configs, sim_settings):
    configs.pop("traps")
    configs.pop("trap_exits")
    goals = configs.pop("goals")
    # TODO WHYYY MUSHROOM_RL???
    width = configs["width"]
    configs["width"] = configs["height"]
    configs["height"] = width
    print(goals)

    configs["goal"] = [(g[1], g[0]) for g in goals]
    ic(centroid(goals))
    # TODO only one possible
    centr = centroid(goals)
    configs["goal"] = (int(round(centr[0])), int(round(centr[1])))
    ic(configs)
    
    configs["start"] = (configs["start"][1], configs["start"][0])


    ic(sim_settings)

    print("FYI I removed traps and trap exits")
    #return GridWorld(**(configs | sim_settings))
    #return GridWorld(**configs)

    return MyGridWorld(**(configs | sim_settings))


def log_data(logger, name, dataset, gamma, save=False):
    #logger.weak_line()
    #logger.info(name)
    J = mrl_dataset.compute_J(dataset, gamma)  # Discounted returns
    n_steps = mrl_dataset.episodes_length(dataset)
    min_score, max_score, mean_score, num = mrl_dataset.compute_metrics(dataset, gamma)

    logger.epoch_info(name, J=J, steps=n_steps, min_score=min_score, max_score=max_score, mean_score=mean_score, num=num)
    
    if save:
        logger.log_numpy(**{f"J_{name}":J})


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def _experiment(algorithm_class, mdp, iterations, save_last=0):
    agent = algorithm_class(mdp.info)
    # Algorithm
    collect_dataset = CollectDataset()
    collect_q= CollectQ(agent.Q)
    iavi_dataconverter = IAVICallback(mdp, agent, statement="fit")

    callbacks = [collect_dataset, collect_q, iavi_dataconverter]
    core = Core(agent, mdp)


    # Evaluation
    #dataset = core.evaluate(n_episodes=10)
    #log_data(logger, f"before {ant}", dataset, mdp.info.gamma)
    save_last = 0
    len_batch = 10

    if save_last==0:
        its = list(range(0, iterations-save_last, len_batch))
        core.callbacks_fit = callbacks
        #data = []
        Js = []
        episode_lens = []

        tds = np.zeros(iavi_dataconverter.cum_td.shape)
        tds_ns = np.zeros(iavi_dataconverter.cum_td.shape)
        for _ in range(len(its)):
            core.learn(n_episodes=len_batch, n_steps_per_fit=1, quiet=False)
            training_dataset = collect_dataset.get()
            collect_dataset.clean()
            Js.append(compute_metrics(training_dataset, mdp.info.gamma))
            episode_lens.extend(mrl_dataset.episodes_length(training_dataset))

            for i in range(iavi_dataconverter.cum_td.shape[0]):
                tds[i] += iavi_dataconverter.cum_td[np.array([i])]
                tds_ns[i] += iavi_dataconverter.cum_td_ns[np.array([i])]

        Js = list(it.chain.from_iterable(Js))

        plt.plot(list(range(len(Js))), Js, label='J')
        plt.title("Cumulative discounted reward")
        plt.legend(loc='best')
        plt.show()

        shape = iavi_dataconverter.V.shape
        v = np.zeros(shape)
        for i in range(shape[0]):
            v[i] = iavi_dataconverter.V[np.array([i])]
        
        plt.clf()
        plt.imshow(np.rot90(v.reshape(mdp._height, mdp._width)))
        plt.title("Value function")
        plt.show()
        plt.clf()
        plt.imshow(np.rot90(tds.reshape(mdp._height, mdp._width)))
        plt.tilte("Cumulative TD Error")
        plt.show()
        plt.clf()
        plt.imshow(np.rot90(tds_ns.reshape(mdp._height, mdp._width)))
        plt.title("Cumulative TD Error for the next state")
        plt.show()


        exit()
    
    if save_last > 0:
        its = list(range(0, iterations-save_last, len_batch))
        
        for _ in range(len(its)):
            # Train batch
            states = [tuple(np.array(mdp._goal) - np.array(action)) for action in mdp.action_space.actions]
            states = [mdp.convert_to_int(s, mdp._width) for s in states]
            core.learn(n_episodes=len_batch, n_steps_per_fit=1, quiet=True)
            ic([agent.policy(s) for s in states])

        # Train the recorded batch
        core.callbacks_fit = callbacks
        core.learn(n_episodes=save_last, n_steps_per_fit=1, quiet=True)
    
        training_dataset = collect_dataset.get()

    qs = collect_q.get()[-1]

    V = iavi_dataconverter.V
    shape = V.shape
    v = np.zeros(shape)
    for i in range(shape[0]):
        v[i] = V[np.array([i])]

    tds, tds_ns = iavi_dataconverter.get_td_errors()
    td_errors_av = iavi_dataconverter.get_td_errors_action_value()

    #log_data(logger, ant, training_dataset, mdp.info.gamma, save=True)

    #dataset = core.evaluate(n_episodes=1, render=True)
    #log_data(logger, f"after {ant}", dataset, mdp.info.gamma)
    
    return training_dataset, tds, tds_ns,\
        td_errors_av, qs, v


def convert_trajectories(data, mdp):
    episodes_lens = mrl_dataset.episodes_length(data)
    episodes = np.split(data, np.cumsum(episodes_lens)[:-1])
    #J = mrl_dataset.compute_J(data, mdp.info.gamma)
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
        
        #df.drop("state", axis="columns", inplace=True)
        #df["J"] = J[i]

        # conert everything but the rewards and J to int
        discrete = ["state", "next_state", "path_x", "path_y", "path_x_next", "path_y_next", "action"]
        df = df.astype({col: 'int32' for col in discrete})
        dfs.append(df)
    return pd.concat(dfs)


def compute_metrics(data, gamma):
    episode_lens = mrl_dataset.episodes_length(data)
    ic(episode_lens)

    episodes = np.split(data, np.cumsum(episode_lens)[:-1])
    Js = list(it.chain.from_iterable([mrl_dataset.compute_J(e, gamma) for  e in episodes])) # list of lists
    #min_score, max_score, mean_score, _ = zip(*[mrl_dataset.compute_metrics(e, gamma) for e in episodes])

    return Js#, list(min_score), list(max_score), list(mean_score)

def experiment(algorithm_class, env_settings, n_ants, iterations, sim_settings):
    print("doing exp")
    np.random.seed()
    
    mdp = environment(env_settings, sim_settings)

    dfs = []
    qss = []
    vss = []
    Jss = []
    min_scores = []
    max_scores = []
    mean_scores = []
    for ant in tqdm(range(n_ants)):
        dataset, tds, tds_s_ns, tds_av, qs, vs = _experiment(algorithm_class, mdp, iterations, save_last=20)
        #ic(tds, type(tds))
        # df = convert_trajectories(dataset, mdp)
        # df.insert(0, "ant_nb", ant)
        # df["td"] = tds
        # df["td_s_ns"] = tds_s_ns
        # df["td_error_q"] = tds_av
        # dfs.append(df)
        qss.append(qs)
        vss.append(vs)
        Js, min_score, max_score, mean_score = compute_metrics(dataset, mdp.info.gamma)
        Jss.append(Js)
        min_scores.append(min_score)
        max_scores.append(max_score)
        mean_scores.append(mean_score)

        print("plot shit")
        fig, ax = plt.subplots(figsize=(5, 2.7), layout='constrained')
        x = list(range(len(Js)))
        ax.plot(x, min_score, label='min')
        ax.plot(x, max_score, label='max')
        ax.plot(x, mean_score, label='mean')
        plt.legend(loc='best')
        plt.show()
        plt.clf()
        plt.plot(x, Js, label='J')
        plt.legend(loc='best')
        plt.show()

        exit()
    df = pd.concat(dfs)
    #ic(df)
    return df, qs, vss, Js


if __name__=="__main__":
    with open(snakemake.input[0], 'rb') as agent_i:
        agent = pickle.load(agent_i)
    with open(snakemake.input[1], 'rb') as env_i:
        env = pickle.load(env_i)

    n_ants = snakemake.config["ants"]
    iterations = snakemake.config["iterations"]

    sim_settings = {
        'reward':eval(snakemake.wildcards["reward"]),
        'trap_cost':eval(snakemake.wildcards["trapcost"]),
        'actions':vars(grid_math)[snakemake.config['actions']],
        'gamma':snakemake.config['simulation_settings']['discount_factor']
    }
    
    df, qss, vss, Js = experiment(agent, env, n_ants, iterations, sim_settings)
    df.to_csv(snakemake.output["results"], index=False)
    np.save(snakemake.output["Qs"], np.array(qss))
    np.save(snakemake.output["Vs"], np.array(vss))
    np.save(snakemake.output["Js"], np.array(Js))
