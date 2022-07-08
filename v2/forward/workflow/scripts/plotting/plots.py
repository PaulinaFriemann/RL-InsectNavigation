from functools import partial
import pickle
from typing import Callable

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.patches import Circle

from icecream import ic


def plot_J():
    print("PLOT J!")


def draw_trajectories_phil(df, ax, colors=None):

    plot_funs = [
                partial(ax.plot, 'path_x', 'path_y'),
                #partial(plt.plot, 'path_x', 'path_y', 'o')
            ]
    runs = pd.unique(df["trial_nb"])

    if colors is None:
        colors = ['k']*len(runs)

    #plt.sca(ax)
    for run, color in zip(runs, colors):

        traj = df[df.trial_nb == run]
        final_x = int(traj['path_x_next'].tail(1))
        final_y = int(traj['path_y_next'].tail(1))
        
        final_step = {'trial_nb	': 0, 'state': 0, 'action': 0,	'reward': 0,	'next_state': 0,	'path_x': final_x	,'path_y': final_y,	'path_x_next': 0, 'path_y_next': 0}
        traj = traj.append(final_step, ignore_index=True)
        for fun in plot_funs:
            fun(data=traj, label=run, c=color)
    
    #ax.axis('equal')
    #ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
    #ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))
    
    #ax.yaxis.set_major_locator(ticker.MultipleLocator(5))
    #ax.yaxis.set_minor_locator(ticker.MultipleLocator(1))


def draw_trajectories(df, plot_funs: Callable, ax, colors=None):

    runs = pd.unique(df["trial_nb"])

    if colors is None:
        colors = ['k']*len(runs)

    plt.sca(ax)

    for run, color in zip(runs, colors):

        traj = df[df.trial_nb == run]
        
        for fun in plot_funs:
            fun(data=traj, label=run, c=color)
    

    ax.axis('equal')
    ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))
    
    ax.yaxis.set_major_locator(ticker.MultipleLocator(5))
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(1))
    
    ax.set_xlim(left=0)
    
    
    #ax.set_aspect('equal')

    ax.grid(which='both')


def draw_all_trajectories(df, plot_funs: Callable):
    ants = pd.unique(df["ant_nb"])
    ax = plt.subplot()
        
    for ant in ants:
        ant_df = df[df.ant_nb == ant]
        draw_trajectories(ant_df, plot_funs, ax)
    
    goal = centroid(env["goals"])
    start = env["start"]
    ic(start)
    patches = [Circle(goal, radius=1, color='red'), Circle(start, radius=1, color='green')]
    ic(goal)
    ax.annotate('Goal', xy=goal,  xycoords='data',
        xytext=(0.8, 0.95), textcoords='axes fraction',
        horizontalalignment='right', verticalalignment='top',
    )
    ax.annotate('Start', xy=start,  xycoords='data',
        xytext=start, textcoords='axes fraction',
        horizontalalignment='right', verticalalignment='top',
    )
    for p in patches:
        ax.add_patch(p)


def draw_traj_heat(df, plot_funs, ants=None, axes=None, surprise=False):
    if ants is None:
        ants = pd.unique(df["ant_nb"])
    if axes is None:
        fig, axes = plt.subplots(max(1,len(ants) // 2),min(len(ants),2))
    
    for ant, ax in zip(ants, fig.axes):
        ant_df = df[df.ant_nb == ant]
        runs = pd.unique(ant_df["trial_nb"])
        spacing = np.linspace(0,1,len(runs))
        colors = list(reversed([(x,x,0,1-x/3) for x in spacing]))

        plt.sca(ax)
        max_y = max(list(df['path_x']))
        max_x = max(list(df['path_y']))

        for run, color in zip(runs, colors):

            traj = ant_df[ant_df.trial_nb == run]

            # PHIL: START
            sur_map = np.zeros((max_x+1, max_y+1))
            y_vals = list(traj['path_x'])
            x_vals = list(traj['path_y'])
            s_vals = list(traj['td_error_q'])
            for i in range(0, len(x_vals)):
              sur_map[x_vals[i]][y_vals[i]] += s_vals[i]
            plt.imshow(sur_map, cmap='jet', interpolation='nearest', vmin=0)  
            # PHIL: END
            
            for fun in plot_funs:
                fun(data=traj, label=run, c=color)

        ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
        ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(5))
        ax.yaxis.set_minor_locator(ticker.MultipleLocator(1))
        ax.set_xlim(left=0)
        ax.grid(which='both')



def plot_td_errors(df, env, q=False):
    # prep heatmap
    max_y = env["width"]#max(list(df['path_x'])) ##
    max_x = env["height"] #max(list(df['path_y'])) ##


    ants = pd.unique(df["ant_nb"])
    sur_map = np.zeros((max_x+1, max_y+1))
    plt.figure()

    for ant in ants:
        ant_df = df[df.ant_nb == ant]

        runs = pd.unique(ant_df["trial_nb"])
        #sur_map = np.zeros((max_x+1, max_y+1))
        #plt.figure()
        for run in runs:
        
            traj = ant_df[ant_df.trial_nb == run]
            #sur_map = np.zeros((max_x+1, max_y+1))

            y_vals = list(traj['path_x'])
            x_vals = list(traj['path_y'])
            if q:
                s_vals = list(traj['td_error_q'])
            else:
                s_vals = list(traj['td'])

            for i in range(0, len(x_vals)):
                sur_map[x_vals[i]][y_vals[i]] += s_vals[i]

            sur_map /= len(runs)
        #print(sur_map)

    sur_map /= len(ants) 
    #plt.xlim([5,15])
    #plt.ylim([35,60])
    plt.imshow(sur_map, cmap='Reds', interpolation='nearest', vmin=0)  


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


def compute_steps(dataset):
    """
    Compute the cumulative discounted reward of each episode in the dataset.

    Args:
        dataset (list): the dataset to consider;
        gamma (float, 1.): discount factor.

    Returns:
        The cumulative discounted reward of each episode in the dataset.

    """

    n_steps = []
    for trial in pd.unique(dataset["trial_nb"]):
        trial_data = dataset[dataset["trial_nb"] == trial]
        
        n_steps.append(len(trial_data.index))

    return n_steps


def plot_steps(df):
    condition_steps = []
    for condition in pd.unique(df["condition"]):
        df_cond = df[df["condition"]==condition]
        n_steps = []
        for ant in pd.unique(df_cond["ant_nb"]):
            ant_df = df_cond[df_cond["ant_nb"] == ant]
            n_steps.append(compute_steps(ant_df))
        n_steps = [np.mean(ant) for ant in n_steps]
        condition_steps.append(n_steps)
    plt.boxplot(condition_steps, labels=["no trap", "trap"])
    plt.show()


def centroid(data):
    x, y = zip(*data)
    l = len(x)
    return int(round(sum(x) / l)), int(round(sum(y) / l))


if __name__=="__main__":

    if "results" in snakemake.input._names:
        df = pd.read_csv(snakemake.input.results)

    if "env" in snakemake.input._names:
        with open(snakemake.input.env[0], 'rb') as env_i:
            env = pickle.load(env_i)

    if "Js" in snakemake.input._names:
        Js = np.load(snakemake.input.Js)
        ic(Js.shape)
    
    match snakemake.rule:
        case "plot_td_errors":
            plot_td_errors(df, env)
            plt.savefig(snakemake.output[0])
            plt.clf()
            plot_td_errors(df, env, q=True)
            plt.savefig(snakemake.output[1])
        
        case "plot_trajectories":
            draw_funs = [
                partial(plt.plot, 'path_x', 'path_y'),
                #partial(plt.plot, 'path_x', 'path_y', 'o')
            ]
            for ant in pd.unique(df["ant_nb"]):
                ax = plt.subplot()
                
                runs = pd.unique(df["trial_nb"])
                spacing = np.linspace(0,1,len(runs))
                colors = list(reversed([(x,x,0,1-x/3) for x in spacing]))

                ant_df = df[df.ant_nb == ant]
                draw_trajectories(ant_df, draw_funs, ax, colors=colors)
                plt.savefig(snakemake.output[ant])
                plt.clf()


        case "plot_all_trajectories":
            draw_funs = [
                partial(plt.plot, 'path_x', 'path_y'),
                #partial(plt.plot, 'path_x', 'path_y', 'o')
            ]
            draw_all_trajectories(df, draw_funs)

            plt.savefig(snakemake.output[0])
        
        case "plot_J":
            plot_J()

        case "plot_steps":
            plot_steps(df)
