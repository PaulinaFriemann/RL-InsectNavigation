import seaborn as sns
import matplotlib.pyplot as plt

from functools import partial
import pickle
from typing import Callable

import pandas as pd
import numpy as np
import matplotlib.ticker as ticker
from matplotlib.patches import Circle


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


def colorbar(mappable):
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    import matplotlib.pyplot as plt
    last_axes = plt.gca()
    ax = mappable.axes
    fig = ax.figure
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad="5%")
    cbar = fig.colorbar(mappable, cax=cax)
    plt.sca(last_axes)
    return cbar


def plot_grids(width, height, start, **vs):
    ncols = min(len(vs), 4)
    fig, axes = plt.subplots(nrows=(len(vs) // 5) +1,
                             ncols=ncols,
                             sharey=True,
                             squeeze=False,
                             gridspec_kw={'width_ratios': [1]*(ncols -1) + [1.1]})
    vs_min = min(v.min() for (_,v) in vs.items())
    vs_max = max(v.max() for (_,v) in vs.items())

    cmap = sns.color_palette("Spectral_r", as_cmap=True)


    for i, (v_name, v) in enumerate(vs.items()):
        ax = i // (ncols + 1)
        col = i % (ncols + 1)
        name = v_name.replace("_", " ")
        axes[ax][col].set_title(name, fontsize="x-large")
        im = axes[ax][col].imshow(v.reshape(height, width), vmin=vs_min, vmax=vs_max, cmap=cmap)
        axes[ax][col].invert_yaxis()
        if col != 0:
            axes[ax][col].axes.get_yaxis().set_visible(False)
        #axes[ax][col].axes.get_yaxis().set_visible(False)
        if col == ncols - 1:
            colorbar(im)
        
        # mark start

    fig.tight_layout()

    return fig


