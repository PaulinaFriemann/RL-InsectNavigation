from functools import partial
from typing import Callable
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


DEFAULT_FUNS = [

]


def draw_trajectories(df, plot_funs: Callable, ants: list[int] =None, axes=None):
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

        for run, color in zip(runs, colors):

            traj = ant_df[ant_df.trial_nb == run]
            
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
        #fig.tight_layout()
        
            #traj.rolling(3).mean().plot(x="path_x", y="path_y", ax=ax, c=color, label="mean")


if __name__ == "__main__":
    df = pd.read_csv(snakemake.input[0])

    draw_funs = [
        partial(plt.plot, 'path_x', 'path_y'),
        partial(plt.plot, 'path_x', 'path_y', 'o')
    ]
    draw_trajectories(df, draw_funs, ants=[pd.unique(df["ant_nb"])[0]])
    plt.savefig(snakemake.output[0])
