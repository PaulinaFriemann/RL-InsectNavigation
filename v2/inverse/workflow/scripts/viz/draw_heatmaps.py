import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def draw_bins(df, bins:tuple[int,int] =None):
    if not bins:
        bins = (
                len(pd.unique(df.path_x)),
                len(pd.unique(df.path_x)) * int((df.path_y.max() // df.path_x.max()))    
               )
    
    heatmap, xedges, yedges = np.histogram2d(df.path_x, df.path_y, bins=bins)

    fig = plt.figure()#figsize = (10,10))
    plt.imshow(heatmap.T, origin='lower', aspect='equal')
    return fig


if __name__ == "__main__":
    df = pd.read_csv(snakemake.input[0])
    bins = snakemake.params

    fig = draw_bins(df, bins=bins)
    plt.savefig(snakemake.output[0])
    
    with open(snakemake.output[1], 'wb') as o:
        pickle.dump(fig,o)
