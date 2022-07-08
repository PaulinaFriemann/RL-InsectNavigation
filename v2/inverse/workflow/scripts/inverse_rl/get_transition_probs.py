import pickle
import numpy as np
import yaml

from insect_rl.simulation import Simulation, EnvironmentConfig
from insect_rl.mdp.utils import grid_math


with open(snakemake.input[0], 'rb') as config_file:
    config = pickle.load(config_file)



# TODO Env Config ignores keyowords that are not recognized, but maybe i want them eg for figs
actions = vars(grid_math)[snakemake.config['actions']]
width = config['width']
height = config['height']

# width, height, actions, wind, traps, trap_exits
transition_probabilities = Simulation.calc_transition_probabilities(
    actions=actions, wind=float(snakemake.wildcards['wind']), **config)

np.save(snakemake.output[0], transition_probabilities)
