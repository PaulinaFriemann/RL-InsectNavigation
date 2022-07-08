import yaml
import json
import pickle

import pandas as pd


def eval_(v):
    if isinstance(v, list):
        return eval('[' + ','.join(v) + ']')
    elif isinstance(v, str):
        return eval(v)
    return v


def read_config_file(config_path, data_path):

    with open(config_path) as f:
        config = yaml.safe_load(f)

    config = {k: eval_(v) for k, v in config.items()}

    df = pd.read_csv(data_path)
    by_ant = df.groupby(["ant_nb", "trial_nb"]).tail(1)

    config["start"] = tuple(config["start"])
    # TODO maybe this needs to be changed? like maybe have an area?
    # TODO or make it specific to the run
    config["goals"] = list(set(zip(by_ant.path_x, by_ant.path_y)))
    config["goals"] = [tuple(goal) for goal in config["goals"]]

    if 'no-trap' in snakemake.wildcards:
        config['traps'] = []
        config['trap_exits'] = []

    config["traps"] = [tuple(trap) for trap in config["traps"]]
    config["trap_exits"] = [[tuple(s) for s in exit_] for exit_ in config["trap_exits"]]

    return config


if __name__=="__main__":

    config = read_config_file(*snakemake.input)

    with open(snakemake.output[0], 'wb') as o:
        #json.dump(config, o)
        pickle.dump(config, o)
