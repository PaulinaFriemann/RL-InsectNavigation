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


def get_goals(data_path):
    df = pd.read_csv(data_path)
    by_ant = df.groupby(["ant_nb", "trial_nb"]).tail(1)
    goals = list(set(zip(by_ant.path_x, by_ant.path_y)))
    goals = [tuple(goal) for goal in goals]
    return goals


def read_config_file(config_path, data_path_no_trap, data_path_trap):

    with open(config_path) as f:
        config = yaml.safe_load(f)

    config = {k: eval_(v) for k, v in config.items()}

    goals_no_trap = get_goals(data_path_no_trap)
    goals_trap = get_goals(data_path_trap)

    goals = list(set(goals_no_trap + goals_trap))

    config["start"] = tuple(config["start"])
    # TODO maybe this needs to be changed? like maybe have an area?
    # TODO or make it specific to the run
    config["goals"] = goals

    config["traps"] = [tuple(trap) for trap in config["traps"]]
    config["trap_exits"] = [[tuple(s) for s in exit_] for exit_ in config["trap_exits"]]

    return config


if __name__=="__main__":
    no_trap = snakemake.input['no_trap_data']
    trap = snakemake.input['trap_data']
    config = read_config_file(snakemake.input[0], no_trap, trap)

    print(config)
    

    with open(snakemake.output[0], 'wb') as o:
        pickle.dump(config, o)
