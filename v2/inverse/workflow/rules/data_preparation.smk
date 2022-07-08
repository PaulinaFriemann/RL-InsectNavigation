configfile: "workflow/config/config.yaml"


import os
import pandas as pd
import operator
from functools import partial


rule all:
    input:
        expand("{experiment}/{condition}_interpolated.csv", 
            experiment=config["experiment"],
            condition=config["conditions"])



# separate data into trap and no trap
rule separate_data:
    input:
        config["data"]
    output:
        expand("{experiment}/{condition}_raw.csv",
            experiment=config["experiment"],
            condition=config["conditions"])
    log:
        "logs/data_transformation/separate_data.log"
    run:
        if not os.path.exists(os.path.dirname(output[0])):
            os.makedirs(os.path.dirname(output[0]))
        df = pd.read_csv(input[0])
        for i, (condition, value) in enumerate(config["conditions"].items()):
            condition_ = eval(value)
            df[condition_(df["trial_nb"])].to_csv(output[i], index=False)

# transform the data
rule transform_data:
    input:
        "{experiment}/{condition}_raw.csv"
    output:
        "{experiment}/temp/{condition}_discrete.csv"
    notebook:
        "../notebooks/transform_data.py.ipynb"

def get_transition_probs(wildcards):
    t = f"irl/{wildcards.experiment}/{wildcards.condition}/transition_probs_wind=0.0.npy"
    print(os.path.abspath(t))
    return t

# interpolate the trajectories
rule interpolate_trajectories:
    input:
        "{experiment}/temp/{condition}_discrete.csv"
    output:
        "{experiment}/{condition}_interpolated.csv"
    threads: 4
    notebook:
        "../notebooks/interpolate.py.ipynb"
