from datetime import datetime
import pickle
import numpy as np
from insect_rl.simulation import EnvironmentConfig
from insect_rl.algorithms.evaluation import IAVIEvaluation
from insect_rl.mdp.utils import grid_math



NOW = datetime.now().strftime('%d_%m_%Y_%H_%M')


module data_prep:
    snakefile:
        "data_preparation.smk"
    config: config
    prefix: "data-prep"


def get_simulation_configs(wildcards):
    return {
        'wind': float(wildcards.wind),
        'reward': float(config["simulation_settings"]["reward"]),
        'trapcost': float(config["simulation_settings"]["trapcost"])
    }


rule construct_environment_config:
    input:
        os.path.abspath(f"{config['env_config']}"),
        [
            Path("./").joinpath(Path(output_)).resolve()
            for output_ in rules.data_prep_interpolate_trajectories.output
        ]
    output:
        "{experiment}/{condition}/env.pickle"
    script:
        "../scripts/inverse_rl/read_config.py"


# converts the coordinates to state int, action int, reward, next state int
rule convert_dataframe:
    input:
        [
            Path("./").joinpath(Path(output_)).resolve()
            for output_ in rules.data_prep_interpolate_trajectories.output
        ],
        rules.construct_environment_config.output
    output:
        "{experiment}/{condition}/data.csv"
    script:
        "../scripts/inverse_rl/convert_dataframe.py"


rule make_envc:
    input:
        rules.construct_environment_config.output
    output:
        "{experiment}/{condition}/{wind}/envc.pickle"
    run:
        with open(input[0], 'rb') as config_file:
            config = pickle.load(config_file)
        settings = get_simulation_configs(wildcards)
        envc = EnvironmentConfig.from_kwargs(**(config | settings))
        print(envc)
        with open(output[0], 'wb') as o:
            pickle.dump(envc, o)


rule transition_probabilities:
    input:
        rules.construct_environment_config.output
    output:
        "{experiment}/{condition}/transition_probs_wind={wind}.npy"
    script:
        "../scripts/inverse_rl/get_transition_probs.py"


rule interpret_data:
    input:
        rules.convert_dataframe.output,
        rules.make_envc.output,
        rules.transition_probabilities.output
    params:
        settings = get_simulation_configs
    output:
        "{experiment}/{condition}/{wind}/results/action_probs.npy",
        "{experiment}/{condition}/{wind}/results/ground_r.npy"
    script:
        "../scripts/run_simulation.py"


rule IAVI:
    input:
        rules.make_envc.output,
        rules.transition_probabilities.output,
        rules.interpret_data.output
    output:
        "{experiment}/{condition}/{wind}/results/q.npy",
        "{experiment}/{condition}/{wind}/results/r.npy",
        "{experiment}/{condition}/{wind}/results/boltz.npy"
        #"{experiment}/{condition}/{wind}/results/v_real.npy",
        #"{experiment}/{condition}/{wind}/results/v_true.npy",
        #"{experiment}/{condition}/{wind}/results/v_iavi.npy"
    run:
        with open(input[0], 'rb') as envc_file:
            envc = pickle.load(envc_file)
        print(envc)
        transition_probs = np.load(input[1])
        action_probs = np.load(input[2])
        #ground_r = np.load(input[3])

        _eval = IAVIEvaluation(envc.width, envc.height, envc.start)
        res = _eval(action_probs, transition_probs)
        np.save(output[0], res['q'])
        np.save(output[1], res['r'])
        np.save(output[2], res['boltz'])


rule plot_results:
    input:
        rules.make_envc.output,
        rules.transition_probabilities.output,
        rules.interpret_data.output, # action_probs, ground_r
        rules.IAVI.output # q, r, boltz
    output:
        "{experiment}/{condition}/{wind}/plots/ground_r.png",
        "{experiment}/{condition}/{wind}/plots/iavi90_iavi99.png",
        "{experiment}/{condition}/{wind}/plots/V.png",
        "{experiment}/{condition}/{wind}/plots/R.png",
        "{experiment}/{condition}/{wind}/plots/IAVIR.png",
        "{experiment}/{condition}/{wind}/plots/optimal_ground_iavi.png"
    notebook:
        "../notebooks/irl/plot_results.py.ipynb"


rule all:
    input:
        expand("{experiment}/{condition}/{wind}/plots/optimal_ground_iavi.png", 
            experiment=config["experiment"],
            condition=config["conditions"],
            wind=config["simulation_settings"]["winds"]
        )
    default_target: True
