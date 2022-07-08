from snakemake.utils import min_version
min_version("6.0")

configfile: "workflow/config/config.yaml"


from pathlib import Path
import pickle


module data_prep:
    snakefile:
        "data_preparation.smk"
    config: config
    prefix: "data-prep"


rule load_data:
    input:
        raw = [
            Path("./").joinpath(Path(input_)).resolve()
            for input_ in rules.data_prep_transform_data.input
        ],
        discrete = [
            Path("./").joinpath(Path(input_)).resolve()
            for input_ in rules.data_prep_interpolate_trajectories.input
        ],
        interpolated = [
            Path("./").joinpath(Path(input_)).resolve()
            for input_ in rules.data_prep_interpolate_trajectories.output
        ]


rule all:
    input:
        expand(
            "{experiment}/figs/{condition}/{version}_{type_}.{ext}", 
            experiment=config["experiment"],
            condition=config["conditions"],
            version=["raw", "discrete", "interpolated"],
            type_=["trajs", "heatmap", "heatmapandtrajs"],
            ext=["png"]
        )


def get_drawing_input(wildcards):
    return rules.load_data.input[wildcards.version]


rule draw_trajectories:
    input:
        get_drawing_input
    output:
        "{experiment}/figs/{condition}/{version}_trajs.png"
    script:
        "../scripts/viz/draw_trajectories.py"


rule draw_heatmap:
    input:
        get_drawing_input
    params:
        x_bin = 20,
        y_bin = 50
    output:
        "{experiment}/figs/{condition}/{version}_heatmap.png",
        temp("{experiment}/figs/{condition}/{version}_heatmap.pickle")
    script:
        "../scripts/viz/draw_heatmaps.py"


rule draw_heatmap_and_trajectories:
    input:
        "{experiment}/figs/{condition}/{version}_heatmap.pickle",
        get_drawing_input
    output:
        "{experiment}/figs/{condition}/{version}_heatmapandtrajs.png",
    notebook:
        "../notebooks/draw_trajsandheatmap.py.ipynb"
