import os
import itertools as it
from pathlib import Path
from pylatex import Document, Section, Subsection, Tabular, Math, TikZ, Axis, \
    Plot, Figure, Alignat, SubFigure, StandAloneGraphic, Package, HFill
from pylatex.section import Paragraph, Subsubsection
from pylatex.utils import italic, escape_latex, fix_filename, NoEscape
import plotresults

tex = """\\begin\{figure\}[h]
	\\centering
	\\begin\{subfigure\}[b]\{0.3\\textwidth\}
		\\centering
		\\includesvg[width=\\textwidth]\{figures/experiments/forward/lr0.8/r10tc-10aINTERg0.99/0-no-trap/episode_lens.svg\}
		\\includesvg[width=\\textwidth]\{figures/experiments/forward/lr0.8/r10tc-10aINTERg0.99/0-no-trap/J.svg\}
		\\caption\{\}
		\\label\{fig:J_notrap\}
	\\end\{subfigure\}
	\\hfill
	\\begin\{subfigure\}[b]\{0.3\\textwidth\}
		\\centering
		\\includesvg[width=\\textwidth]\{figures/experiments/forward/lr0.8/r10tc-10aINTERg0.99/1-trap/episode_lens.svg\}
		\\includesvg[width=\\textwidth]\{figures/experiments/forward/lr0.8/r10tc-10aINTERg0.99/1-trap/J.svg\}
		\\caption\{\}
		\\label\{fig:episode_len_notrap\}
	\\end\{subfigure\}
	\\hfill
	\\begin\{subfigure\}[b]\{0.3\\textwidth\}
		\\centering
		\\includesvg[width=\\textwidth]\{figures/experiments/forward/lr0.8/r10tc-10aINTERg0.99/2-no-trap/episode_lens.svg\}
		\\includesvg[width=\\textwidth]\{figures/experiments/forward/lr0.8/r10tc-10aINTERg0.99/2-no-trap/J.svg\}
		\\caption\{\}
		\\label\{fig:episode_len_notrap\}
	\\end\{subfigure\}
	\\caption\{\\todo\{Goal reward: 10, Trap cost: -10\}\}
	\\label\{fig:trajectories\}
\\end\{figure\}
"""

class StandAloneSVG(StandAloneGraphic):
    r"""A class representing a stand alone image."""

    _latex_name = "includesvg"

    packages = [Package('svg')]

    _repr_attributes_mapping = {
        "filename": "arguments",
        "image_options": "options"
    }

    def __init__(self, filename, image_options=..., extra_arguments=None):
        super().__init__(filename, image_options, extra_arguments)

    
def add_svg(self, filename, *, width=NoEscape(r'\linewidth'),
                placement=None):
    """Add an svg image to the subfigure.
    Args
    ----
    filename: str
        Filename of the image.
    width: str
        Width of the image in LaTeX terms.
    placement: str
        Placement of the figure, `None` is also accepted.
    """

    if width is not None:
        if self.escape:
            width = escape_latex(width)

        width = 'width=' + str(width)

    if placement is not None:
        self.append(placement)

    self.append(StandAloneSVG(image_options=width,
                                    filename=fix_filename(filename)))

SubFigure.add_svg = add_svg

def reward_name(reward_dir):
    rew, tc = [int(r) for r in reward_dir[1:-11].split("tc")]
    return f"{rew=}_{tc=}"

RESULTS_DIR = Path('RESULTS/FULLEXPERIMENT')
FIG_DIR = Path("/home/paulina/Documents/thesis/figures/experiments/forward")
PIC_LIST = [['episode_lens', 'J'], ['all_last_trajectories']]


def make_fig(paragraph, fig_dir, pic, cap="", subcap=""):
    if cap == "":
        cap = paragraph.title
    if subcap == "":
        subcap = " ".join(fig_dir.parts[-3:])
    with paragraph.create(Figure(position='h!')) as fig:
        fig.content_separator = "\n"

        for condition in ["0-no-trap", "1-trap", "2-no-trap"]:
            size = 1 / len(["0-no-trap", "1-trap", "2-no-trap"])
            with fig.create(SubFigure(position='h', width=NoEscape(f'{size - 0.01}\\textwidth'))) as subfig:
                subfig.content_separator = "\n"

                for sub_pic in pic:
                    subfig.add_svg(str(fig_dir / condition / f'{sub_pic}.svg'))
                subfig.add_caption(condition + subcap)
            fig.append(HFill())
        fig.add_caption("Fig" + cap)


if __name__ == '__main__':

    results = {}
    
    for lr in [d for d in RESULTS_DIR.iterdir() if d.is_dir() and d.name.startswith('lr')]:
        results[lr.name] = {}
        for mc in lr.iterdir():
            #for reward_settings in mc.iterdir():
            results[lr.name][mc.name] = [d.name for d in mc.iterdir()]
            #for reward_settings in mc.iterdir():
                #fig_dir = FIG_DIR / lr.name / mc.name / reward_settings.name
                #plotresults.run(reward_settings, fig_dir)

    default = ['lr0.5', 'nomovecost', 'r100tc-100aINTERg0.99']
    fig_dir = FIG_DIR / '/'.join(default)
    par_name = " ".join(default[:2] + [reward_name(default[2])])
    paragraph = Paragraph(par_name)
    paragraph.content_separator = "\n"
    default_pics = PIC_LIST + [['']]
    for pic in PIC_LIST:
        make_fig(paragraph, fig_dir, pic)
    paragraph.generate_tex(str(FIG_DIR / "default"))
    results[default[0]][default[1]].remove(default[2])
    
    ## differing rewards
    rewards = results[default[0]][default[1]]
    subsec = Subsubsection("Differing Rewards")
    for reward_set in rewards:
        fig_dir = FIG_DIR / '/'.join(default[0:2] + [reward_set])
        par_name = " ".join(default[:2] + [reward_name(reward_set)])
        with subsec.create(Paragraph(par_name)) as paragraph:
            paragraph.content_separator = "\n"
            for pic in PIC_LIST:
                make_fig(paragraph, fig_dir, pic)
        subsec.generate_tex(str(FIG_DIR / "diff_rewards"))

    ## movecosts
    rewards = results[default[0]]['movecost']
    subsec = Subsubsection("Adding Move Costs")
    for reward_set in rewards:
        fig_dir = FIG_DIR / default[0] / 'movecost' / reward_set
        par_name = default[0] + ' movecost ' + reward_name(reward_set) + "."
        with subsec.create(Paragraph(par_name)) as paragraph:
            paragraph.content_separator = "\n"
            for pic in PIC_LIST:
                make_fig(paragraph, fig_dir, pic)
        subsec.generate_tex(str(FIG_DIR / "movecost"))
    
    ## learning rates
    subsec = Subsubsection("Different Learning Rates")
    for lr in [lr for lr in results.keys() if lr != default[0]]:
        par_name = f"Learning rate {lr[2:]}."

        with subsec.create(Paragraph(par_name)) as paragraph:
            paragraph.content_separator = "\n"
            rewards = results[lr]['nomovecost']
            for reward_set in rewards:
                fig_dir = FIG_DIR / lr / 'nomovecost' / reward_set
                for pic in PIC_LIST:
                    make_fig(paragraph, fig_dir, pic)
        subsec.generate_tex(str(FIG_DIR / "learningrates"))
