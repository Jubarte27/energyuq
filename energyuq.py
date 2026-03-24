from contextlib import redirect_stdout
import os
from pathlib import Path
from typing import Any, Union
import typing
from collections.abc import Callable
import dill
import numpy as np
import easyvvuq as uq
import chaospy as cp
import matplotlib.pyplot as plt
from easyvvuq.actions import CreateRunDirectory, Encode, Decode, ExecutePython, Actions

from util import persist
import energy_wrapper
from programs import *
from machines import *
import re

# Quantity of Interest
QOI = "energy_uj"
MACHINE = Glados


class ExecuteWrapper:
    def __init__(self, function):
        self.function = dill.dumps(function)
        self.params = None
        self.eval_result = None

    def start(self, previous: typing.Union[None, dict[str, Any]] = None):
        if not previous:
            return None

        function = dill.loads(self.function)

        if "rundir" in previous:
            self.old_dir = os.getcwd()
            os.chdir(previous["rundir"])

        function(previous["run_info"]["params"])

        self.finalise()
        return previous

    def finished(self):
        return True

    def finalise(self):
        if hasattr(self, "old_dir"):
            os.chdir(self.old_dir)
            del self.old_dir

    def succeeded(self):
        if not self.finished():
            raise RuntimeError("action did not finish yet")
        else:
            return True


params_type = dict[str, dict[str, Any]]
vary_type = dict[str, cp.Distribution]


def default_params(machine: type[Machine]) -> tuple[params_type, vary_type]:
    params: params_type = {}
    vary: vary_type = {}

    # Current machine maximum number of cores
    params["N_THREADS"] = {"type": "integer", "default": machine.physical_core_count}
    vary["N_THREADS"] = cp.DiscreteUniform(1, machine.physical_core_count)

    # Clock frequencies available for our current machine:
    params["CLK"] = {"type": "integer", "default": len(machine.freq) - 1}
    vary["CLK"] = cp.DiscreteUniform(0, len(machine.freq) - 1)

    # params['POWER_CAP'] = {'type': 'integer', 'default': 220.0}  # power cap in watts

    return params, vary


def energy_wraper_actions(
    program: type[Program], machine: type[Machine], root: Path
) -> uq.actions.Actions:
    # input file encoder
    encoder = uq.encoders.GenericEncoder(
        template_fname="energy.template", delimiter="$", target_filename="input.csv"
    )

    # CSV output file decoder
    decoder = uq.decoders.SimpleCSV(target_filename="output.csv", output_columns=[QOI])

    # Local execution of the wrapper around benchmarks
    parent_path = root.joinpath("output").absolute()
    create_dir(parent_path)

    def wrapper(params: dict):
        params_str = " and ".join(f"{k} = {params[k]}" for k in params)
        print(f"Running {program.name} on {machine.name} with {params_str}")
        file_path = next_file(parent_path, program.name)
        with open(file_path.as_posix(), "w") as f:
            with redirect_stdout(f):
                energy_wrapper.main(program, machine)

    execute = ExecuteWrapper(wrapper)

    # actions to be undertaken: make rundirs, encode input files, execute local model ensemble, decode output files
    return Actions(
        CreateRunDirectory(root=campaign_path(root), flatten=True),
        Encode(encoder),
        execute,
        Decode(decoder),
    )


def create_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def create_file(path: Path):
    path.touch()


def campaign_path(root: Path):
    return root.joinpath("campaign")


def prepare_campaign(
    program: type[Program], machine: type[Machine], root: Path
) -> uq.campaign.Campaign:
    """
    Creates a campaign and runs the first execution
    """
    create_dir(campaign_path(root))

    params, vary = default_params(machine)
    campaign = uq.Campaign(
        name="energy",
        params=params,
        actions=energy_wraper_actions(program, machine, root),
        work_dir=campaign_path(root).as_posix(),
    )
    setattr(campaign, "root_path", root)

    sampler = uq.sampling.SCSampler(
        vary=vary,
        polynomial_order=1,
        quadrature_rule="C",
        sparse=True,
        midpoint_level1=True,
        dimension_adaptive=True,
    )
    campaign.set_sampler(sampler)

    campaign.execute(sequential=True).collate(progress_bar=True)

    return campaign


def prepare_analysis(campaign: uq.campaign.Campaign) -> uq.analysis.SCAnalysis:
    analysis = uq.analysis.SCAnalysis(
        sampler=campaign.get_active_sampler(), qoi_cols=[QOI]
    )

    # perform analysis (basically estimates moments, Sobol analysis, and updates internal state of analysis)
    campaign.apply_analysis(analysis)

    return analysis


def plot_new_points(new_points):
    plt.figure()
    xs = np.array([x for x, y in new_points])
    ys = np.array([y for x, y in new_points])
    plt.plot(xs, ys, "o")
    plt.show()


def get_sampler(campaign: uq.campaign.Campaign):
    sampler = campaign.get_active_sampler()
    sampler = typing.cast(uq.sampling.SCSampler, sampler)
    return sampler


def refine_sampling_plan(
    campaign: uq.campaign.Campaign,
    analysis: uq.analysis.SCAnalysis,
    number_of_refinements,
):
    """
    Refine the sampling plan.

    Parameters
    ----------
    number_of_refinements (int)
    The number of refinement iterations that must be performed.

    Returns
    -------
    None. The new accepted indices are stored in analysis.l_norm and the admissible indices
    in sampler.admissible_idx.
    """
    sampler = get_sampler(campaign)

    for i in range(number_of_refinements):
        # compute the admissible indices
        sampler.look_ahead(analysis.l_norm)

        if sampler.n_new_points[-1] == 0:
            # maybe we should stop??
            pass

        if len(sampler.admissible_idx) == 0:
            # we searched everything
            break

        print(f"-------{i + 1} iteration: {sampler.n_new_points[-1]} new points------")
        # run the ensemble
        campaign.execute(sequential=True).collate(progress_bar=True)

        # accept one of the multi indices of the new admissible set
        data_frame = campaign.get_collation_result()
        analysis.adapt_dimension(QOI, data_frame, method="var")


def refine_and_analyse(
    campaign: uq.campaign.Campaign,
    analysis: uq.analysis.SCAnalysis,
    number_of_refinements,
):
    refine_sampling_plan(campaign, analysis, number_of_refinements)
    campaign.apply_analysis(analysis)


def create(
    program: type[Program], machine: type[Machine], /, dir: Union[str, None] = None
):
    if dir is None:
        root = next_dir("run_results", "energy")
    else:
        root = Path(dir)
    create_dir(root)

    ### campaign and analysis are the only independent ones, maybe we could cut a few steps at the saving stage????
    campaign = prepare_campaign(program, machine, root)
    analysis = prepare_analysis(campaign)

    return campaign, analysis


def next_path(
    dir: Union[str, Path],
    prefix: str = "",
    suffix="",
    restriction: Callable[[Path], bool] = lambda _: True,
) -> Path:
    name_pattern = re.compile(f"{re.escape(prefix)}([0-9]+){re.escape(suffix)}")
    parent = Path(dir)
    last = max(
        (
            int(m.group(1))
            for p in parent.iterdir()
            if restriction(p)
            and (name := p.relative_to(parent).as_posix())
            and (m := name_pattern.fullmatch(name))
            and m
        ),
        default=-1,
    )
    return Path(parent, f"{prefix}{last + 1}{suffix}")


def next_dir(dir: Union[str, Path], base_name: str) -> Path:
    return next_path(dir, prefix=base_name + "_", restriction=Path.is_dir)


def next_file(dir: Union[str, Path], base_name: str) -> Path:
    return next_path(dir, prefix=base_name + "_", restriction=Path.is_file)


def save(
    campaign: uq.campaign.Campaign,
    analysis: uq.analysis.SCAnalysis,
    /,
    dir: Union[str, None] = None,
):
    data_frame = campaign.get_collation_result()
    sampler = get_sampler(campaign)
    vary = sampler.vary

    if not dir:
        path = getattr(
            campaign, "root_path", next_dir("run_results", campaign.campaign_name)
        ).joinpath("pickles")
    else:
        path = Path(dir)

    create_dir(path)

    persist.save(analysis, sampler, data_frame, vary, [QOI], path.as_posix())
