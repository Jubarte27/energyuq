from contextlib import redirect_stdout
import os
from pathlib import Path
from typing import Any, Union, cast
import typing
import dill
import numpy as np
import easyvvuq as uq
from easyvvuq.sampling.stochastic_collocation import SCSampler
import chaospy as cp
import matplotlib.pyplot as plt
from easyvvuq.actions import CreateRunDirectory, Encode, Decode, ExecutePython, Actions

from .util import persist
from .wrappers import easy_wrapper
from .programs import *
from .machines import *
from .util.path import change_dir_permissions, latest_dir, next_file, next_dir


params_type = dict[str, dict[str, Any]]
vary_type = dict[str, cp.Distribution]

# Quantity of Interest
QOI = "energy_uj"
QOIS = ["energy_uj", "energy_scaled", "time"]
RESULTS_DIR = "run_results"


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


def default_params(machine: type[Machine]) -> tuple[params_type, vary_type]:
    params: params_type = {}
    vary: vary_type = {}

    # Current machine maximum number of cores
    # params["N_THREADS"] = {"type": "integer", "default": machine.physical_core_count}
    # vary["N_THREADS"] = cp.DiscreteUniform(1, machine.physical_core_count)
    # Glados limited number of cores
    params["N_THREADS"] = {"type": "integer", "default": machine.max_threads}
    vary["N_THREADS"] = cp.DiscreteUniform(1, machine.max_threads)

    # Clock frequencies available for our current machine:
    params["CLK"] = {"type": "integer", "default": len(machine.freq) - 1}
    vary["CLK"] = cp.DiscreteUniform(0, len(machine.freq) - 1)

    # params['POWER_CAP'] = {'type': 'integer', 'default': 220.0}  # power cap in watts

    # params["PLACE_WIDE"] = {"type": "integer", "default": len(machine.places) - 1}
    # vary["PLACE_WIDE"] = cp.DiscreteUniform(0, len(machine.places) - 1)
    
    # params["AFF_DISTANCE"] = {"type": "integer", "default": len(machine.proc_bind) - 1}
    # vary["AFF_DISTANCE"] = cp.DiscreteUniform(0, len(machine.proc_bind) - 1)

    return params, vary


def energy_wraper_actions(
    program: type[Program], machine: type[Machine], root: Path
) -> uq.actions.Actions:
    # input file encoder
    
    template = ",".join([f"${param}" for param in default_params(machine)[0]])
    
    with open("easy/energy.template", "w") as f:
        f.write(f"{template}\n")
    
    encoder = uq.encoders.GenericEncoder(
        template_fname="easy/energy.template", delimiter="$", target_filename="input.csv"
    )

    # CSV output file decoder
    decoder = uq.decoders.SimpleCSV(target_filename="output.csv", output_columns=QOIS)

    # Local execution of the wrapper around benchmarks
    parent_path = root.joinpath("output").absolute()
    create_dir(parent_path)

    def wrapper(params: dict):
        params_str = " and ".join(f"{k} = {params[k]}" for k in params)
        print(f"Running {program.name} on {machine.name} with {params_str}")
        file_path = next_file(parent_path, program.name)
        with open(file_path.as_posix(), "w") as f:
            with redirect_stdout(f):
                easy_wrapper.main(program, machine)

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


def campaign_path(root: Path):
    return root.joinpath("campaign")


def create_campaign(program: type[Program], machine: type[Machine], root: Path):
    path = campaign_path(root)
    create_dir(path)
    change_dir_permissions(path, 0o777)

    params, vary = default_params(machine)
    campaign = uq.Campaign(
        name="energy",
        db_location="sqlite:///" + path.as_posix() + "/campaign.db",
        work_dir=path.as_posix(),
    )
    setattr(campaign, "root_path", root)
    
    # the database was empty, create a new app
    if campaign.get_active_app() is None:
        campaign.add_app(name=campaign.campaign_name, params=params, actions=energy_wraper_actions(program, machine, root))
        sampler = uq.sampling.SCSampler(
            vary=vary,
            polynomial_order=1,
            quadrature_rule="C",
            sparse=True,
            midpoint_level1=True,
            dimension_adaptive=True,
        )
        campaign.set_sampler(sampler)
    
    return campaign


def prepare_campaign(
    program: type[Program], machine: type[Machine], root: Path
) -> uq.campaign.Campaign:
    """
    Creates a campaign and runs the first execution
    """

    campaign = create_campaign(program, machine, root)
    campaign.execute(sequential=True).collate(progress_bar=True)

    return campaign


def prepare_analysis(campaign: uq.campaign.Campaign) -> uq.analysis.SCAnalysis:
    sampler: SCSampler = cast(SCSampler, campaign.get_active_sampler())
    analysis = uq.analysis.SCAnalysis(
        sampler=sampler, qoi_cols=[QOI]
    )
    if not hasattr(analysis, "l_norm"):
        analysis.l_norm = sampler.l_norm

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
    start_index=1, # for display only
    min_number_of_refinements = -1,
    max_number_of_refinements = 100
):
    sampler = get_sampler(campaign)

    def single_iteration(i):
        # compute the admissible indices
        sampler.look_ahead(analysis.l_norm)

        if len(sampler.admissible_idx) == 0:
            # we searched everything
            return False

        print(f"-------{i + start_index + 1}{({0: 'st', 1: 'nd', 2: 'rd'}.get(i + start_index, 'th'))} iteration: {sampler.n_new_points[-1]} new points------")
        campaign.execute(sequential=True).collate(progress_bar=True)

        analysis.adapt_dimension(QOI, campaign.get_collation_result(), method="var")
        return True
    i = 0

    def explored_enough(thresh=1e-3):
        sobols = analysis.get_sobol_indices(QOI)
        max_orders = np.max(analysis.l_norm, 0)
        return (all(
            order > 1 or
            not all(sobol <= thresh for permut, sobol in sobols.items() if dim + 1 in permut)
            for dim, order in enumerate(max_orders)
        ))

    def advance():
        nonlocal i
        return single_iteration(i) and (i := i + 1) < max_number_of_refinements

    # pick better rtol and atol
    def converged(rtol=1e-2, atol=1e-3) -> bool:
        assert len(analysis.adaptation_errors) >= 3
        last3 = analysis.adaptation_errors[-3:]
        return np.all(np.isclose(last3[:-1], last3[1:], rtol=rtol, atol=atol), 0)
    
    while len(analysis.adaptation_errors) < 3:
        print("Adapt because too few runs")
        if not advance(): return
    
    while i < min_number_of_refinements:
        print("Adapt because min_number_of_refinements")
        if not advance(): return

    while not explored_enough():
        print("Adapt because something was not properly explored")
        if not advance(): return
    
    while not converged():
        print(f"Adapt because it has not converged yet {analysis.adaptation_errors[-3:]}")
        if not advance(): return



def refine_and_analyse(
    campaign: uq.campaign.Campaign,
    analysis: uq.analysis.SCAnalysis,
    min_number_of_refinements=-1,
    max_number_of_refinements=100
):
    refine_sampling_plan(campaign, analysis, min_number_of_refinements=min_number_of_refinements, max_number_of_refinements=max_number_of_refinements)
    campaign.apply_analysis(analysis)

def run_dir(*, name: str = "energy", dir: Union[str, None] = None, campaign: Union[uq.campaign.Campaign, None] = None):
    if dir:
        return Path(dir)
    if campaign:
        if hasattr(campaign, "root_path"):
            return getattr(campaign, "root_path")
        app = campaign.get_active_app()
        if app:
            name = app["name"]
    
    create_dir(Path(RESULTS_DIR))
    return next_dir(RESULTS_DIR, name)

def create(
    program: type[Program], machine: type[Machine], /, dir: Union[str, None] = None
):
    root = run_dir(dir=dir)
    create_dir(root)

    ### campaign and analysis are the only independent ones, maybe we could cut a few steps at the saving stage????
    campaign = prepare_campaign(program, machine, root)
    analysis = prepare_analysis(campaign)
    # perform analysis (basically estimates moments, Sobol analysis, and updates internal state of analysis)
    campaign.apply_analysis(analysis)

    return campaign, analysis


def save(
    campaign: uq.campaign.Campaign,
    analysis: uq.analysis.SCAnalysis,
    /,
    dir: Union[str, None] = None,
):
    path = run_dir(dir=dir, campaign=campaign)

    create_dir(path)

    analysis.save_state(path.joinpath("analysis").as_posix())


def load(
    program: type[Program],
    machine: type[Machine],
    campaign_name: str,
    /,
    dir: Union[str, None] = None,
) -> tuple[uq.campaign.Campaign, uq.analysis.SCAnalysis]:

    if not dir:
        path = latest_dir(RESULTS_DIR, campaign_name)
        if path is None:
            raise Exception(
                "No directory was informed and could not find using the default pattern"
            )
    else:
        path = Path(dir)

    campaign = create_campaign(program, machine, path)
    analysis = prepare_analysis(campaign)
    analysis.load_state(path.absolute().joinpath("analysis"))
    
    # perform analysis (basically estimates moments, Sobol analysis, and updates internal state of analysis)
    campaign.apply_analysis(analysis)

    return campaign, analysis
