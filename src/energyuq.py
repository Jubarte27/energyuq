from collections.abc import Callable
from contextlib import redirect_stdout
from dataclasses import fields
import os
from pathlib import Path
from typing import Any, cast

import chaospy as cp
import dill
import easyvvuq as uq
from easyvvuq.actions import Actions, CreateRunDirectory, Decode, Encode
from easyvvuq.sampling.stochastic_collocation import SCSampler
import matplotlib.pyplot as plt
import numpy as np

from .machines.machine import Machine
from .programs.program import Program
from .util.constants import QOI, QOIS, RESULTS_DIR, params_type, vary_type
from .util.morris import (
    MorrisScreeningResult,
    _msgpack_default,
    _pack,
    _unpack,
    add_morris_runs_to_campaign,
    create_dir,
    morris_screen,
)
from .util.path import change_dir_permissions, latest_dir, next_dir, next_file
from .wrappers import easy_wrapper


class ExecuteWrapper:
    def __init__(self, function: Callable[[dict[str, Any]], Any]):
        self.function = dill.dumps(function)

    def start(self, previous: dict[str, Any] | None = None) -> dict[str, Any] | None:
        if not previous:
            return None
        rundir = previous.get("rundir")
        old_dir = os.getcwd() if rundir else None
        try:
            if rundir:
                os.chdir(rundir)
            dill.loads(self.function)(previous["run_info"]["params"])
        finally:
            if old_dir:
                os.chdir(old_dir)
        return previous

    def finished(self) -> bool:
        return True

    def finalise(self) -> None:
        pass

    def succeeded(self) -> bool:
        return True


def default_params(
    machine: Machine, active_params: list[str] | None = None
) -> tuple[params_type, vary_type]:
    params: params_type = {
        "N_THREADS": {"type": "integer", "default": machine.max_threads},
        "CLK": {"type": "integer", "default": len(machine.freq) - 1},
        "PLACE_WIDE": {"type": "integer", "default": len(machine.places) - 1},
        "AFF_DISTANCE": {"type": "integer", "default": len(machine.proc_bind) - 1},
        "BOOST": {"type": "integer", "default": len(machine.turbo_boost) - 1},
    }
    all_vary = {
        "N_THREADS": cp.DiscreteUniform(1, machine.max_threads),
        "CLK": cp.DiscreteUniform(0, len(machine.freq) - 1),
        "PLACE_WIDE": cp.DiscreteUniform(0, len(machine.places) - 1),
        "AFF_DISTANCE": cp.DiscreteUniform(0, len(machine.proc_bind) - 1),
        "BOOST": cp.DiscreteUniform(0, len(machine.turbo_boost) - 1),
    }
    active_set = set(active_params) if active_params is not None else set(all_vary.keys())
    vary: vary_type = {k: dist for k, dist in all_vary.items() if k in active_set}
    return params, vary


def energy_wraper_actions(
    program: type[Program], machine: Machine, root: Path
) -> Actions:
    params_def, _ = default_params(machine)
    template = ",".join(f"${param}" for param in params_def)

    Path("easy").mkdir(parents=True, exist_ok=True)
    Path("easy/energy.template").write_text(f"{template}\n")

    encoder = uq.encoders.GenericEncoder(
        template_fname="easy/energy.template", delimiter="$", target_filename="input.csv"
    )
    decoder = uq.decoders.SimpleCSV(target_filename="output.csv", output_columns=QOIS)

    parent_path = root.joinpath("output").absolute()
    create_dir(parent_path)

    def wrapper(params: dict):
        params_str = " and ".join(f"{k} = {v}" for k, v in params.items())
        print(f"Running {program.name} on {machine.name} with {params_str}")
        file_path = next_file(parent_path, program.name)
        with open(file_path.as_posix(), "w") as f, redirect_stdout(f):
            easy_wrapper.main(program, machine)

    return Actions(
        CreateRunDirectory(root=campaign_path(root), flatten=True),
        Encode(encoder),
        ExecuteWrapper(wrapper),
        Decode(decoder),
    )


def campaign_path(root: Path | str) -> Path:
    return Path(root) / "campaign"


def create_campaign(
    program: type[Program],
    machine: Machine,
    root: Path,
    active_params: list[str] | None = None,
) -> uq.Campaign:
    path = campaign_path(root)
    create_dir(path)
    change_dir_permissions(path, 0o755)

    params, vary = default_params(machine, active_params=active_params)
    campaign = uq.Campaign(
        name="energy",
        db_location="sqlite:///" + path.as_posix() + "/campaign.db",
        work_dir=path.as_posix(),
    )
    setattr(campaign, "root_path", root)
    setattr(campaign, "machine", machine)
    if active_params is not None:
        setattr(campaign, "active_params", active_params)

    if campaign.get_active_app() is None:
        campaign.add_app(
            name=campaign.campaign_name,
            params=params,
            actions=energy_wraper_actions(program, machine, root),
        )
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
    program: type[Program],
    machine: Machine,
    root: Path,
    active_params: list[str] | None = None,
    screening_result: MorrisScreeningResult | None = None,
) -> uq.Campaign:
    """
    Creates a campaign, optionally adds Morris screening runs to the database,
    and runs the first execution.
    """
    campaign = create_campaign(program, machine, root, active_params=active_params)
    if screening_result is not None:
        add_morris_runs_to_campaign(campaign, screening_result)
        setattr(campaign, "morris_screening", screening_result)

    campaign.execute(sequential=True).collate(progress_bar=True)
    return campaign


def prepare_analysis(campaign: uq.Campaign) -> uq.analysis.SCAnalysis:
    sampler: SCSampler = cast(SCSampler, campaign.get_active_sampler())
    analysis = uq.analysis.SCAnalysis(sampler=sampler, qoi_cols=[QOI])
    if not hasattr(analysis, "l_norm"):
        analysis.l_norm = sampler.l_norm
    return analysis


def plot_new_points(new_points: list[tuple[float, float]]) -> None:
    pts = np.asarray(new_points)
    plt.figure()
    plt.plot(pts[:, 0], pts[:, 1], "o")
    plt.show()


def get_sampler(campaign: uq.Campaign) -> SCSampler:
    return cast(SCSampler, campaign.get_active_sampler())


def _ordinal(n: int) -> str:
    if 11 <= (n % 100) <= 13:
        return f"{n}th"
    suffix = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def refine_sampling_plan(
    campaign: uq.Campaign,
    analysis: uq.analysis.SCAnalysis,
    start_index: int = 1,
    min_number_of_refinements: int = -1,
    max_number_of_refinements: int = 100,
    surplus_tol: float = 0.1,
    mean_tol: float = 0.1,
    var_tol: float = 0.1,
    patience: int = 2,
    epsilon: float = 1e-12,
    sobol_thresh: float = 1e-3,
    ignored_dims: set[int] | list[int] | None = None,
) -> None:
    sampler = get_sampler(campaign)
    ignored = set(ignored_dims) if ignored_dims is not None else set()

    for d in ignored:
        if d < sampler.N:
            sampler.max_level[d] = 1

    def single_iteration(idx: int) -> bool:
        sampler.look_ahead(analysis.l_norm)
        if len(sampler.admissible_idx) == 0:
            return False

        iter_num = idx + start_index + 1
        print(f"-------{_ordinal(iter_num)} iteration-------")
        print(f"-------{sampler.n_new_points[-1]} new points------")
        print(f"-------Executed {np.sum(sampler.n_new_points)} in total-------")
        campaign.execute(sequential=True).collate(progress_bar=True)
        analysis.adapt_dimension(QOI, campaign.get_collation_result(), method="var")
        return True

    i = 0
    stable_steps = 0

    def _rel_change(history: list[np.ndarray]) -> float:
        delta = np.linalg.norm(history[-1] - history[-2], np.inf)
        norm_prev = np.linalg.norm(history[-2], np.inf)
        return float(delta / (norm_prev + epsilon))

    def convergence_check() -> dict[str, Any]:
        nonlocal stable_steps
        latest_surplus = analysis.adaptation_errors[-1]
        surplus_ok = latest_surplus < (surplus_tol * analysis.std_history[-1])

        rel_diff_mean = _rel_change(analysis.mean_history)
        rel_diff_var = _rel_change(analysis.std_history)

        mean_ok = rel_diff_mean < mean_tol
        var_ok = rel_diff_var < var_tol

        stable_steps = (stable_steps + 1) if (mean_ok and var_ok) else 0

        return {
            "converged": stable_steps >= patience,
            "consecutive_passed": stable_steps,
            "latest_surplus": latest_surplus,
            "surplus_ok": surplus_ok,
            "rel_diff_mean": rel_diff_mean,
            "mean_ok": mean_ok,
            "rel_diff_var": rel_diff_var,
            "var_ok": var_ok,
        }

    def explored_enough(thresh: float = sobol_thresh) -> bool:
        sobols = analysis.get_sobol_indices(QOI)
        max_orders = np.max(analysis.l_norm, 0)
        for dim, order in enumerate(max_orders):
            if dim in ignored:
                continue
            is_significant = any(
                sobol > thresh
                for perm, sobol in sobols.items()
                if (dim + 1) in perm
            )
            if is_significant and order <= 1:
                return False
        return True

    def advance() -> bool:
        nonlocal i
        if not single_iteration(i):
            return False
        i += 1
        return i < max_number_of_refinements

    def is_converged() -> bool:
        check = convergence_check()
        if check["converged"]:
            return True
        thresh = surplus_tol * analysis.std_history[-1]
        print(
            f"Iteration {i:02d} | "
            f"Surplus: {check['latest_surplus']:.3e} (Pass[{thresh}]: {check['surplus_ok']}) | "
            f"Rel Mean Δ: {check['rel_diff_mean']:.3e} | "
            f"Rel Var Δ: {check['rel_diff_var']:.3e} | "
            f"Consecutive Stable: {check['consecutive_passed']}/{patience}"
        )
        return False

    while len(analysis.adaptation_errors) < 3:
        print("Adapt because too few runs")
        if not advance():
            return

    while i < min_number_of_refinements:
        print("Adapt because min_number_of_refinements")
        if not advance():
            return

    while not explored_enough():
        print("Adapt because something was not properly explored")
        if not advance():
            return

    while not is_converged():
        print(f"Adapt because it has not converged yet {analysis.adaptation_errors[-3:]}")
        if not advance():
            return

    print(f"Converged [{analysis.std_history[-1]}]: {analysis.adaptation_errors[-3:]}")


def refine_and_analyse(
    campaign: uq.Campaign,
    analysis: uq.analysis.SCAnalysis,
    min_number_of_refinements: int = -1,
    max_number_of_refinements: int = 100,
    **kwargs,
) -> None:
    refine_sampling_plan(
        campaign,
        analysis,
        min_number_of_refinements=min_number_of_refinements,
        max_number_of_refinements=max_number_of_refinements,
        **kwargs,
    )
    campaign.apply_analysis(analysis)


def run_dir(
    *,
    name: str = "energy",
    dir: str | None = None,
    campaign: uq.Campaign | None = None,
) -> Path:
    if dir:
        return Path(dir)
    if campaign:
        if hasattr(campaign, "root_path"):
            return Path(getattr(campaign, "root_path"))
        app = campaign.get_active_app()
        if app:
            name = app["name"]

    create_dir(RESULTS_DIR)
    return next_dir(RESULTS_DIR, name)


def create(
    program: type[Program],
    machine: Machine,
    /,
    dir: str | None = None,
    active_params: list[str] | None = None,
    screen_morris: bool = True,
    morris_r: int = 4,
    morris_num_levels: int = 4,
    morris_threshold_ratio: float = 0.05,
    evaluate_fn: Callable[[dict[str, int]], dict[str, Any]] | None = None,
    morris_include_dummy: bool = True,
    morris_seed: int | None = None,
) -> tuple[uq.Campaign, uq.analysis.SCAnalysis]:
    root = run_dir(dir=dir)
    create_dir(root)

    screening_result = None
    if screen_morris:
        screening_result = morris_screen(
            program,
            machine,
            r=morris_r,
            num_levels=morris_num_levels,
            threshold_ratio=morris_threshold_ratio,
            evaluate_fn=evaluate_fn,
            include_dummy=morris_include_dummy,
            seed=morris_seed,
            default_params_fn=default_params,
        )
        active_params = screening_result.active_params
        screening_result.save(root)
        try:
            screening_result.plot(root / "morris_screening.png")
        except Exception as e:
            print(f"Warning: Could not save Morris screening plot: {e}")

    campaign = prepare_campaign(
        program,
        machine,
        root,
        active_params=active_params,
        screening_result=screening_result,
    )

    analysis = prepare_analysis(campaign)
    campaign.apply_analysis(analysis)
    return campaign, analysis


def save(
    campaign: uq.Campaign,
    analysis: uq.analysis.SCAnalysis,
    /,
    dir: str | None = None,
    machine: Machine | None = None,
) -> None:
    path = run_dir(dir=dir, campaign=campaign)
    create_dir(path)

    machine_to_save = machine if machine is not None else getattr(campaign, "machine", None)
    if machine_to_save is None:
        raise ValueError("No machine information available to save for this campaign")

    _pack(machine_to_save, path / "machine.msgpack")

    active_params = getattr(campaign, "active_params", None)
    if active_params is not None:
        _pack(active_params, path / "active_params.msgpack")

    analysis.save_state((path / "analysis").as_posix())


def load(
    program: type[Program],
    default_machine: Machine,
    campaign_name: str,
    /,
    dir: str | None = None,
) -> tuple[uq.Campaign, uq.analysis.SCAnalysis, Machine]:
    if not dir:
        path = latest_dir(RESULTS_DIR, campaign_name)
        if path is None:
            raise RuntimeError(
                "No directory was informed and could not find using the default pattern"
            )
    else:
        path = Path(dir)

    machine_path = path / "machine.msgpack"
    if machine_path.exists():
        machine_data = _unpack(machine_path)
        if isinstance(machine_data, dict):
            valid_fields = {f.name for f in fields(Machine)}
            machine = Machine(**{k: v for k, v in machine_data.items() if k in valid_fields})
        elif isinstance(machine_data, Machine):
            machine = machine_data
        else:
            raise RuntimeError(f"machine at {machine_path.as_posix()} is invalid")
    else:
        machine = default_machine

    active_params_path = path / "active_params.msgpack"
    active_params = _unpack(active_params_path) if active_params_path.exists() else None

    campaign = create_campaign(program, machine, path, active_params=active_params)
    setattr(campaign, "machine", machine)

    screening_path = path / "morris_screening.msgpack"
    if screening_path.exists():
        try:
            screening_result = MorrisScreeningResult.load(screening_path)
            setattr(campaign, "morris_screening", screening_result)
        except Exception:
            pass

    analysis = prepare_analysis(campaign)
    analysis.load_state((path / "analysis").as_posix())

    collation = campaign.get_collation_result()
    if collation is not None and not collation.empty:
        campaign.apply_analysis(analysis)

    return campaign, analysis, machine
