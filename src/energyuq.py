from __future__ import annotations

import datetime
import json
import os
from collections.abc import Callable
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any, cast

import chaospy as cp
import dill
import easyvvuq as uq
import matplotlib.pyplot as plt
import numpy as np
from easyvvuq.actions import Actions, CreateRunDirectory, Decode, Encode
from easyvvuq.sampling.stochastic_collocation import SCSampler

from .machines.machine import Machine, load_machine, save_machine
from .morris import (
    MorrisScreeningResult,
    _pack,
    _unpack,
    add_morris_runs_to_campaign,
    create_dir,
    morris_screen,
)
from .programs.program import Program
from .util.constants import QOI, QOIS, RESULTS_DIR, params_type, vary_type
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


class EnergyUQCampaign:
    """Encapsulates an EasyVVUQ Campaign along with EnergyUQ metadata:
    root directory, machine profile, screening results, and active parameters.
    """

    def __init__(
        self,
        campaign: uq.Campaign,
        root_path: Path | str,
        machine: Machine,
        screening_result: MorrisScreeningResult | None = None,
        active_params: list[str] | None = None,
    ):
        self.campaign: uq.Campaign = campaign
        self.root_path = Path(root_path)
        self.machine: Machine = machine
        self.screening_result = screening_result
        self.active_params = list(active_params) if active_params is not None else []

    @property
    def sampler(self) -> SCSampler:
        return cast(SCSampler, self.campaign.get_active_sampler())

    def __getattr__(self, name: str) -> Any:
        return getattr(self.campaign, name)

    def __repr__(self) -> str:
        camp_name = getattr(self.campaign, "campaign_name", None)
        return (
            f"EnergyUQCampaign(name={camp_name!r}, "
            f"root_path={self.root_path}, machine={self.machine.name}, "
            f"screening_result={'yes' if self.screening_result else 'no'})"
        )


def default_params(
    machine: Machine, active_params: list[str] | None = None, numa: bool = False
) -> tuple[params_type, vary_type]:
    params: params_type = {
        "N_THREADS": {"type": "integer", "default": machine.max_threads},
        "CLK": {"type": "integer", "default": len(machine.freq) - 1},
        "PLACES": {"type": "integer", "default": len(machine.places) - 1},
        "BINDING": {"type": "integer", "default": len(machine.proc_bind) - 1},
        "BOOST": {"type": "integer", "default": len(machine.turbo_boost) - 1},
    }
    all_vary = {
        "N_THREADS": cp.DiscreteUniform(1, machine.max_threads),
        "CLK": cp.DiscreteUniform(0, len(machine.freq) - 1),
        "PLACES": cp.DiscreteUniform(0, len(machine.places) - 1),
        "BINDING": cp.DiscreteUniform(0, len(machine.proc_bind) - 1),
        "BOOST": cp.DiscreteUniform(0, len(machine.turbo_boost) - 1),
    }
    if (numa or (active_params is not None and "NUMA" in active_params)) and machine.has_numa:
        params["NUMA"] = {"type": "integer", "default": len(machine.numactl) - 1}
        all_vary["NUMA"] = cp.DiscreteUniform(0, len(machine.numactl) - 1)

    active_set = set(active_params) if active_params is not None else set(all_vary.keys())
    vary: vary_type = {k: dist for k, dist in all_vary.items() if k in active_set}
    return params, vary


def energy_wraper_actions(
    program: type[Program], machine: Machine, root: Path, numa: bool = False, active_params: list[str] | None = None
) -> Actions:
    params_def, _ = default_params(machine, active_params=active_params, numa=numa)
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
    p = Path(root)
    if (p / "campaign.db").exists():
        return p
    return p / "campaign"


def create_campaign(
    program: type[Program],
    machine: Machine,
    root: Path,
    active_params: list[str] | None = None,
    numa: bool = False,
    screening_result: MorrisScreeningResult | None = None,
) -> EnergyUQCampaign:
    path = campaign_path(root)
    create_dir(path)
    change_dir_permissions(path, 0o755)

    params, vary = default_params(machine, active_params=active_params, numa=numa)
    campaign = uq.Campaign(
        name="energy",
        db_location="sqlite:///" + path.as_posix() + "/campaign.db",
        work_dir=path.as_posix(),
    )

    if campaign.get_active_app() is None:
        campaign.add_app(
            name=campaign.campaign_name,
            params=params,
            actions=energy_wraper_actions(program, machine, root, numa=numa, active_params=active_params),
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

    return EnergyUQCampaign(
        campaign=campaign,
        root_path=root,
        machine=machine,
        screening_result=screening_result,
        active_params=active_params if active_params is not None else list(vary.keys()),
    )


def prepare_campaign(
    program: type[Program],
    machine: Machine,
    root: Path,
    active_params: list[str] | None = None,
    screening_result: MorrisScreeningResult | None = None,
    numa: bool = False,
) -> EnergyUQCampaign:
    """
    Creates a campaign, optionally adds Morris screening runs to the database,
    and runs the first execution.
    """
    campaign = create_campaign(
        program, machine, root, active_params=active_params, numa=numa, screening_result=screening_result
    )
    if screening_result is not None:
        add_morris_runs_to_campaign(campaign, screening_result)

    campaign.execute(sequential=True).collate(progress_bar=True)
    return campaign


def prepare_analysis(campaign: EnergyUQCampaign) -> uq.analysis.SCAnalysis:
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


def _ordinal(n: int) -> str:
    if 11 <= (n % 100) <= 13:
        return f"{n}th"
    suffix = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def refine_sampling_plan(
    campaign: EnergyUQCampaign,
    analysis: uq.analysis.SCAnalysis,
    start_index: int | None = None,
    min_number_of_refinements: int = -1,
    max_number_of_refinements: int = 100,
    surplus_tol: float = 0.1,
    mean_tol: float = 0.1,
    var_tol: float = 0.1,
    patience: int = 2,
    epsilon: float = 1e-12,
    sobol_thresh: float = 1e-3,
    ignored_dims: set[int] | list[int] | None = None,
    save_every: int = 2,
    save_dir: Path | str | None = None,
) -> None:
    sampler = campaign.sampler
    ignored = set(ignored_dims) if ignored_dims is not None else set()

    for d in ignored:
        if d < sampler.N:
            sampler.max_level[d] = 1

    effective_start = len(analysis.adaptation_errors) if start_index is None else (start_index - 1)

    def single_iteration(idx: int) -> bool:
        sampler.look_ahead(analysis.l_norm)
        if len(sampler.admissible_idx) == 0:
            return False

        iter_num = idx + effective_start + 2
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
                if dim in perm
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

    def advance_and_save() -> bool:
        continue_advancing = advance()
        if not continue_advancing:
            save(campaign, analysis, dir=save_dir, status="completed", converged=False)
            return False
        total_adaptations = len(analysis.adaptation_errors)
        if save_every > 0 and (total_adaptations % save_every == 0):
            print(f"[Checkpoint] Periodic save at iteration {total_adaptations} (every {save_every} iterations)...")
            save(campaign, analysis, dir=save_dir, status="in_progress")
        return True

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
        if not advance_and_save():
            return

    while i < min_number_of_refinements:
        print("Adapt because min_number_of_refinements")
        if not advance_and_save():
            return

    while not explored_enough():
        print("Adapt because something was not properly explored")
        if not advance_and_save():
            return

    while not is_converged():
        print(f"Adapt because it has not converged yet {analysis.adaptation_errors[-3:]}")
        if not advance_and_save():
            return

    print(f"Converged [{analysis.std_history[-1]}]: {analysis.adaptation_errors[-3:]}")
    save(campaign, analysis, dir=save_dir, status="converged", converged=True)


def refine_and_analyse(
    campaign: EnergyUQCampaign,
    analysis: uq.analysis.SCAnalysis,
    min_number_of_refinements: int = -1,
    max_number_of_refinements: int = 100,
    save_every: int = 2,
    save_dir: Path | str | None = None,
    **kwargs,
) -> None:
    refine_sampling_plan(
        campaign,
        analysis,
        min_number_of_refinements=min_number_of_refinements,
        max_number_of_refinements=max_number_of_refinements,
        save_every=save_every,
        save_dir=save_dir,
        **kwargs,
    )
    campaign.apply_analysis(analysis)
    save(campaign, analysis, dir=save_dir, status="completed")


def run_dir(
    *,
    name: str = "energy",
    dir: str | None = None,
    campaign: EnergyUQCampaign | None = None,
) -> Path:
    if dir:
        return Path(dir)
    if campaign:
        if hasattr(campaign, "root_path"):
            return Path(campaign.root_path)
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
    resume: bool = False,
    numa: bool = False,
) -> tuple[EnergyUQCampaign, uq.analysis.SCAnalysis]:
    if resume:
        target_dir = Path(dir) if dir else latest_dir(RESULTS_DIR, "energy")
        if target_dir is not None and (
            (target_dir / "campaign" / "campaign.db").exists()
            or (target_dir / "campaign.db").exists()
            or (target_dir / "analysis").exists()
        ):
            print(f"Resuming campaign from {target_dir.as_posix()}...")
            c, a, _ = load(program, machine, "energy", dir=target_dir)
            return c, a
        elif dir is not None:
            raise FileNotFoundError(f"Cannot resume: checkpoint directory '{dir}' not found or invalid")
        else:
            print("No existing campaign found to resume. Starting fresh campaign...")

    root = run_dir(dir=dir)
    create_dir(root)

    if numa and not machine.has_numa:
        print("Warning: NUMA balancing requested but not available on this machine (has <= 1 NUMA node).")

    use_numa = (numa or (active_params is not None and "NUMA" in active_params)) and machine.has_numa

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
            default_params_fn=lambda m: default_params(m, numa=use_numa),
        )
        active_params = screening_result.active_params
        screening_result.save(root)
        screening_result.plot(root / "morris_screening.png")

    campaign = prepare_campaign(
        program,
        machine,
        root,
        active_params=active_params,
        screening_result=screening_result,
        numa=use_numa,
    )

    analysis = prepare_analysis(campaign)
    campaign.apply_analysis(analysis)
    save(campaign, analysis, dir=root.as_posix(), machine=machine, status="in_progress")
    return campaign, analysis


def save(
    campaign: EnergyUQCampaign,
    analysis: uq.analysis.SCAnalysis,
    /,
    dir: str | Path | None = None,
    machine: Machine | None = None,
    status: str = "in_progress",
    converged: bool = False,
) -> Path:
    path = run_dir(dir=str(dir) if dir is not None else None, campaign=campaign)
    create_dir(path)

    machine_to_save = machine if machine is not None else getattr(campaign, "machine", None)
    if machine_to_save is not None:
        save_machine(machine_to_save, path)
    elif not (path / "machine.msgpack").exists() and not (path / "machine.pkl").exists():
        raise ValueError("No machine information available to save for this campaign")

    active_params = getattr(campaign, "active_params", None)
    if active_params is not None:
        _pack(active_params, path / "active_params.msgpack")

    analysis.save_state((path / "analysis").as_posix())

    sampler = campaign.sampler
    if sampler is not None:
        sampler.save_state((path / "sampler").as_posix())

    total_samples = 0
    collation = campaign.get_collation_result()
    if collation is not None and not collation.empty:
        total_samples = len(collation)

    latest_surplus = None
    if hasattr(analysis, "adaptation_errors") and len(analysis.adaptation_errors) > 0:
        latest_surplus = float(analysis.adaptation_errors[-1])

    checkpoint_data = {
        "iteration": len(getattr(analysis, "adaptation_errors", [])),
        "total_samples": total_samples,
        "status": status,
        "converged": converged,
        "latest_surplus": latest_surplus,
        "timestamp": datetime.datetime.now().isoformat(),
    }
    with open(path / "checkpoint.json", "w", encoding="utf-8") as f:
        json.dump(checkpoint_data, f, indent=2)

    return path


def load(
    program: type[Program],
    default_machine: Machine,
    campaign_name: str = "energy",
    /,
    dir: str | Path | None = None,
) -> tuple[EnergyUQCampaign, uq.analysis.SCAnalysis, Machine]:
    if not dir:
        path = latest_dir(RESULTS_DIR, campaign_name)
        if path is None:
            raise RuntimeError(
                "No directory was informed and could not find using the default pattern"
            )
    else:
        path = Path(dir)

    loaded_machine = load_machine(path)
    machine = loaded_machine if loaded_machine is not None else default_machine

    active_params_path = path / "active_params.msgpack"
    active_params = _unpack(active_params_path) if active_params_path.exists() else None

    screening_result = None
    screening_path = path / "morris_screening.msgpack"
    if screening_path.exists():
        screening_result = MorrisScreeningResult.load(screening_path)

    campaign = create_campaign(
        program, machine, path, active_params=active_params, screening_result=screening_result
    )

    sampler_path = path / "sampler"
    if sampler_path.exists():
        sampler = campaign.sampler
        if sampler is not None:
            sampler.load_state(sampler_path.as_posix())

    analysis = prepare_analysis(campaign)
    analysis_path = path / "analysis"
    if analysis_path.exists():
        analysis.load_state(analysis_path.as_posix())

    collation = campaign.get_collation_result()
    if collation is not None and not collation.empty:
        campaign.apply_analysis(analysis)

    return campaign, analysis, machine


def resume(
    program: type[Program],
    default_machine: Machine,
    campaign_name: str = "energy",
    /,
    dir: str | Path | None = None,
) -> tuple[EnergyUQCampaign, uq.analysis.SCAnalysis]:
    campaign, analysis, _ = load(program, default_machine, campaign_name, dir=dir)
    return campaign, analysis
