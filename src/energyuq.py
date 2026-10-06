import datetime
import json
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any, cast

import chaospy as cp
import easyvvuq as uq
import numpy as np
from easyvvuq.actions import Actions, CreateRunDirectory, Decode, Encode
from easyvvuq.sampling.stochastic_collocation import SCSampler

from src.util.system import pack_dir

from .machines.machine import Machine, load_machine, save_machine
from .programs.program import Program
from .util.constants import QOI, QOIS, RESULTS_DIR, params_type, vary_type
from .util.data import EnergyUQCampaign, ExecuteWrapper
from .util.path import change_dir_permissions, latest_dir, next_dir, next_file
from .wrappers import easy_wrapper


def create_dir(path: Path | str) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def default_params(
    machine: Machine, numa: bool = False
) -> tuple[params_type, vary_type]:
    params: params_type = {
        "N_THREADS": {"type": "integer", "default": machine.max_threads},
        "CLK": {"type": "integer", "default": len(machine.freq) - 1},
        "PLACES": {"type": "integer", "default": len(machine.places) - 1},
        "BINDING": {"type": "integer", "default": len(machine.proc_bind) - 1},
        "BOOST": {"type": "integer", "default": len(machine.turbo_boost) - 1},
    }
    vary: vary_type = {
        "N_THREADS": cp.DiscreteUniform(1, machine.max_threads),
        "CLK": cp.DiscreteUniform(0, len(machine.freq) - 1),
        "PLACES": cp.DiscreteUniform(0, len(machine.places) - 1),
        "BINDING": cp.DiscreteUniform(0, len(machine.proc_bind) - 1),
        "BOOST": cp.DiscreteUniform(0, len(machine.turbo_boost) - 1),
    }
    if numa and machine.has_numa:
        params["NUMA"] = {"type": "integer", "default": len(machine.numactl) - 1}
        vary["NUMA"] = cp.DiscreteUniform(0, len(machine.numactl) - 1)

    return params, vary


def energy_wraper_actions(
    program: type[Program], machine: Machine, root: Path, numa: bool = False,
) -> Actions:
    params_def, _ = default_params(machine, numa=numa)
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
    numa: bool = False,
    qoi: str = QOI,
    qois: list[str] = QOIS,
    params_vary: tuple[params_type,vary_type] | None = None,
    actions: Actions | None = None
) -> EnergyUQCampaign:
    path = campaign_path(root)
    create_dir(path)
    change_dir_permissions(path, 0o755)

    params, vary = default_params(machine, numa=numa) if params_vary is None else params_vary
    campaign = uq.Campaign(
        name="energy",
        db_location="sqlite:///" + path.as_posix() + "/campaign.db",
        work_dir=path.as_posix(),
    )

    if campaign.get_active_app() is None:
        campaign.add_app(
            name=campaign.campaign_name,
            params=params,
            actions=energy_wraper_actions(program, machine, root, numa=numa) if actions is None else actions,
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
        numa=numa,
        qoi=qoi,
        qois=qois,
    )


def prepare_campaign(
    program: type[Program],
    machine: Machine,
    root: Path,
    numa: bool = False,
    qoi: str = QOI,
    qois: list[str] = QOIS,
    params_vary: tuple[params_type,vary_type] | None = None,
    actions: Actions | None = None,
) -> EnergyUQCampaign:
    """
    Creates a campaign and runs the first execution
    """
    campaign = create_campaign(
        program, machine, root, numa=numa, qoi=qoi, qois=qois, params_vary=params_vary, actions=actions
    )

    campaign.execute(sequential=True).collate(progress_bar=True)
    return campaign


def prepare_analysis(campaign: EnergyUQCampaign) -> uq.analysis.SCAnalysis:
    sampler: SCSampler = cast(SCSampler, campaign.get_active_sampler())
    analysis = uq.analysis.SCAnalysis(sampler=sampler, qoi_cols=campaign.qois)
    # if not hasattr(analysis, "l_norm"):
    #     analysis.l_norm = sampler.l_norm
    return analysis


def _ordinal(n: int) -> str:
    if 11 <= (n % 100) <= 13:
        return f"{n}th"
    suffix = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def refine_sampling_plan(
    campaign: EnergyUQCampaign,
    analysis: uq.analysis.SCAnalysis,
    min_number_of_refinements: int = -1,
    max_number_of_refinements: int = 200,
    mean_tol: float = 0.05,
    var_tol: float = 0.05,
    patience: int = 3,
    epsilon: float = 1e-12,
    save_every: int = 2,
    save_dir: Path | str | None = None,
    force_two: bool = False
) -> bool:
    sampler = campaign.sampler

    def ensure_order_two():
        print("Ensuring at least order 2")
        adm = np.array(sampler.admissible_idx)
        if adm.size == 0:
            return

        max_orders = np.max(analysis.l_norm, 0)
        dims = []
        for dim, order in enumerate(max_orders):
            if order < 2:
                dims.append(dim)
        if len(dims) < 1:
            return

        print(f"Dimensions {dims} still not at two")
        force = adm[adm[:, np.array(dims)].max(axis=1) == 2]
        print(f"{force} will be added to l_norm")

        analysis.l_norm = np.unique(np.concatenate((np.asarray(analysis.l_norm), force)), axis=0)
        campaign.apply_analysis(analysis)
        max_orders = np.max(analysis.l_norm, 0)
        dims = []
        for dim, order in enumerate(max_orders):
            if order < 2:
                dims.append(dim)
        if len(dims) < 1:
            return
        print(f"Dimensions {dims} still not two")
        raise

    def single_iteration(idx: int) -> bool:
        anouce_run(idx)
        sampler.look_ahead(analysis.l_norm)
        if len(sampler.admissible_idx) == 0:
            return False
        campaign.execute(sequential=True).collate(progress_bar=True)
        analysis.adapt_dimension(campaign.qoi, campaign.get_collation_result(), method="var")

        nonlocal stable_steps
        stable_steps = stable_steps + 1 if settling() else 0
        anouce_end_run()
        report()
        return True

    i = 0
    stable_steps = 0

    def _rel_change(history: list[np.ndarray]) -> float:
        delta = np.linalg.norm(history[-1] - history[-2], np.inf)
        norm_prev = np.linalg.norm(history[-2], np.inf)
        return float(delta / (norm_prev + epsilon))

    def anouce_run(idx):
        why = (
            "too few runs"
            if len(analysis.adaptation_errors) < 3
            else "min refinements"
            if idx < min_number_of_refinements
            else "not converged"
        )
        print(f"\n{_ordinal(len(analysis.adaptation_errors) + 1)} iteration, {why}\n")

    def anouce_end_run():
        print(f"\n{_ordinal(len(analysis.adaptation_errors))} iteration, ended\n")

    def deltas():
        d_mean = _rel_change(analysis.mean_history)
        d_var = _rel_change(analysis.std_history)
        return d_mean, d_var
        

    def settling() -> bool:
        if len(analysis.adaptation_errors) < 3: return False
        d_mean, d_var = deltas()
        settling = d_mean < mean_tol and d_var < var_tol
        return settling

    def report():
        if len(analysis.adaptation_errors) < 3: return
        d_mean, d_var = deltas()

        mark = "ok" if settling() else "--"
        print(
            f"  samples   {len(campaign.get_collation_result()):,} run, "
            f"+{sampler.n_new_points[-1]} planned, {np.sum(sampler.n_new_points):,} total\n"
            f"  changes   mean {d_mean:.2e}, var {d_var:.2e}  [{mark}]\n"
            f"  progress  stable {stable_steps}/{patience}, "
            f"{len(analysis.l_norm)} PCE terms at level {sampler.L}, "
            f"orders [{' '.join(str(o) for o in np.max(analysis.l_norm, 0))}]"
        )

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
        return stable_steps >= patience

    while len(analysis.adaptation_errors) < 3:
        if not advance_and_save():
            return is_converged()

    while i < min_number_of_refinements:
        if not advance_and_save():
            return is_converged()

    if force_two:
        ensure_order_two()

    while not is_converged():
        if not advance_and_save():
            print("Ran out of space to explore")
            return is_converged()

    if is_converged():
        print("Converged!!!")
    report()
    save(campaign, analysis, dir=save_dir, status="converged", converged=is_converged())
    return is_converged()


def refine_and_analyse(
    campaign: EnergyUQCampaign,
    analysis: uq.analysis.SCAnalysis,
    min_number_of_refinements: int = -1,
    max_number_of_refinements: int = 100,
    save_every: int = 2,
    save_dir: Path | str | None = None,
    **kwargs,
) -> None:
    converged = refine_sampling_plan(
        campaign,
        analysis,
        min_number_of_refinements=min_number_of_refinements,
        max_number_of_refinements=max_number_of_refinements,
        save_every=save_every,
        save_dir=save_dir,
        **kwargs,
    )
    campaign.apply_analysis(analysis)
    save(campaign, analysis, dir=save_dir, status="completed", converged=converged)


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
    resume: bool = False,
    numa: bool = False,
    qoi: str = QOI,
    qois: list[str] = QOIS,
    params_vary: tuple[params_type,vary_type] | None = None,
    actions: Actions | None = None,
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
            raise RuntimeError("No existing campaign found to resume.")

    root = run_dir(dir=dir)
    create_dir(root)

    if numa and not machine.has_numa:
        print("Warning: NUMA balancing not available on this machine.")

    use_numa = numa and machine.has_numa

    campaign = prepare_campaign(
        program,
        machine,
        root,
        numa=use_numa,
        qoi=qoi,
        qois=qois,
        params_vary=params_vary,
        actions=actions,
    )

    analysis = prepare_analysis(campaign)
    campaign.apply_analysis(analysis)
    save(campaign, analysis, dir=root.as_posix(), machine=machine, status="in_progress")
    return campaign, analysis


def save(
    campaign: EnergyUQCampaign,
    analysis: uq.analysis.SCAnalysis,
    /,
    name: str | None = None,
    dir: str | Path | None = None,
    machine: Machine | None = None,
    pack: bool = False,
    status: str = "in_progress",
    converged: bool = False,
) -> Path:
    path = run_dir(name="energy" if name is None else name, dir=str(dir) if dir is not None else None, campaign=campaign)
    create_dir(path)

    machine_to_save = machine if machine is not None else getattr(campaign, "machine", None)
    if machine_to_save is not None:
        save_machine(machine_to_save, path)
    elif not (path / "machine.msgpack").exists() and not (path / "machine.pkl").exists():
        raise ValueError("No machine information available to save for this campaign")

    (path / "numa").write_text("1" if campaign.numa else "0")
    (path / "qoi").write_text(f"{campaign.qoi}:{','.join(campaign.qois)}")
 
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


    if pack:
        pack_dir(path)

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

    numa_path = path / "numa"
    numa = numa_path.is_file() and numa_path.read_text() == "1"
    qoi_path = path / "qoi"
    if qoi_path.is_file():
        raw = qoi_path.read_text().split(":")
        qoi = raw[0]
        qois = ":".join(raw[1:]).split(",")
    else:
        qoi, qois = QOI, QOIS

    campaign = create_campaign(
        program, machine, path, numa=numa, qoi=qoi, qois=qois
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
