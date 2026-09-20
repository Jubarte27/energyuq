from collections.abc import Callable
from dataclasses import asdict, dataclass, field, fields, is_dataclass
import json
from math import ceil, floor
import os
from pathlib import Path
from typing import Any

import chaospy as cp
import easyvvuq as uq
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import matplotlib.pyplot as plt
import msgpack
import numpy as np
from SALib.analyze import morris as morris_analyzer
from SALib.sample import morris as morris_sampler

from ..machines.machine import Machine
from ..programs.program import Program
from ..wrappers import base_wrapper
from .constants import QOI, QOIS, params_type, vary_type
from .data import ExecutionParams


def create_dir(path: Path | str) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _msgpack_default(obj: Any) -> Any:
    if is_dataclass(obj) and not isinstance(obj, type):
        return asdict(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (set, tuple)):
        return list(obj)
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Type {type(obj)} not serializable by msgpack")


def _pack(obj: Any, path: Path) -> None:
    with path.open("wb") as f:
        msgpack.pack(obj, f, default=_msgpack_default)


def _unpack(path: Path) -> Any:
    with path.open("rb") as f:
        return msgpack.unpack(f)


def _normalize_output(res: Any, qoi: str = QOI) -> dict[str, float]:
    if isinstance(res, dict):
        out = {k: float(v) for k, v in res.items() if isinstance(v, (int, float, np.number))}
        if qoi not in out and out:
            out[qoi] = float(next(iter(out.values())))
    elif isinstance(res, (int, float, np.number)):
        out = {qoi: float(res)}
    else:
        out = {qoi: 0.0}
    if "EDP" not in out and "energy_uj" in out and "time" in out:
        out["EDP"] = float((out["energy_uj"] * 1e-6) * out["time"])
    for col in QOIS:
        out.setdefault(col, 1.0)
    return out


def _dist_bounds(dist: cp.Distribution) -> list[int]:
    low = ceil(dist.lower[0])
    high = floor(dist.upper[0])
    return [int(low), int(high)]


def _format_param_value(machine: Machine, param_name: str, def_idx: int) -> str:
    if param_name == "N_THREADS":
        return f"{def_idx} threads"
    if param_name == "CLK" and 0 <= def_idx < len(machine.freq):
        return f"{machine.freq[def_idx]} Hz (idx {def_idx})"
    if param_name == "PLACES" and 0 <= def_idx < len(machine.places):
        return f"{machine.places[def_idx]} (idx {def_idx})"
    if param_name == "BINDING" and 0 <= def_idx < len(machine.proc_bind):
        return f"{machine.proc_bind[def_idx]} (idx {def_idx})"
    if param_name == "BOOST" and 0 <= def_idx < len(machine.turbo_boost):
        return f"{machine.turbo_boost[def_idx]} (idx {def_idx})"
    if param_name == "NUMA" and 0 <= def_idx < len(machine.numactl):
        return f"{machine.numactl[def_idx]} (idx {def_idx})"
    return str(def_idx)


@dataclass
class MorrisScreeningResult:
    active_params: list[str]
    ignored_params: list[str]
    mu_star: dict[str, float]
    sigma: dict[str, float]
    sample_points: list[dict[str, int]]
    outputs: list[float]
    threshold_ratio: float = 0.05
    frozen_details: dict[str, dict[str, Any]] = field(default_factory=dict)
    output_dicts: list[dict[str, Any]] = field(default_factory=list)
    mu_star_conf: dict[str, float] = field(default_factory=dict)
    mu_star_lower: dict[str, float] = field(default_factory=dict)
    dummy_stats: dict[str, Any] = field(default_factory=dict)

    @property
    def frozen_params(self) -> list[str]:
        """Alias for ignored_params: the dimensions frozen to default values."""
        return self.ignored_params

    def to_dict(self) -> dict[str, Any]:
        return {
            "active_params": self.active_params,
            "ignored_params": self.ignored_params,
            "frozen_params": self.frozen_params,
            "frozen_details": self.frozen_details,
            "mu_star": self.mu_star,
            "sigma": self.sigma,
            "mu_star_conf": self.mu_star_conf,
            "mu_star_lower": self.mu_star_lower,
            "dummy_stats": self.dummy_stats,
            "threshold_ratio": self.threshold_ratio,
            "num_samples": len(self.outputs),
            "output_dicts": self.output_dicts,
        }

    def summary(self) -> str:
        lines = [
            "=" * 80,
            "[Morris Screening Complete] Parameter Sensitivity & Freezing Details",
            "=" * 80,
        ]
        max_mu = max(self.mu_star.values()) if self.mu_star else 0.0
        dominant = max(self.mu_star.keys(), key=lambda k: self.mu_star[k]) if self.mu_star else "None"
        total_dims = len(self.active_params) + len(self.ignored_params)
        lines.append(f"Screening Threshold: {self.threshold_ratio:.1%} of max(mu*) [Lower 95% CI rule] | Dominant: {dominant} (mu*={max_mu:.2e})")
        lines.append(f"Dimension Reduction: {total_dims}D -> {len(self.active_params)}D ({len(self.ignored_params)} dimensions frozen)")
        lines.append("")
        lines.append("ACTIVE PARAMETERS (included in Stochastic Collocation sparse grid):")
        for name in self.active_params:
            m = self.mu_star.get(name, 0.0)
            s = self.sigma.get(name, 0.0)
            conf = self.mu_star_conf.get(name, 0.0)
            m_lower = self.mu_star_lower.get(name, m)
            rel = (m / max_mu * 100) if max_mu > 0 else 0.0
            rel_lower = (m_lower / max_mu * 100) if max_mu > 0 else 0.0
            lines.append(
                f"  * {name:<15s}: mu* = {m:.2e} +/- {conf:.2e} (lower: {rel_lower:5.1f}%, mean: {rel:5.1f}%) | sigma = {s:.2e} | [ACTIVE]"
            )
        lines.append("")
        lines.append("FROZEN PARAMETERS (omitted from sampling, fixed to machine defaults):")
        if not self.ignored_params:
            lines.append("  (None - all parameters are active)")
        else:
            for name in self.ignored_params:
                d = self.frozen_details.get(name, {})
                m = self.mu_star.get(name, 0.0)
                s = self.sigma.get(name, 0.0)
                conf = self.mu_star_conf.get(name, 0.0)
                m_lower = self.mu_star_lower.get(name, 0.0)
                rel = (m / max_mu * 100) if max_mu > 0 else 0.0
                rel_lower = (m_lower / max_mu * 100) if max_mu > 0 else 0.0
                fixed_val = d.get("fixed_default_repr", str(d.get("fixed_default_index", "default")))
                reason = d.get("reason", "lower CI below threshold")
                lines.append(
                    f"  * {name:<15s}: mu* = {m:.2e} +/- {conf:.2e} (lower: {rel_lower:5.1f}%) | fixed to: {fixed_val:<22s} | reason: {reason}"
                )
        if self.dummy_stats:
            d_name = self.dummy_stats.get("name", "DUMMY")
            d_m = self.dummy_stats.get("mu_star", 0.0)
            d_s = self.dummy_stats.get("sigma", 0.0)
            d_conf = self.dummy_stats.get("mu_star_conf", 0.0)
            d_rel = (d_m / max_mu * 100) if max_mu > 0 else 0.0
            lines.append("")
            lines.append("SYSTEM NOISE BENCHMARK (Dummy Variable - human visual double-check):")
            lines.append(
                f"  * {d_name:<15s}: mu* = {d_m:.2e} +/- {d_conf:.2e} (mean: {d_rel:5.1f}% of dominant) | sigma = {d_s:.2e} | [NOISE FLOOR REFERENCE]"
            )
        lines.append("=" * 80)
        return "\n".join(lines)

    def save(self, path: Path | str) -> None:
        p = Path(path)
        target = p if p.suffix else p / "morris_screening.msgpack"
        create_dir(target.parent)
        _pack(self.to_dict(), target)

    @classmethod
    def load(cls, path: Path | str) -> "MorrisScreeningResult":
        p = Path(path)
        file_path = p if p.is_file() else p / "morris_screening.msgpack"
        data = _unpack(file_path)
        assert isinstance(data, dict)
        valid_fields = {f.name for f in fields(cls)}
        kwargs = {k: v for k, v in data.items() if k in valid_fields}
        kwargs.setdefault("active_params", [])
        kwargs.setdefault("ignored_params", [])
        kwargs.setdefault("mu_star", {})
        kwargs.setdefault("sigma", {})
        kwargs.setdefault("sample_points", [])
        kwargs.setdefault("outputs", [])
        kwargs.setdefault("threshold_ratio", 0.05)
        return cls(**kwargs)

    def plot(self, save_path: Path | str | None = None) -> None:
        fig, ax = plt.subplots(figsize=(9, 5))
        base_params = [k for k in self.mu_star if k != "DUMMY"]
        max_mu = max((self.mu_star[k] for k in base_params), default=0.0)

        plot_names = list(base_params)
        mu_vals = [self.mu_star.get(k, 0.0) for k in base_params]
        err_vals = [self.mu_star_conf.get(k, self.sigma.get(k, 0.0)) for k in base_params]
        bar_colors = ["cornflowerblue" if k in self.active_params else "lightgray" for k in base_params]
        bar_edgecolors = ["black" if k in self.active_params else "gray" for k in base_params]

        d_stats = self.dummy_stats or (
            {"name": "DUMMY", "mu_star": self.mu_star["DUMMY"], "sigma": self.sigma["DUMMY"], "mu_star_conf": self.mu_star_conf.get("DUMMY", 0.0)}
            if "DUMMY" in self.mu_star else None
        )
        if d_stats:
            plot_names.append(f"{d_stats.get('name', 'DUMMY')}\n(noise)")
            mu_vals.append(d_stats.get("mu_star", 0.0))
            err_vals.append(d_stats.get("mu_star_conf", d_stats.get("sigma", 0.0)))
            bar_colors.append("sandybrown")
            bar_edgecolors.append("darkorange")

        x = np.arange(len(plot_names))
        ax.bar(x, mu_vals, yerr=err_vals, capsize=5, color=bar_colors, edgecolor=bar_edgecolors)
        ax.set_xticks(x)
        ax.set_xticklabels(plot_names)
        ax.set_ylabel(r"$\mu^*$ with 95% CI")
        ax.set_title("Morris Screening: Parameter Sensitivity vs System Variation (95% CI)")

        if max_mu > 0 and self.threshold_ratio > 0:
            ax.axhline(self.threshold_ratio * max_mu, color="crimson", linestyle="--", alpha=0.7)

        legend_elements: list[Patch | Line2D] = [
            Patch(facecolor="cornflowerblue", edgecolor="black", label="Active Parameter"),
            Patch(facecolor="lightgray", edgecolor="gray", label="Frozen Parameter"),
        ]
        if d_stats:
            legend_elements.append(
                Patch(facecolor="sandybrown", edgecolor="darkorange", label="Dummy (System Noise Floor)")
            )
        if max_mu > 0 and self.threshold_ratio > 0:
            legend_elements.append(
                Line2D([0], [0], color="crimson", linestyle="--", label=f"Threshold ({self.threshold_ratio:.0%})")
            )

        ax.legend(handles=legend_elements, loc="upper right")
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
            plt.close(fig)
        else:
            plt.show()


def morris_screen(
    program: type[Program],
    machine: Machine,
    r: int = 4,
    num_levels: int = 4,
    threshold_ratio: float = 0.05,
    qoi: str = QOI,
    evaluate_fn: Callable[[dict[str, int]], dict[str, Any]] | None = None,
    include_dummy: bool = True,
    dummy_name: str = "DUMMY",
    seed: int | None = None,
    default_params_fn: Callable[..., tuple[params_type, vary_type]] | None = None,
) -> MorrisScreeningResult:
    if default_params_fn is None:
        from ..energyuq import default_params
        default_params_fn = default_params

    defaults_dict, all_vary = default_params_fn(machine)
    param_names = list(defaults_dict.keys())
    all_bounds = [_dist_bounds(all_vary[name]) for name in param_names]

    var_indices = [i for i, b in enumerate(all_bounds) if b[1] > b[0]]
    const_indices = [i for i, b in enumerate(all_bounds) if b[1] <= b[0]]

    zero_stats = {param_names[i]: 0.0 for i in const_indices}
    mu_star = dict(zero_stats)
    sigma = dict(zero_stats)
    mu_star_conf = dict(zero_stats)
    mu_star_lower = dict(zero_stats)
    dummy_stats: dict[str, Any] = {}

    if not var_indices:
        if include_dummy:
            dummy_stats = {
                "name": dummy_name,
                "mu_star": 0.0,
                "sigma": 0.0,
                "mu_star_conf": 0.0,
                "mu_star_lower": 0.0,
            }
        return MorrisScreeningResult(
            active_params=param_names[:1],
            ignored_params=param_names[1:],
            mu_star=mu_star,
            sigma=sigma,
            mu_star_conf=mu_star_conf,
            mu_star_lower=mu_star_lower,
            dummy_stats=dummy_stats,
            sample_points=[],
            outputs=[],
            threshold_ratio=threshold_ratio,
        )

    var_names = [param_names[i] for i in var_indices]
    var_bounds = [all_bounds[i] for i in var_indices]

    salib_names = list(var_names)
    salib_bounds = [list(b) for b in var_bounds]
    if include_dummy:
        salib_names.append(dummy_name)
        salib_bounds.append([0, num_levels - 1])

    problem = {"num_vars": len(salib_names), "names": salib_names, "bounds": salib_bounds}
    sample_kwargs = {"seed": seed} if seed is not None else {}
    X_var = morris_sampler.sample(problem, N=r, num_levels=num_levels, local_optimization=True, **sample_kwargs)
    X_var_int = np.round(X_var).astype(int)

    sample_points_list: list[dict[str, int]] = []
    dummy_values: list[int] = []
    for row in range(len(X_var_int)):
        pt = {param_names[v_idx]: int(X_var_int[row, c_idx]) for c_idx, v_idx in enumerate(var_indices)}
        for c_idx in const_indices:
            pt[param_names[c_idx]] = int(all_bounds[c_idx][0])
        sample_points_list.append(pt)
        if include_dummy:
            dummy_values.append(int(X_var_int[row, len(var_indices)]))

    if evaluate_fn is None:
        def default_eval(point: dict[str, int]) -> dict[str, Any]:
            execution_params = ExecutionParams(
                machine=machine,
                n_threads=point["N_THREADS"],
                freq_level=point["CLK"],
                place_wideness=point["PLACES"],
                binding=point["BINDING"],
                boost=point["BOOST"],
                numa=point.get("NUMA"),
            )
            return base_wrapper.prepare_and_execute(machine, program, execution_params, [])
        evaluate_fn = default_eval

    y_vals: list[float] = []
    output_dicts: list[dict[str, Any]] = []
    for i, pt in enumerate(sample_points_list):
        dummy_str = f" [dummy={dummy_values[i]}]" if include_dummy else ""
        print(f"[Morris Screening {i + 1}/{len(sample_points_list)}] Evaluating {pt}{dummy_str}", flush=i % 4 == 0)
        res_dict = _normalize_output(evaluate_fn(pt), qoi)
        output_dicts.append(res_dict)
        y_vals.append(float(res_dict.get(qoi, 1.0)))

    res_salib = morris_analyzer.analyze(
        problem,
        X_var_int.astype(float),
        np.array(y_vals, dtype=float),
        conf_level=0.95,
        print_to_console=False,
    )

    salib_conf = res_salib.get("mu_star_conf", [0.0] * len(salib_names))
    for name, m, s, conf in zip(salib_names, res_salib["mu_star"], res_salib["sigma"], salib_conf):
        conf_val = float(conf) if (conf is not None and not np.isnan(conf)) else 0.0
        m_val, s_val = float(m), float(s)
        lower_val = max(0.0, m_val - conf_val)

        if include_dummy and name == dummy_name:
            dummy_stats = {
                "name": dummy_name,
                "mu_star": m_val,
                "sigma": s_val,
                "mu_star_conf": conf_val,
                "mu_star_lower": lower_val,
            }
        else:
            mu_star[name] = m_val
            sigma[name] = s_val
            mu_star_conf[name] = conf_val
            mu_star_lower[name] = lower_val

    max_mu = max((mu_star[name] for name in param_names), default=0.0)
    const_names = {param_names[c] for c in const_indices}

    active_params: list[str] = []
    ignored_params: list[str] = []
    for name in param_names:
        m_lower = mu_star_lower.get(name, 0.0)
        if max_mu > 0 and (m_lower / max_mu) < threshold_ratio:
            ignored_params.append(name)
        elif max_mu == 0.0 and name in const_names:
            ignored_params.append(name)
        else:
            active_params.append(name)

    if not active_params:
        active_params = [param_names[0]]
        if param_names[0] in ignored_params:
            ignored_params.remove(param_names[0])

    frozen_details: dict[str, dict[str, Any]] = {}
    for name in ignored_params:
        m = mu_star.get(name, 0.0)
        s = sigma.get(name, 0.0)
        conf = mu_star_conf.get(name, 0.0)
        m_lower = mu_star_lower.get(name, 0.0)
        def_idx = defaults_dict[name]["default"]

        if name in const_names:
            reason = f"constant bound on machine ({all_bounds[param_names.index(name)]})"
        else:
            rel_pct = (m / max_mu * 100) if max_mu > 0 else 0.0
            rel_lower_pct = (m_lower / max_mu * 100) if max_mu > 0 else 0.0
            if rel_pct >= threshold_ratio * 100:
                reason = (
                    f"borderline mean ({rel_pct:.1f}%) but lower 95% CI "
                    f"({rel_lower_pct:.1f}%) < threshold {threshold_ratio * 100:.1f}%"
                )
            else:
                reason = f"lower 95% CI {rel_lower_pct:.1f}% (mean {rel_pct:.1f}%) < threshold {threshold_ratio * 100:.1f}%"

        frozen_details[name] = {
            "param": name,
            "mu_star": float(m),
            "sigma": float(s),
            "mu_star_conf": float(conf),
            "mu_star_lower": float(m_lower),
            "rel_influence": float((m / max_mu) if max_mu > 0 else 0.0),
            "rel_influence_lower": float((m_lower / max_mu) if max_mu > 0 else 0.0),
            "fixed_default_index": int(def_idx),
            "fixed_default_repr": _format_param_value(machine, name, def_idx),
            "reason": reason,
        }

    result = MorrisScreeningResult(
        active_params=active_params,
        ignored_params=ignored_params,
        mu_star=mu_star,
        sigma=sigma,
        mu_star_conf=mu_star_conf,
        mu_star_lower=mu_star_lower,
        dummy_stats=dummy_stats,
        sample_points=sample_points_list,
        outputs=y_vals,
        threshold_ratio=threshold_ratio,
        frozen_details=frozen_details,
        output_dicts=output_dicts,
    )

    print(f"\n{result.summary()}\n")
    return result


def add_morris_runs_to_campaign(
    campaign: uq.campaign.Campaign,
    screening_result: MorrisScreeningResult,
    morris_dir: Path | str | None = None,
) -> None:
    """
    Saves the results of the initial Morris screening runs and adds them to the
    EasyVVUQ campaign database using campaign.add_external_runs.
    """
    if not screening_result.sample_points:
        return

    if morris_dir is not None:
        runs_dir = Path(morris_dir)
    elif hasattr(campaign, "root_path"):
        runs_dir = Path(campaign.root_path) / "morris_runs" # type: ignore
    elif hasattr(campaign, "campaign_dir"):
        runs_dir = Path(campaign.campaign_dir).parent / "morris_runs"
    else:
        runs_dir = Path(campaign.work_dir).parent / "morris_runs"

    create_dir(runs_dir)

    input_files: list[str] = []
    output_files: list[str] = []
    sample_points = screening_result.sample_points
    output_dicts = screening_result.output_dicts

    last_out_data: dict[str, float] = {}
    for i, point in enumerate(sample_points):
        run_folder = runs_dir / f"run_{i + 1}"
        create_dir(run_folder)

        input_file = run_folder / "input.json"
        output_file = run_folder / "output.json"

        input_file.write_text(json.dumps(point))

        if i < len(output_dicts) and output_dicts[i]:
            out_data = _normalize_output(output_dicts[i], QOI)
        elif i < len(screening_result.outputs):
            out_data = _normalize_output(screening_result.outputs[i], QOI)
        else:
            out_data = _normalize_output(0.0, QOI)

        output_file.write_text(json.dumps(out_data))
        last_out_data = out_data

        input_files.append(str(input_file))
        output_files.append(str(output_file))

    param_names = list(sample_points[0].keys())
    out_cols = list(last_out_data.keys())

    campaign.add_external_runs(
        input_files=input_files,
        output_files=output_files,
        input_decoder=uq.decoders.JSONDecoder(target_filename="input.json", output_columns=param_names),
        output_decoder=uq.decoders.JSONDecoder(target_filename="output.json", output_columns=out_cols),
        validate_params=True,
        run_prefix="morris_run",
    )
    print(f"[Morris Screening] Added {len(input_files)} screening runs to campaign database.")

