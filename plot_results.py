#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def generate_plots(compilation_path: str = "run_results/energy_0/compilation.csv", output_dir: str = "run_results/energy_0/plots"):
    path = Path(compilation_path)
    if not path.exists():
        print(f"Compilation file {path} not found.")
        return

    df = pd.read_csv(path)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    metrics = ["energy_uj", "energy_scaled", "time"]
    independent_vars = ["THREADS", "CLK_LEVEL"]

    for x_var in independent_vars:
        for metric in metrics:
            if x_var not in df.columns or metric not in df.columns:
                continue

            plt.figure(figsize=(8, 6))
            
            # Group by x_var and compute mean/std or scatter all points
            grouped = df.groupby(x_var)[metric].mean().reset_index()
            
            plt.scatter(df[x_var], df[metric], alpha=0.5, label="Raw data", color="tab:blue")
            plt.plot(grouped[x_var], grouped[metric], marker="o", color="tab:red", linewidth=2, label="Mean trend")

            plt.xlabel(x_var, fontsize=12)
            plt.ylabel(metric, fontsize=12)
            plt.title(f"{metric} vs {x_var}", fontsize=14)
            plt.grid(True, linestyle="--", alpha=0.6)
            plt.legend()
            
            plot_file = out_path / f"{metric}_vs_{x_var}.png"
            plt.savefig(plot_file, dpi=300, bbox_inches="tight")
            plt.close()
            print(f"Saved plot: {plot_file}")

if __name__ == "__main__":
    generate_plots()
