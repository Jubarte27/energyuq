#!/usr/bin/env python3
import os
import csv
from pathlib import Path

def compile_results(base_dir: str = "run_results"):
    base_path = Path(base_dir)
    if not base_path.exists():
        print(f"Base directory {base_path} does not exist.")
        return

    # Scan through energy_### folders
    for energy_dir in base_path.glob("energy_*"):
        if not energy_dir.is_dir():
            continue

        campaign_dir = energy_dir / "campaign"
        if not campaign_dir.is_dir():
            continue

        # Scan through campaign/energy####### folders
        for camp_sub in campaign_dir.glob("energy*"):
            if not camp_sub.is_dir():
                continue

            runs_dir = camp_sub / "runs"
            if not runs_dir.is_dir():
                continue

            compiled_rows = []
            run_id = 1

            # Iterate over run_# directories or similar
            for run_path in sorted(runs_dir.iterdir(), key=lambda p: p.name):
                if not run_path.is_dir():
                    continue

                input_file = run_path / "input.csv"
                output_file = run_path / "output.csv"

                if not input_file.exists() or not output_file.exists():
                    continue

                # Read input data
                try:
                    input_text = input_file.read_text().strip()
                    input_reader = csv.reader(input_text.splitlines())
                    input_headers = next(input_reader, None)
                    input_values = next(input_reader, None)
                    if not input_values and input_headers:
                        # Handle case where input.csv has no header row (e.g. "20,6")
                        input_values = input_headers
                        input_headers = ["THREADS", "CLK_LEVEL"]
                    elif len(input_headers) >= 2 and not input_headers[0].replace('.', '', 1).isdigit():
                        pass
                    else:
                        input_headers = ["THREADS", "CLK_LEVEL"]
                except Exception as e:
                    print(f"Error reading {input_file}: {e}")
                    continue

                # Read output data
                try:
                    output_text = output_file.read_text().strip()
                    output_reader = csv.reader(output_text.splitlines())
                    output_headers = next(output_reader, [])
                    output_values = next(output_reader, [])
                except Exception as e:
                    print(f"Error reading {output_file}: {e}")
                    continue

                # Combine headers and values with fixed input names
                row_data = {"id": run_id}
                if len(input_values) >= 2:
                    row_data["THREADS"] = input_values[0]
                    row_data["CLK_LEVEL"] = input_values[1]
                elif len(input_values) == 1:
                    row_data["THREADS"] = input_values[0]
                    row_data["CLK_LEVEL"] = ""

                for h, v in zip(output_headers, output_values):
                    row_data[h] = v

                compiled_rows.append(row_data)
                run_id += 1

            if compiled_rows:
                compilation_file = energy_dir / "compilation.csv"
                fieldnames = ["id", "THREADS", "CLK_LEVEL", "energy_uj", "energy_scaled", "time"]

                with open(compilation_file, "w", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    for row in compiled_rows:
                        writer.writerow(row)

                print(f"Successfully compiled {len(compiled_rows)} runs into {compilation_file}")

if __name__ == "__main__":
    compile_results()
