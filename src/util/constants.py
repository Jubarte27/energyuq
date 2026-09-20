from typing import Any
import chaospy as cp

# Quantity of Interest constants
QOI: str = "energy_uj"
QOIS: list[str] = ["energy_uj", "EDP", "time"]
RESULTS_DIR: str = "run_results"

# Type aliases
params_type = dict[str, dict[str, Any]]
vary_type = dict[str, cp.Distribution]

