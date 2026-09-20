import unittest
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np

from src.util.multi_run import RunData


class TestSobolDimensionMapping(unittest.TestCase):
    """
    Unit tests verifying Sobol index dimension mapping in multi_run.py:148-161,
    specifically focusing on the off-by-one behavior when 1-based indexing
    for parameter permutations is encountered versus 0-based indexing.
    """

    def test_off_by_one_sobol_index_dimension_mapping(self):
        """
        Verifies the Off-by-One Sobol Index Dimension Mapping in multi_run.py:148-161:
        If parameter permutations use 1-based indexing (e.g., (1,) denotes parameter 0,
        and (2,) denotes parameter 1, as handled in energyuq.py:293):
          In multi_run.py:154:
            dim = perm[0]
            param_name = self.input_params[dim] if dim < len(self.input_params) else f"param_{dim}"
        dim = perm[0] without subtracting 1 offsets every parameter name by one,
        leaving the first parameter missing, mapping (1,) to the second parameter,
        and mapping (2,) to an out-of-bounds fallback 'param_2'.
        """
        input_params = ["N_THREADS", "CLK"]
        first_order_indices_1based = {
            (1,): np.array([0.75]),  # Intended for parameter 0 (N_THREADS) in 1-based indexing
            (2,): np.array([0.25]),  # Intended for parameter 1 (CLK) in 1-based indexing
        }

        mock_analysis = MagicMock()
        mock_analysis.get_sobol_indices.return_value = first_order_indices_1based

        run_data = RunData(
            path=Path("/tmp/test_run"),
            benchmark_name="bench",
            machine_name="mach",
            input_params=input_params,
            analysis=mock_analysis,
            results=None,
        )

        sobols = run_data.get_sobols("energy_uj")

        # 1. Parameter 0 ("N_THREADS") is NOT present in the returned dictionary
        self.assertNotIn("N_THREADS", sobols)

        # 2. Parameter 1 ("CLK") receives the Sobol value for permutation (1,)
        #    which attributes parameter 0's variance (0.75) to parameter 1 ("CLK")
        self.assertIn("CLK", sobols)
        self.assertAlmostEqual(sobols["CLK"], 0.75)

        # 3. Permutation (2,) has dim = 2 >= len(input_params), so it falls back to "param_2"
        #    instead of mapping to parameter 1 ("CLK")
        self.assertIn("param_2", sobols)
        self.assertAlmostEqual(sobols["param_2"], 0.25)

    def test_sobol_index_mapping_with_0_based_indexing(self):
        """
        Verifies that multi_run.py:148-161 correctly maps parameter names when
        permutations are 0-based (e.g., (0,) -> input_params[0], (1,) -> input_params[1]).
        """
        input_params = ["N_THREADS", "CLK"]
        first_order_indices_0based = {
            (0,): np.array([0.75]),
            (1,): np.array([0.25]),
        }

        mock_analysis = MagicMock()
        mock_analysis.get_sobol_indices.return_value = first_order_indices_0based

        run_data = RunData(
            path=Path("/tmp/test_run"),
            benchmark_name="bench",
            machine_name="mach",
            input_params=input_params,
            analysis=mock_analysis,
            results=None,
        )

        sobols = run_data.get_sobols("energy_uj")

        self.assertIn("N_THREADS", sobols)
        self.assertAlmostEqual(sobols["N_THREADS"], 0.75)
        self.assertIn("CLK", sobols)
        self.assertAlmostEqual(sobols["CLK"], 0.25)
        self.assertNotIn("param_2", sobols)

    def test_easyvvuq_sc_analysis_returns_0_based_permutations(self):
        """
        Verifies that EasyVVUQ's SCAnalysis.get_sobol_indices() natively returns
        0-based permutation tuples (0,) and (1,) from powerset(np.arange(self.N)).
        """
        from easyvvuq.analysis.sc_analysis import powerset

        # SCAnalysis.get_sobol_indices uses U = np.arange(self.N)
        n_params = 2
        U = np.arange(n_params)
        P = list(powerset(U))[0 : n_params + 1]

        # P[0] is empty tuple (), first-order indices are P[1:]
        first_order_perms = P[1:]
        self.assertEqual(first_order_perms, [(0,), (1,)])

    def test_energyuq_explored_enough_1_based_indexing_discrepancy(self):
        """
        Verifies the indexing mismatch in energyuq.py:293:
          any(sobol > thresh for perm, sobol in sobols.items() if (dim + 1) in perm)
        Demonstrates that checking (dim + 1) in perm treats (1,) as dim 0 and (2,) as dim 1,
        which conflicts with EasyVVUQ's native 0-based permutation keys (0,) and (1,).
        """
        thresh = 0.05
        # Native EasyVVUQ 0-based output: dim 0 has high Sobol, dim 1 has low Sobol
        native_sobols = {(0,): 0.8, (1,): 0.01}

        # For dim = 0: (dim + 1) is 1. Checking '1 in perm' matches (1,) instead of (0,)
        matches_dim0 = [sobol for perm, sobol in native_sobols.items() if (0 + 1) in perm]
        self.assertEqual(matches_dim0, [0.01])  # Checked dim 1's Sobol index!

        # For dim = 1: (dim + 1) is 2. Checking '2 in perm' matches nothing
        matches_dim1 = [sobol for perm, sobol in native_sobols.items() if (1 + 1) in perm]
        self.assertEqual(matches_dim1, [])  # Never finds dim 1!

    def test_results_sobols_first_precedence_over_analysis(self):
        """
        Verifies that when self.results has sobols_first(), it takes precedence
        over self.analysis.get_sobol_indices() in multi_run.py:139-146.
        """
        mock_results = MagicMock()
        mock_results.sobols_first.return_value = {"N_THREADS": [0.6], "CLK": [0.4]}

        mock_analysis = MagicMock()
        mock_analysis.get_sobol_indices.return_value = {(1,): [0.99], (2,): [0.01]}

        run_data = RunData(
            path=Path("/tmp/test_run"),
            benchmark_name="bench",
            machine_name="mach",
            input_params=["N_THREADS", "CLK"],
            analysis=mock_analysis,
            results=mock_results,
        )

        sobols = run_data.get_sobols("energy_uj")
        self.assertEqual(sobols, {"N_THREADS": 0.6, "CLK": 0.4})
        mock_analysis.get_sobol_indices.assert_not_called()


if __name__ == "__main__":
    unittest.main()

