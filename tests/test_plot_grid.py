import unittest
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.machines.machine import Machine
from src.plotting.plot import Plotter, pad_to_even_and_split
from src.util.data import Result, EasyResult


class MockAnalysisResults:
    def __init__(self, sobol_dict):
        self._sobol_dict = sobol_dict

    def sobols_first(self, qoi):
        return self._sobol_dict


class TestPlotGrid(unittest.TestCase):
    def setUp(self):
        self.mach = Machine(
            name="test_mach",
            freq=[800000, 1000000, 2000000, 3600000],
            max_threads=16,
            places=["threads"],
            proc_bind=["true"],
        )
        self.plotter = Plotter(self.mach)

    def test_pad_to_even_and_split(self):
        # 1D array with even elements
        a1 = np.array([1, 2, 3, 4])
        r1 = pad_to_even_and_split(a1, 0)
        self.assertEqual(r1.shape, (2, 2))
        np.testing.assert_array_equal(r1[0], [1, 3])
        np.testing.assert_array_equal(r1[1], [2, 4])

        # 1D array with odd elements
        a2 = np.array([1, 2, 3])
        r2 = pad_to_even_and_split(a2, 0)
        self.assertEqual(r2.shape, (2, 2))
        np.testing.assert_array_equal(r2[0], [1, 3])
        np.testing.assert_array_equal(r2[1], [2, 0])

        # 2D array
        a3 = np.array([[10, 20, 30], [11, 21, 31]])
        r3 = pad_to_even_and_split(a3, 0)
        self.assertEqual(r3.shape, (2, 2, 2))
        np.testing.assert_array_equal(r3[:, 0, 0], [10, 11])
        np.testing.assert_array_equal(r3[:, 1, 0], [20, 21])
        np.testing.assert_array_equal(r3[:, 0, 1], [30, 31])
        np.testing.assert_array_equal(r3[:, 1, 1], [0, 0])

    def test_plot_grid_2D_best_various_param_counts(self):
        for n_params in [1, 2, 3, 4, 5]:
            cols = ["N_THREADS", "CLK", "PLACES", "BINDING", "BOOST"][:n_params]
            data = {c: np.random.randint(1, 10, 10) for c in cols}
            data["energy_uj"] = np.random.rand(10) * 100
            data["time"] = np.random.rand(10) * 10
            df = pd.DataFrame(data)
            res = Result(df=df, qois=["energy_uj", "time"])

            # Test standalone figure
            fig = self.plotter.plot_grid_2D_best(res, "energy_uj")
            self.assertIsNotNone(fig)
            for ax in fig.axes:
                self.assertGreater(len(ax.collections), 0)
                offsets = ax.collections[0].get_offsets()
                xlim = ax.get_xlim()
                ylim = ax.get_ylim()
                self.assertTrue(np.all(offsets[:, 0] >= xlim[0]))
                self.assertTrue(np.all(offsets[:, 0] <= xlim[1]))
                self.assertTrue(np.all(offsets[:, 1] >= ylim[0]))
                self.assertTrue(np.all(offsets[:, 1] <= ylim[1]))
            plt.close(fig)

            # Test subfigure integration (like notebook)
            fig_sub = plt.figure(figsize=(8, 8), layout="constrained")
            fs = fig_sub.subfigures(1, 2).flatten()
            f0 = self.plotter.plot_grid_2D_best(res, "energy_uj", subfig=fs[0])
            self.assertEqual(f0, fs[0])
            f1 = self.plotter.plot_grid_2D_best(res, "time", subfig=fs[1])
            self.assertEqual(f1, fs[1])
            plt.close(fig_sub)

    def test_empty_plot_prevention_with_default_machine(self):
        dummy_mach = Machine(name="None", freq=[], max_threads=1)
        plotter = Plotter(dummy_mach)

        df = pd.DataFrame({
            "N_THREADS": [4, 12, 24, 36, 48],
            "CLK": [2, 4, 8, 11, 15],
            "energy_uj": [1e9, 2e9, 1.5e9, 1.8e9, 2.5e9],
            "time": [10, 20, 15, 18, 25],
        })
        res = Result(df=df, qois=["energy_uj", "time"])

        fig = plotter.plot_grid_2D_best(res, "energy_uj")
        self.assertIsNotNone(fig)
        ax = fig.axes[0]
        offsets = ax.collections[0].get_offsets()
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()

        self.assertTrue(np.all(offsets[:, 0] >= xlim[0]))
        self.assertTrue(np.all(offsets[:, 0] <= xlim[1]))
        self.assertTrue(np.all(offsets[:, 1] >= ylim[0]))
        self.assertTrue(np.all(offsets[:, 1] <= ylim[1]))
        self.assertEqual(len(offsets), 5)
        plt.close(fig)

    def test_plot_2D_single_dimension(self):
        df = pd.DataFrame({
            "N_THREADS": [4, 12, 24, 36, 48],
            "CLK": [800000, 1000000, 2000000, 3600000, 3600000],
            "energy_uj": [1e9, 2e9, 1.5e9, 1.8e9, 2.5e9],
        })
        res = Result(df=df, qois=["energy_uj"])
        fig = self.plotter.plot_2D_single_dimension(res, "energy_uj")
        self.assertIsNotNone(fig)
        self.assertEqual(len(fig.axes), 2)
        plt.close(fig)

    def test_plot_sorted(self):
        df = pd.DataFrame({
            "N_THREADS": [4, 12, 24],
            "CLK": [800000, 1000000, 2000000],
            "energy_uj": [1e9, 2e9, 1.5e9],
        })
        res = Result(df=df, qois=["energy_uj"])
        fig = self.plotter.plot_sorted(res, "energy_uj")
        self.assertIsNotNone(fig)
        self.assertEqual(len(fig.axes), 1)
        plt.close(fig)

    def test_plot_boxplot(self):
        df = pd.DataFrame({
            "N_THREADS": [4, 4, 12, 12, 24, 24],
            "CLK": [800000, 800000, 1000000, 1000000, 2000000, 2000000],
            "energy_uj": [1e9, 1.1e9, 2e9, 1.9e9, 1.5e9, 1.4e9],
        })
        res = Result(df=df, qois=["energy_uj"])
        fig = self.plotter.plot_boxplot(res, "energy_uj")
        self.assertIsNotNone(fig)
        self.assertEqual(len(fig.axes), 2)
        plt.close(fig)

    def test_plot_sobols1(self):
        df = pd.DataFrame({
            "N_THREADS": [4, 12, 24],
            "CLK": [800000, 1000000, 2000000],
            "energy_uj": [1e9, 2e9, 1.5e9],
        })
        mock_results = MockAnalysisResults({"N_THREADS": 0.45, "CLK": 0.55})
        res = EasyResult(
            df=df,
            qois=["energy_uj"],
            analysis=None,
            campaign=None,
            sampler=None,
            results=mock_results,
        )
        fig = self.plotter.plot_sobols1(res, "energy_uj")
        self.assertIsNotNone(fig)
        ax = fig.axes[0]
        self.assertEqual(len(ax.get_xticks()), 3)
        plt.close(fig)

    def test_plot_sobols1_array_values(self):
        """Test plot_sobols1 when sobols_first returns 1D numpy arrays (standard EasyVVUQ behavior)."""
        df = pd.DataFrame({
            "N_THREADS": [4, 12, 24],
            "CLK": [800000, 1000000, 2000000],
            "energy_uj": [1e9, 2e9, 1.5e9],
        })
        mock_results = MockAnalysisResults({"N_THREADS": np.array([0.45]), "CLK": np.array([0.55])})
        res = EasyResult(
            df=df,
            qois=["energy_uj"],
            analysis=None,
            campaign=None,
            sampler=None,
            results=mock_results,
        )
        fig = self.plotter.plot_sobols1(res, "energy_uj")
        self.assertIsNotNone(fig)
        ax = fig.axes[0]
        self.assertEqual(len(ax.get_xticks()), 3)
        plt.close(fig)

    def test_get_sobols_up_to_order_and_plot_sobols(self):
        from src.plotting.plot import get_sobols_up_to_order, sobols_up_to_order, plot_sobols, SobolOrderResult

        # Mock sampler and analysis
        class MockSamplerVary:
            def __init__(self, keys):
                self._keys = keys
            def get_keys(self):
                return self._keys

        class MockSampler:
            def __init__(self, keys):
                self.vary = MockSamplerVary(keys)
                self.N = len(keys)

        class MockSCAnalysis:
            def __init__(self, sobol_indices_dict, param_names):
                self.sampler = MockSampler(param_names)
                self.N = len(param_names)
                self.qoi_cols = ["energy_uj"]
                self._sobol_indices = sobol_indices_dict

            def get_pce_sobol_indices(self, qoi, typ="first_order", **kwargs):
                if typ == "first_order":
                    s_u = {u: v for u, v in self._sobol_indices.items() if len(u) == 1}
                else:
                    s_u = self._sobol_indices
                d_u = {u: np.array([float(np.asarray(v).ravel()[0]) * 100.0]) for u, v in s_u.items()}
                return np.array([50.0]), np.array([100.0]), d_u, s_u

        # Order 1: 0.70 + 0.25 + 0.02 = 0.97 (97%)
        # Order 2: (0, 1) = 0.025, (0, 2) = 0.003, (1, 2) = 0.001 -> 0.029 (2.9%)
        # Order 3: (0, 1, 2) = 0.001 -> 0.001 (0.1%)
        sobol_dict = {
            (0,): np.array([0.70]),
            (1,): np.array([0.25]),
            (2,): np.array([0.02]),
            (0, 1): np.array([0.025]),
            (0, 2): np.array([0.003]),
            (1, 2): np.array([0.001]),
            (0, 1, 2): np.array([0.001]),
        }
        param_names = ["CLK", "N_THREADS", "PLACES"]
        mock_analysis = MockSCAnalysis(sobol_dict, param_names)

        # 1. With k = 5.0% -> higher orders (2.9% + 0.1% = 3.0%) < 5.0%, so n = 1
        res_k5 = get_sobols_up_to_order(mock_analysis, "energy_uj", k=5.0)
        self.assertIsInstance(res_k5, SobolOrderResult)
        self.assertEqual(res_k5.n, 1)
        self.assertEqual(res_k5.order, 1)
        self.assertAlmostEqual(res_k5.higher_order_pct, 3.0, places=2)
        self.assertAlmostEqual(res_k5.total_influence, 0.97, places=2)
        # Should only contain 3 terms (first order)
        self.assertEqual(len(res_k5), 3)
        self.assertIn("CLK", res_k5)
        self.assertIn("N_THREADS", res_k5)
        self.assertIn("PLACES", res_k5)

        # 2. With k = 1.0% -> higher orders for n=1 is 3.0% >= 1.0%; for n=2 is 0.1% < 1.0%, so n = 2
        res_k1 = sobols_up_to_order(mock_analysis, "energy_uj", k=1.0)
        self.assertEqual(res_k1.n, 2)
        self.assertAlmostEqual(res_k1.higher_order_pct, 0.1, places=2)
        # Included terms: 3 first-order + 3 second-order = 6 terms
        self.assertEqual(len(res_k1), 6)
        self.assertIn("CLK × N_THREADS", res_k1)

        # 3. With k = 0.05% -> higher orders for n=2 is 0.1% >= 0.05%; for n=3 is 0.0% < 0.05%, so n = 3
        res_k005 = self.plotter.get_sobols_up_to_order(mock_analysis, "energy_uj", k=0.05)
        self.assertEqual(res_k005.n, 3)
        self.assertAlmostEqual(res_k005.higher_order_pct, 0.0, places=2)
        self.assertEqual(len(res_k005), 7)
        self.assertIn("CLK × N_THREADS × PLACES", res_k005)

        # 4. Test plot_sobols with EasyResult container
        easy_res = EasyResult(
            df=pd.DataFrame({"CLK": [1000], "N_THREADS": [4], "PLACES": [0], "energy_uj": [1.0]}),
            qois=["energy_uj"],
            analysis=mock_analysis,
            campaign=None,
            sampler=mock_analysis.sampler,
            results=None,
        )
        fig = plot_sobols(easy_res, "energy_uj", k=1.0)
        self.assertIsNotNone(fig)
        ax = fig.axes[0]
        # Total bar + 6 terms + higher orders bar = 8 bars
        self.assertEqual(len(ax.get_xticks()), 8)
        self.assertTrue(hasattr(fig, "_sobol_result"))
        self.assertEqual(fig._sobol_result.n, 2)
        plt.close(fig)

        # 5. Test plot_sobols with manual order override
        fig_manual = self.plotter.plot_sobols(easy_res, "energy_uj", order=1, include_higher_orders=False)
        self.assertEqual(fig_manual._sobol_result.n, 1)
        ax_manual = fig_manual.axes[0]
        # Total bar + 3 order-1 terms = 4 bars
        self.assertEqual(len(ax_manual.get_xticks()), 4)
        plt.close(fig_manual)

        # 6. Test invalid object without get_pce_sobol_indices
        with self.assertRaises(ValueError):
            get_sobols_up_to_order("not_an_analysis")

        # 7. Test plot_sobols stacked into a single column adding up to 1
        fig_stacked = plot_sobols(easy_res, "energy_uj", k=1.0, stacked=True)
        self.assertIsNotNone(fig_stacked)
        ax_stacked = fig_stacked.axes[0]
        # Single column
        self.assertEqual(len(ax_stacked.get_xticks()), 1)
        self.assertEqual(ax_stacked.get_ylim(), (0.0, 1.0))
        # Total height of all stacked bars adds up to 1.0
        patches = ax_stacked.patches
        total_height = sum(p.get_height() for p in patches)
        self.assertAlmostEqual(total_height, 1.0, places=5)
        # Check visually distinct colors assigned
        facecolors = [p.get_facecolor() for p in patches]
        self.assertEqual(len(facecolors), len(set(facecolors)))
        # Legend present and has corresponding items
        legend = ax_stacked.get_legend()
        self.assertIsNotNone(legend)
        self.assertEqual(len(legend.get_texts()), len(patches))
        plt.close(fig_stacked)

        # 8. Test plotter.plot_sobols with stacked=True and order=1
        fig_stacked_order1 = self.plotter.plot_sobols(easy_res, "energy_uj", order=1, stacked=True)
        ax_s1 = fig_stacked_order1.axes[0]
        self.assertEqual(len(ax_s1.get_xticks()), 1)
        self.assertAlmostEqual(sum(p.get_height() for p in ax_s1.patches), 1.0, places=5)
        plt.close(fig_stacked_order1)

    def test_get_distinct_colors(self):
        from src.plotting.colors import get_distinct_colors, distinct_colors, get_n_distinct_colors

        # Edge cases
        self.assertEqual(get_distinct_colors(0), [])
        self.assertEqual(len(get_distinct_colors(1)), 1)
        self.assertEqual(len(get_distinct_colors(5)), 5)
        self.assertEqual(len(get_distinct_colors(10)), 10)
        self.assertEqual(len(get_distinct_colors(20)), 20)
        self.assertEqual(len(get_distinct_colors(35)), 35)

        # Distinctness: all generated colors must be unique
        for count in [5, 12, 25]:
            cols = get_distinct_colors(count)
            self.assertEqual(len(cols), count)
            self.assertEqual(len(set(cols)), count)

        # RGB tuple mode
        rgb_cols = get_distinct_colors(4, as_hex=False)
        self.assertEqual(len(rgb_cols), 4)
        for c in rgb_cols:
            self.assertIsInstance(c, tuple)
            self.assertEqual(len(c), 3)

        # Colormap sampling
        palette_cols = get_distinct_colors(6, palette="Set1")
        self.assertEqual(len(palette_cols), 6)
        self.assertEqual(len(set(palette_cols)), 6)

        # Aliases
        self.assertEqual(distinct_colors(5), get_distinct_colors(5))
        self.assertEqual(get_n_distinct_colors(5), get_distinct_colors(5))


if __name__ == "__main__":
    unittest.main()

