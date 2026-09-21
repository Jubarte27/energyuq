import unittest
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.machines.machine import Machine
from src.util.plot import Plotter, pad_to_even_and_split
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


if __name__ == "__main__":
    unittest.main()
