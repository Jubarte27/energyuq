import unittest
import tempfile
import msgpack
from pathlib import Path
import numpy as np

from src.machines.machine import Machine
from src.programs import NONE
from src import energyuq


class TestMorrisScreening(unittest.TestCase):

    def setUp(self):
        self.machine = Machine(
            name="TestMachine",
            freq=[1000, 1500, 2000, 2500],
            max_threads=8,
            places=["threads", "cores", "sockets"],
            proc_bind=["true", "close", "spread", "false"],
            turbo_boost=["false", "true"],
        )

    def test_default_params_active_filtering(self):
        # When active_params is None, all 5 params should be in vary
        params, vary = energyuq.default_params(self.machine)
        self.assertEqual(len(params), 5)
        self.assertEqual(len(vary), 5)

        # When active_params is specified, only those should be in vary
        active = ["N_THREADS", "CLK"]
        params_sub, vary_sub = energyuq.default_params(self.machine, active_params=active)
        self.assertEqual(len(params_sub), 5)  # App still defines all parameters with defaults
        self.assertEqual(list(vary_sub.keys()), active)  # Only active are sampled

    def test_morris_screening_identifies_unimportant_variables(self):
        # Synthetic evaluation: Only N_THREADS and CLK affect the energy output.
        # PLACES, BINDING, and BOOST have ZERO effect.
        def mock_evaluate(point: dict[str, int]) -> dict[str, float]:
            return {
                "energy_uj": float(point["N_THREADS"] * 500.0 + point["CLK"] * 100.0 + 50.0),
                "time": float(point["N_THREADS"] * 2.0),
            }

        result = energyuq.morris_screen(
            program=NONE,
            machine=self.machine,
            r=4,
            num_levels=4,
            threshold_ratio=0.05,
            evaluate_fn=mock_evaluate,
        )

        self.assertIn("N_THREADS", result.active_params)
        self.assertIn("CLK", result.active_params)
        self.assertIn("PLACES", result.ignored_params)
        self.assertIn("BINDING", result.ignored_params)
        self.assertIn("BOOST", result.ignored_params)

        self.assertGreater(result.mu_star["N_THREADS"], 0.0)
        self.assertGreater(result.mu_star["CLK"], 0.0)
        self.assertEqual(result.mu_star["PLACES"], 0.0)
        self.assertEqual(result.mu_star["BINDING"], 0.0)
        self.assertEqual(result.mu_star["BOOST"], 0.0)

        # Test serialization
        with tempfile.TemporaryDirectory() as tmpdir:
            result.save(tmpdir)
            saved_msgpack = Path(tmpdir) / "morris_screening.msgpack"
            self.assertTrue(saved_msgpack.exists())
            with open(saved_msgpack, "rb") as f:
                data = msgpack.unpack(f)
            self.assertEqual(data["active_params"], result.active_params)
            self.assertEqual(data["ignored_params"], result.ignored_params)

            # Test load
            loaded_result = energyuq.MorrisScreeningResult.load(tmpdir)
            self.assertEqual(loaded_result.active_params, result.active_params)
            self.assertEqual(loaded_result.ignored_params, result.ignored_params)
            self.assertEqual(loaded_result.frozen_params, result.frozen_params)
            self.assertIn("PLACES", loaded_result.frozen_details)
            self.assertTrue(len(loaded_result.summary()) > 0)

    def test_morris_screening_with_constant_bounds(self):
        # Machine with fixed turbo boost (length 1)
        fixed_boost_machine = Machine(
            name="FixedBoostMachine",
            freq=[1000, 2000],
            max_threads=4,
            places=["cores"],
            proc_bind=["close"],
            turbo_boost=["false"],
        )

        def mock_evaluate(point: dict[str, int]) -> dict[str, float]:
            return {"energy_uj": float(point["N_THREADS"] * 100.0)}

        result = energyuq.morris_screen(
            program=NONE,
            machine=fixed_boost_machine,
            r=4,
            num_levels=4,
            threshold_ratio=0.05,
            evaluate_fn=mock_evaluate,
        )

        self.assertEqual(result.mu_star["BOOST"], 0.0)
        self.assertEqual(result.mu_star["PLACES"], 0.0)
        self.assertIn("N_THREADS", result.active_params)

    def test_campaign_creation_with_active_params(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            active_params = ["N_THREADS", "CLK"]

            campaign = energyuq.create_campaign(
                program=NONE,
                machine=self.machine,
                root=root,
                active_params=active_params,
            )

            sampler = energyuq.get_sampler(campaign)
            self.assertEqual(sampler.N, 2)
            self.assertEqual(list(sampler.vary.get_keys()), active_params)
            self.assertEqual(getattr(campaign, "active_params"), active_params)

    def test_save_and_load_active_params(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            active_params = ["N_THREADS", "CLK"]

            campaign = energyuq.create_campaign(
                program=NONE,
                machine=self.machine,
                root=root,
                active_params=active_params,
            )
            analysis = energyuq.prepare_analysis(campaign)

            energyuq.save(campaign, analysis, dir=root.as_posix(), machine=self.machine)

            active_msgpack = root / "active_params.msgpack"
            self.assertTrue(active_msgpack.exists())
            with open(active_msgpack, "rb") as f:
                saved = msgpack.unpack(f)
            self.assertEqual(saved, active_params)

            machine_msgpack = root / "machine.msgpack"
            self.assertTrue(machine_msgpack.exists())

            # Test reload
            loaded_campaign, loaded_analysis, loaded_machine = energyuq.load(
                NONE, self.machine, "test_camp", dir=root.as_posix()
            )
            loaded_sampler = energyuq.get_sampler(loaded_campaign)
            self.assertEqual(loaded_sampler.N, 2)
            self.assertEqual(list(loaded_sampler.vary.get_keys()), active_params)
            self.assertEqual(loaded_machine.name, self.machine.name)
            self.assertEqual(loaded_machine.freq, self.machine.freq)

    def test_create_with_screen_morris(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)

            def mock_eval(pt):
                # Only N_THREADS and CLK matter
                return {"energy_uj": float(pt["N_THREADS"] * 100.0 + pt["CLK"] * 20.0)}

            def mock_easy_wrapper(program, machine, input_file="input.csv", output_file="output.csv"):
                with open(input_file, "r") as f:
                    line = f.readline().strip()
                vals = [float(x) for x in line.split(",") if x.strip()]
                n_threads = vals[0] if len(vals) > 0 else 1
                clk = vals[1] if len(vals) > 1 else 0
                energy = float(n_threads * 100.0 + clk * 20.0)
                with open(output_file, "w") as f:
                    f.write("energy_uj,EDP,time\n")
                    f.write(f"{energy},1.0,1.0\n")

            import unittest.mock as mock
            with mock.patch("src.wrappers.easy_wrapper.main", side_effect=mock_easy_wrapper):
                campaign, analysis = energyuq.create(
                    NONE,
                    self.machine,
                    dir=root.as_posix(),
                    screen_morris=True,
                    morris_r=4,
                    evaluate_fn=mock_eval,
                )

                self.assertTrue(root.joinpath("morris_screening.msgpack").exists())
                self.assertTrue(hasattr(campaign, "morris_screening"))
                self.assertEqual(campaign.morris_screening.active_params, ["N_THREADS", "CLK"])

                sampler = energyuq.get_sampler(campaign)
                self.assertEqual(sampler.N, 2)

    def test_refinement_convergence_with_unimportant_dimension(self):
        # Test that when an unimportant dimension is present, refinement converges
        # fast and does NOT hang until max_number_of_refinements
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            # Include an uninfluential parameter in vary
            active_params = ["N_THREADS", "PLACES"]

            def mock_easy_wrapper(program, machine, input_file="input.csv", output_file="output.csv"):
                with open(input_file, "r") as f:
                    line = f.readline().strip()
                vals = [float(x) for x in line.split(",") if x.strip()]
                n_threads = vals[0]
                # PLACES has 0 effect on energy
                energy = float(n_threads * 100.0 + 50.0)
                with open(output_file, "w") as f:
                    f.write("energy_uj,EDP,time\n")
                    f.write(f"{energy},1.0,1.0\n")

            import unittest.mock as mock
            with mock.patch("src.wrappers.easy_wrapper.main", side_effect=mock_easy_wrapper):
                campaign = energyuq.prepare_campaign(NONE, self.machine, root, active_params=active_params)
                analysis = energyuq.prepare_analysis(campaign)
                campaign.apply_analysis(analysis)

                # Refine with max 10
                energyuq.refine_sampling_plan(
                    campaign,
                    analysis,
                    max_number_of_refinements=10,
                    surplus_tol=0.2,
                    mean_tol=0.2,
                    var_tol=0.2,
                    patience=1,
                    sobol_thresh=1e-3,
                )

                # The loop should have converged without exhausting all 10 refinements
                self.assertLess(len(analysis.adaptation_errors), 10)

    def test_add_morris_runs_to_campaign(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)

            def mock_eval(pt):
                return {"energy_uj": float(pt["N_THREADS"] * 100.0), "EDP": 1.0, "time": 1.0}

            result = energyuq.morris_screen(
                NONE,
                self.machine,
                r=2,
                num_levels=4,
                evaluate_fn=mock_eval,
            )

            campaign = energyuq.create_campaign(NONE, self.machine, root, active_params=result.active_params)

            # Before adding, DB should have 0 collated runs
            before_runs = list(campaign.campaign_db.runs(status=energyuq.uq.constants.Status.COLLATED))
            self.assertEqual(len(before_runs), 0)

            # Add Morris runs
            energyuq.add_morris_runs_to_campaign(campaign, result)

            after_runs = list(campaign.campaign_db.runs(status=energyuq.uq.constants.Status.COLLATED))
            self.assertEqual(len(after_runs), len(result.sample_points))
            self.assertTrue(after_runs[0][1]["run_name"].startswith("morris_run_"))

    def test_create_registers_morris_runs_in_database(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)

            def mock_eval(pt):
                return {"energy_uj": float(pt["N_THREADS"] * 100.0 + pt["CLK"] * 20.0), "EDP": 1.0, "time": 1.0}

            def mock_easy_wrapper(program, machine, input_file="input.csv", output_file="output.csv"):
                with open(input_file, "r") as f:
                    vals = [float(x) for x in f.readline().split(",") if x.strip()]
                energy = float(vals[0] * 100.0 + vals[1] * 20.0)
                with open(output_file, "w") as f:
                    f.write(f"energy_uj,EDP,time\n{energy},1.0,1.0\n")

            import unittest.mock as mock
            with mock.patch("src.wrappers.easy_wrapper.main", side_effect=mock_easy_wrapper):
                campaign, analysis = energyuq.create(
                    NONE,
                    self.machine,
                    dir=root.as_posix(),
                    screen_morris=True,
                    morris_r=2,
                    evaluate_fn=mock_eval,
                )

                # Database should contain Morris screening runs
                collated_runs = list(campaign.campaign_db.runs(status=energyuq.uq.constants.Status.COLLATED))
                morris_runs = [r for r in collated_runs if r[1]["run_name"].startswith("morris_run_")]
                self.assertGreater(len(morris_runs), 0)
                self.assertEqual(len(morris_runs), len(campaign.morris_screening.sample_points))

    def test_morris_screening_mean_threshold_rule(self):
        # Machine with 2 levels per parameter
        test_machine = Machine(
            name="BorderlineTestMachine",
            freq=[1000, 2000],
            max_threads=4,
            places=["threads", "cores"],
            proc_bind=["close", "spread"],
            turbo_boost=["false", "true"],
        )

        # N_THREADS is dominant: max_mu ~ 3000
        # CLK has mean effect around ~5.25% (157.5), but with variance / interaction
        # such that its lower 95% CI is ~4.3% (< 5.0% threshold).
        def mock_eval(pt):
            y = pt["N_THREADS"] * 1000.0
            if pt["BOOST"] == 1:
                y += pt["CLK"] * 160.0
            else:
                y += pt["CLK"] * 50.0
            return {"energy_uj": float(y)}

        result = energyuq.morris_screen(
            NONE,
            test_machine,
            r=20,
            num_levels=4,
            threshold_ratio=0.05,
            evaluate_fn=mock_eval,
            seed=2024,
        )

        max_mu = result.mu_star["N_THREADS"]
        clk_mu = result.mu_star["CLK"]
        clk_conf = result.mu_star_conf["CLK"]
        clk_lower = result.mu_star_lower["CLK"]

        # Verify CLK's mean is above 5% threshold
        self.assertGreaterEqual(clk_mu / max_mu, 0.05)
        # Verify confidence interval is non-zero
        self.assertGreater(clk_conf, 0.0)
        # Verify lower CI bound drops below 5% threshold
        self.assertLess(clk_lower / max_mu, 0.05)
        # Under the mean-based screening rule, CLK is active despite lower CI < threshold
        self.assertIn("CLK", result.active_params)
        self.assertNotIn("CLK", result.ignored_params)

        # Unimportant parameters (e.g. PLACES, BINDING) should be frozen
        self.assertIn("PLACES", result.ignored_params)
        places_details = result.frozen_details["PLACES"]
        self.assertIn("mean", places_details["reason"])
        self.assertIn("< threshold", places_details["reason"])

    def test_morris_screening_dummy_variable_system_variation(self):
        # 1. Deterministic system: dummy variable should have mu* = 0, sigma = 0
        def det_eval(pt):
            return {"energy_uj": float(pt["N_THREADS"] * 100.0)}

        det_result = energyuq.morris_screen(
            NONE,
            self.machine,
            r=4,
            num_levels=4,
            evaluate_fn=det_eval,
            include_dummy=True,
        )

        self.assertIn("name", det_result.dummy_stats)
        self.assertEqual(det_result.dummy_stats["name"], "DUMMY")
        self.assertEqual(det_result.dummy_stats["mu_star"], 0.0)
        self.assertEqual(det_result.dummy_stats["sigma"], 0.0)

        # Ensure DUMMY is NOT in active_params or ignored_params (only for humans to see)
        self.assertNotIn("DUMMY", det_result.active_params)
        self.assertNotIn("DUMMY", det_result.ignored_params)
        self.assertIn("SYSTEM NOISE BENCHMARK", det_result.summary())

        # 2. System with stochastic noise / variation: dummy variable captures noise floor
        np.random.seed(42)
        def noisy_eval(pt):
            # Deterministic signal + stochastic system variation
            noise = float(np.random.normal(0.0, 10.0))
            return {"energy_uj": float(pt["N_THREADS"] * 500.0 + noise)}

        noisy_result = energyuq.morris_screen(
            NONE,
            self.machine,
            r=10,
            num_levels=4,
            evaluate_fn=noisy_eval,
            include_dummy=True,
        )

        # The dummy variable should capture the system noise (mu* > 0 or sigma > 0)
        self.assertGreater(noisy_result.dummy_stats["mu_star"], 0.0)
        self.assertGreater(noisy_result.dummy_stats["sigma"], 0.0)
        self.assertNotIn("DUMMY", noisy_result.active_params)
        self.assertNotIn("DUMMY", noisy_result.ignored_params)

        # Test plot generation with dummy variable at the end
        with tempfile.TemporaryDirectory() as tmpdir:
            plot_file = Path(tmpdir) / "test_dummy_plot.png"
            noisy_result.plot(plot_file)
            self.assertTrue(plot_file.exists())

    def test_morris_only_creates_screening_files_and_stops(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)

            def mock_eval(pt):
                return {"energy_uj": float(pt["N_THREADS"] * 100.0 + pt["CLK"] * 20.0), "EDP": 1.0, "time": 1.0}

            c, a = energyuq.create(
                NONE,
                self.machine,
                dir=root.as_posix(),
                morris_only=True,
                morris_r=2,
                evaluate_fn=mock_eval,
            )

            self.assertIsNone(c)
            self.assertIsNone(a)
            self.assertTrue((root / "morris_screening.msgpack").exists())
            self.assertTrue((root / "morris_screening.png").exists())
            self.assertFalse((root / "campaign" / "campaign.db").exists())

    def test_create_from_morris_bypasses_screening_and_registers_runs(self):
        with tempfile.TemporaryDirectory() as tmpdir1, tempfile.TemporaryDirectory() as tmpdir2:
            root1 = Path(tmpdir1)
            root2 = Path(tmpdir2)

            call_count = 0

            def mock_eval(pt):
                nonlocal call_count
                call_count += 1
                return {"energy_uj": float(pt["N_THREADS"] * 100.0 + pt["CLK"] * 20.0), "EDP": 1.0, "time": 1.0}

            # Phase 1: generate morris screening only
            energyuq.create(
                NONE,
                self.machine,
                dir=root1.as_posix(),
                morris_only=True,
                morris_r=2,
                evaluate_fn=mock_eval,
            )
            initial_calls = call_count
            self.assertGreater(initial_calls, 0)

            # Phase 2: run campaign from morris path
            def mock_easy_wrapper(program, machine, input_file="input.csv", output_file="output.csv"):
                with open(input_file, "r") as f:
                    vals = [float(x) for x in f.readline().split(",") if x.strip()]
                energy = float(vals[0] * 100.0 + vals[1] * 20.0)
                with open(output_file, "w") as f:
                    f.write(f"energy_uj,EDP,time\n{energy},1.0,1.0\n")

            import unittest.mock as mock

            with mock.patch("src.wrappers.easy_wrapper.main", side_effect=mock_easy_wrapper):
                campaign, analysis = energyuq.create(
                    NONE,
                    self.machine,
                    dir=root2.as_posix(),
                    from_morris=root1.as_posix(),
                    evaluate_fn=mock_eval,
                )

                # evaluate_fn (morris_screen) should NOT have been called during from_morris
                self.assertEqual(call_count, initial_calls)

                # Active parameters should match the screening result
                self.assertEqual(campaign.active_params, ["N_THREADS", "CLK"])

                # Screening files should be saved in the new campaign dir
                self.assertTrue((root2 / "morris_screening.msgpack").exists())
                self.assertTrue((root2 / "morris_screening.png").exists())

                # Screening runs should be registered as external runs in campaign DB
                collated = list(campaign.campaign_db.runs(status=energyuq.uq.constants.Status.COLLATED))
                morris_runs = [r for r in collated if r[1]["run_name"].startswith("morris_run_")]
                self.assertGreater(len(morris_runs), 0)

    def test_run_py_parse_args(self):
        import run

        with unittest.mock.patch("sys.argv", ["run.py", "--morris-only"]):
            args = run.parse_args()
            self.assertTrue(args.morris_only)
            self.assertIsNone(args.from_morris)

        with unittest.mock.patch("sys.argv", ["run.py", "--from-morris", "path/to/morris"]):
            args = run.parse_args()
            self.assertFalse(args.morris_only)
            self.assertEqual(args.from_morris, "path/to/morris")


if __name__ == "__main__":
    unittest.main()


