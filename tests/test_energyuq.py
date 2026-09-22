import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import chaospy as cp
import numpy as np

from src import energyuq
from src.machines.machine import Machine
from src.programs import NONE
from src.util.constants import QOI, RESULTS_DIR


class TestExecuteWrapper(unittest.TestCase):
    def test_start_with_none_returns_none(self):
        called = False

        def dummy_fn(params):
            nonlocal called
            called = True

        wrapper = energyuq.ExecuteWrapper(dummy_fn)
        result = wrapper.start(None)
        self.assertIsNone(result)
        self.assertFalse(called)

    def test_start_executes_function_with_params(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            out_file = Path(tmpdir) / "params.json"

            def dummy_fn(params):
                import json
                with open("params.json", "w") as f:
                    json.dump(params, f)

            wrapper = energyuq.ExecuteWrapper(dummy_fn)
            previous = {
                "rundir": tmpdir,
                "run_info": {"params": {"N_THREADS": 4, "CLK": 2}},
            }
            res = wrapper.start(previous)
            self.assertEqual(res, previous)
            self.assertTrue(out_file.exists())
            with open(out_file) as f:
                saved = json.load(f)
            self.assertEqual(saved, {"N_THREADS": 4, "CLK": 2})

    def test_start_changes_and_restores_cwd(self):
        orig_cwd = os.getcwd()

        with tempfile.TemporaryDirectory() as tmpdir:
            def dummy_fn(params):
                with open("ran_in_cwd.txt", "w") as f:
                    f.write(os.getcwd())

            wrapper = energyuq.ExecuteWrapper(dummy_fn)
            previous = {
                "rundir": tmpdir,
                "run_info": {"params": {"N_THREADS": 1}},
            }
            wrapper.start(previous)

            # Inside wrapper it should have run in tmpdir
            marker_file = Path(tmpdir) / "ran_in_cwd.txt"
            self.assertTrue(marker_file.exists())
            self.assertEqual(os.path.realpath(marker_file.read_text()), os.path.realpath(tmpdir))
            # Outside wrapper it should have restored original cwd
            self.assertEqual(os.path.realpath(os.getcwd()), os.path.realpath(orig_cwd))

    def test_start_restores_cwd_on_exception(self):
        orig_cwd = os.getcwd()

        with tempfile.TemporaryDirectory() as tmpdir:
            def failing_fn(params):
                raise RuntimeError("Boom!")

            wrapper = energyuq.ExecuteWrapper(failing_fn)
            previous = {
                "rundir": tmpdir,
                "run_info": {"params": {}},
            }

            with self.assertRaises(RuntimeError):
                wrapper.start(previous)

            # CWD must still be restored even when exception is raised
            self.assertEqual(os.path.realpath(os.getcwd()), os.path.realpath(orig_cwd))

    def test_wrapper_lifecycle_methods(self):
        wrapper = energyuq.ExecuteWrapper(lambda p: None)
        self.assertTrue(wrapper.finished())
        self.assertTrue(wrapper.succeeded())
        self.assertIsNone(wrapper.finalise())


class TestOrdinalHelper(unittest.TestCase):
    def test_standard_suffixes(self):
        self.assertEqual(energyuq._ordinal(1), "1st")
        self.assertEqual(energyuq._ordinal(2), "2nd")
        self.assertEqual(energyuq._ordinal(3), "3rd")
        self.assertEqual(energyuq._ordinal(4), "4th")
        self.assertEqual(energyuq._ordinal(5), "5th")
        self.assertEqual(energyuq._ordinal(10), "10th")
        self.assertEqual(energyuq._ordinal(21), "21st")
        self.assertEqual(energyuq._ordinal(22), "22nd")
        self.assertEqual(energyuq._ordinal(23), "23rd")
        self.assertEqual(energyuq._ordinal(24), "24th")

    def test_teen_numbers_all_end_in_th(self):
        self.assertEqual(energyuq._ordinal(11), "11th")
        self.assertEqual(energyuq._ordinal(12), "12th")
        self.assertEqual(energyuq._ordinal(13), "13th")
        self.assertEqual(energyuq._ordinal(111), "111th")
        self.assertEqual(energyuq._ordinal(112), "112th")
        self.assertEqual(energyuq._ordinal(113), "113th")
        self.assertEqual(energyuq._ordinal(211), "211th")

    def test_larger_numbers(self):
        self.assertEqual(energyuq._ordinal(101), "101st")
        self.assertEqual(energyuq._ordinal(102), "102nd")
        self.assertEqual(energyuq._ordinal(103), "103rd")
        self.assertEqual(energyuq._ordinal(104), "104th")


class TestCampaignPathAndRunDir(unittest.TestCase):
    def test_campaign_path_with_existing_db(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "campaign.db").touch()
            # If campaign.db exists in root, campaign_path returns root
            self.assertEqual(energyuq.campaign_path(root), root)

    def test_campaign_path_default(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            # Default returns root / "campaign"
            self.assertEqual(energyuq.campaign_path(root), root / "campaign")

    def test_run_dir_with_explicit_dir(self):
        path = energyuq.run_dir(dir="/custom/path/to/run")
        self.assertEqual(path, Path("/custom/path/to/run"))

    def test_run_dir_with_campaign_root_path(self):
        mock_campaign = MagicMock()
        mock_campaign.root_path = "/saved/campaign/root"
        path = energyuq.run_dir(campaign=mock_campaign)
        self.assertEqual(path, Path("/saved/campaign/root"))

    def test_run_dir_default_generates_path_in_results_dir(self):
        path = energyuq.run_dir()
        self.assertTrue(path.name.startswith("energy_"))
        self.assertEqual(path.parent.name, RESULTS_DIR)


class TestDefaultParams(unittest.TestCase):
    def setUp(self):
        self.machine = Machine(
            name="TestMachine",
            freq=[1000, 1500, 2000, 2500],
            max_threads=8,
            places=["threads", "cores"],
            proc_bind=["close", "spread"],
            turbo_boost=["0", "1"],
        )

    def test_default_params_structure(self):
        params, vary = energyuq.default_params(self.machine)
        self.assertEqual(set(params.keys()), {"N_THREADS", "CLK", "PLACES", "BINDING", "BOOST"})
        self.assertEqual(params["N_THREADS"]["default"], 8)
        self.assertEqual(params["CLK"]["default"], 3)
        self.assertEqual(params["PLACES"]["default"], 1)
        self.assertEqual(params["BINDING"]["default"], 1)
        self.assertEqual(params["BOOST"]["default"], 1)

        # Check chaospy distributions
        self.assertEqual(set(vary.keys()), {"N_THREADS", "CLK", "PLACES", "BINDING", "BOOST"})
        for dist in vary.values():
            self.assertIsInstance(dist, cp.Distribution)

    def test_default_params_subset(self):
        params, vary = energyuq.default_params(self.machine, active_params=["N_THREADS", "BOOST"])
        # params defines all 5 for the app template
        self.assertEqual(len(params), 5)
        # vary only samples the active ones
        self.assertEqual(set(vary.keys()), {"N_THREADS", "BOOST"})


class TestEnergyActionsAndAnalysis(unittest.TestCase):
    def setUp(self):
        self.machine = Machine(
            name="TestMachine",
            freq=[1000, 2000],
            max_threads=4,
            places=["cores"],
            proc_bind=["close"],
            turbo_boost=["0", "1"],
        )

    def test_energy_wrapper_actions_creation(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            actions = energyuq.energy_wraper_actions(NONE, self.machine, root)
            self.assertIsNotNone(actions)
            self.assertTrue(Path("easy/energy.template").exists())

    def test_prepare_analysis_and_get_sampler(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            campaign = energyuq.create_campaign(NONE, self.machine, root, active_params=["N_THREADS", "CLK"])
            sampler = energyuq.get_sampler(campaign)
            self.assertIsNotNone(sampler)
            self.assertEqual(sampler.N, 2)

            analysis = energyuq.prepare_analysis(campaign)
            self.assertEqual(analysis.sampler, sampler)
            self.assertEqual(analysis.qoi_cols, [QOI])
            self.assertTrue(hasattr(analysis, "l_norm"))


class TestEnergyUQCampaign(unittest.TestCase):
    def setUp(self):
        self.machine = Machine(
            name="TestMachine",
            freq=[1000, 2000],
            max_threads=4,
            places=["cores"],
            proc_bind=["close"],
            turbo_boost=["0", "1"],
        )

    def test_campaign_encapsulation_and_delegation(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            campaign = energyuq.create_campaign(NONE, self.machine, root, active_params=["N_THREADS"])
            self.assertIsInstance(campaign, energyuq.EnergyUQCampaign)
            self.assertEqual(campaign.root_path, root)
            self.assertEqual(campaign.machine, self.machine)
            self.assertEqual(campaign.active_params, ["N_THREADS"])
            self.assertIsNone(campaign.screening_result)
            self.assertIsNone(campaign.morris_screening)

            # Test property setter alias
            dummy_screening = MagicMock()
            campaign.morris_screening = dummy_screening
            self.assertEqual(campaign.screening_result, dummy_screening)
            self.assertEqual(campaign.morris_screening, dummy_screening)
            self.assertEqual(campaign.campaign.morris_screening, dummy_screening)

            # Test delegation of EasyVVUQ methods and attributes
            self.assertIsNotNone(campaign.get_active_app())
            self.assertIsNotNone(campaign.get_active_sampler())
            self.assertEqual(campaign.campaign_name, "energy")
            self.assertTrue(repr(campaign).startswith("EnergyUQCampaign"))

    def test_prepare_campaign_returns_energyuq_campaign(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            with patch("src.wrappers.base_wrapper.prepare_and_execute", return_value={"energy_uj": 100.0, "time": 1.0, "EDP": 100.0}):
                campaign = energyuq.prepare_campaign(NONE, self.machine, root, active_params=["N_THREADS"])
                self.assertIsInstance(campaign, energyuq.EnergyUQCampaign)
                self.assertEqual(campaign.root_path, root)
                self.assertEqual(campaign.machine, self.machine)


class TestSaveAndLoadEdgeCases(unittest.TestCase):
    def setUp(self):
        self.machine = Machine(
            name="TestMachine",
            freq=[1000, 2000],
            max_threads=4,
            places=["cores"],
            proc_bind=["close"],
            turbo_boost=["0", "1"],
        )

    def test_save_without_machine_raises_value_error(self):
        mock_campaign = MagicMock(spec=[])  # no machine attribute
        mock_analysis = MagicMock()
        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaises(ValueError):
                energyuq.save(mock_campaign, mock_analysis, dir=tmpdir)

    def test_load_without_dir_and_no_runs_raises_runtime_error(self):
        with patch("src.energyuq.latest_dir", return_value=None):
            with self.assertRaises(RuntimeError):
                energyuq.load(NONE, self.machine, "energy")

    def test_load_corrupted_machine_raises_runtime_error(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            # Write invalid object to machine.msgpack
            energyuq._pack(12345, root / "machine.msgpack")
            with self.assertRaises(RuntimeError):
                energyuq.load(NONE, self.machine, "energy", dir=root.as_posix())

    def test_load_machine_as_dict(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            campaign = energyuq.create_campaign(NONE, self.machine, root, active_params=["N_THREADS"])
            analysis = energyuq.prepare_analysis(campaign)
            energyuq.save(campaign, analysis, dir=root.as_posix(), machine=self.machine)

            # Overwrite machine.msgpack with dict representation
            machine_dict = {
                "name": "DictMachine",
                "freq": [800, 1600],
                "max_threads": 2,
                "places": ["threads"],
                "proc_bind": ["spread"],
                "turbo_boost": ["0", "1"],
            }
            energyuq._pack(machine_dict, root / "machine.msgpack")

            loaded_camp, loaded_anal, loaded_mach = energyuq.load(
                NONE, self.machine, "energy", dir=root.as_posix()
            )
            self.assertEqual(loaded_mach.name, "DictMachine")
            self.assertEqual(loaded_mach.max_threads, 2)


class TestRefinementOptions(unittest.TestCase):
    def setUp(self):
        self.machine = Machine(
            name="TestMachine",
            freq=[1000, 2000],
            max_threads=4,
            places=["cores"],
            proc_bind=["close"],
            turbo_boost=["0", "1"],
        )

    def mock_execute(self, *args, **kwargs):
        return {"energy_uj": 100.0, "time": 1.0, "EDP": 100.0}

    def test_refinement_ignored_dims(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            with patch("src.wrappers.base_wrapper.prepare_and_execute", side_effect=self.mock_execute):
                campaign = energyuq.create_campaign(NONE, self.machine, root, active_params=["N_THREADS", "CLK"])
                campaign.execute(sequential=True).collate()
                analysis = energyuq.prepare_analysis(campaign)
                campaign.apply_analysis(analysis)

                sampler = energyuq.get_sampler(campaign)
                # CLK is dimension 1; ignore it
                energyuq.refine_sampling_plan(
                    campaign,
                    analysis,
                    ignored_dims=[1],
                    max_number_of_refinements=1,
                    save_dir=root,
                )
                self.assertEqual(sampler.max_level[1], 1)

    def test_refine_and_analyse_applies_and_saves(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            with patch("src.wrappers.base_wrapper.prepare_and_execute", side_effect=self.mock_execute):
                campaign = energyuq.create_campaign(NONE, self.machine, root, active_params=["N_THREADS", "CLK"])
                campaign.execute(sequential=True).collate()
                analysis = energyuq.prepare_analysis(campaign)
                campaign.apply_analysis(analysis)

                energyuq.refine_and_analyse(
                    campaign,
                    analysis,
                    max_number_of_refinements=1,
                    save_dir=root,
                )

                # Verify checkpoint was written with status="completed"
                import json
                with open(root / "checkpoint.json", "r", encoding="utf-8") as f:
                    ckpt = json.load(f)
                self.assertEqual(ckpt["status"], "completed")

    def test_refinement_min_number_of_refinements(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            with patch("src.wrappers.base_wrapper.prepare_and_execute", side_effect=self.mock_execute):
                campaign = energyuq.create_campaign(NONE, self.machine, root, active_params=["N_THREADS", "CLK"])
                campaign.execute(sequential=True).collate()
                analysis = energyuq.prepare_analysis(campaign)
                campaign.apply_analysis(analysis)

                energyuq.refine_sampling_plan(
                    campaign,
                    analysis,
                    min_number_of_refinements=2,
                    max_number_of_refinements=10,
                    save_dir=root,
                )
                self.assertGreaterEqual(len(analysis.adaptation_errors), 2)


class TestCampaignCreationAndResumption(unittest.TestCase):
    def setUp(self):
        self.machine = Machine(
            name="TestMachine",
            freq=[1000, 2000],
            max_threads=4,
            places=["cores"],
            proc_bind=["close"],
            turbo_boost=["0", "1"],
        )

    def mock_execute(self, *args, **kwargs):
        return {"energy_uj": 100.0, "time": 1.0, "EDP": 100.0}

    def test_prepare_campaign_executes_initial_collation(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            with patch("src.wrappers.base_wrapper.prepare_and_execute", side_effect=self.mock_execute):
                campaign = energyuq.prepare_campaign(NONE, self.machine, root, active_params=["N_THREADS"])
                collation = campaign.get_collation_result()
                self.assertIsNotNone(collation)
                self.assertGreater(len(collation), 0)

    def test_create_and_resume_functions(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            with patch("src.wrappers.base_wrapper.prepare_and_execute", side_effect=self.mock_execute):
                # Test create()
                c1, a1 = energyuq.create(
                    NONE, self.machine, dir=root.as_posix(), screen_morris=False, active_params=["N_THREADS"]
                )
                self.assertIsNotNone(c1)
                self.assertIsNotNone(a1)

                # Test resume() function directly
                c2, a2 = energyuq.resume(NONE, self.machine, "energy", dir=root.as_posix())
                self.assertIsNotNone(c2)
                self.assertIsNotNone(a2)
                self.assertEqual(len(c2.get_collation_result()), len(c1.get_collation_result()))

    def test_run_dir_with_campaign_app_fallback(self):
        mock_campaign = MagicMock(spec=["get_active_app"])
        del mock_campaign.root_path  # ensure no root_path
        mock_campaign.get_active_app.return_value = {"name": "custom_app"}
        path = energyuq.run_dir(campaign=mock_campaign)
        self.assertTrue(path.name.startswith("custom_app_"))


class TestPlotNewPoints(unittest.TestCase):
    @patch("matplotlib.pyplot.show")
    def test_plot_new_points(self, mock_show):
        pts = [(1.0, 2.0), (3.0, 4.0)]
        energyuq.plot_new_points(pts)
        mock_show.assert_called_once()


class TestNumaBalancing(unittest.TestCase):
    def setUp(self):
        self.numa_machine = Machine(
            name="NumaMachine",
            freq=[1000, 2000],
            max_threads=4,
            places=["threads", "cores"],
            proc_bind=["close", "spread"],
            turbo_boost=["false", "true"],
            has_numa=True,
            numactl=["false", "true"],
        )
        self.non_numa_machine = Machine(
            name="NonNumaMachine",
            freq=[1000, 2000],
            max_threads=4,
            places=["threads", "cores"],
            proc_bind=["close", "spread"],
            turbo_boost=["false", "true"],
            has_numa=False,
            numactl=["false", "true"],
        )

    def test_default_params_numa_enabled(self):
        params, vary = energyuq.default_params(self.numa_machine, numa=True)
        self.assertIn("NUMA", params)
        self.assertIn("NUMA", vary)
        self.assertEqual(params["NUMA"]["default"], 1)
        self.assertEqual(set(params.keys()), {"N_THREADS", "CLK", "PLACES", "BINDING", "BOOST", "NUMA"})

    def test_default_params_numa_not_added_when_flag_false(self):
        params, vary = energyuq.default_params(self.numa_machine, numa=False)
        self.assertNotIn("NUMA", params)
        self.assertNotIn("NUMA", vary)

    def test_default_params_numa_not_added_on_non_numa_machine(self):
        params, vary = energyuq.default_params(self.non_numa_machine, numa=True)
        self.assertNotIn("NUMA", params)
        self.assertNotIn("NUMA", vary)

    def test_default_params_numa_via_active_params(self):
        params, vary = energyuq.default_params(self.numa_machine, active_params=["N_THREADS", "NUMA"])
        self.assertIn("NUMA", params)
        self.assertEqual(set(vary.keys()), {"N_THREADS", "NUMA"})

    @patch("src.wrappers.base_wrapper.try_exec", return_value=True)
    def test_set_numa_executes_sysctl(self, mock_try_exec):
        from src.wrappers import base_wrapper
        base_wrapper.set_numa(self.numa_machine, 0)
        mock_try_exec.assert_called_with([["sudo", "/sbin/sysctl", "kernel.numa_balancing=0"]])

        base_wrapper.set_numa(self.numa_machine, 1)
        mock_try_exec.assert_called_with([["sudo", "/sbin/sysctl", "kernel.numa_balancing=1"]])

    @patch("src.wrappers.base_wrapper.set_numa")
    @patch("src.wrappers.base_wrapper.set_boost")
    @patch("src.wrappers.base_wrapper.cpu_set")
    @patch("src.wrappers.base_wrapper.run", return_value=(100, 1.0))
    def test_prepare_and_execute_calls_set_numa_only_when_appropriate(
        self, mock_run, mock_cpu_set, mock_set_boost, mock_set_numa
    ):
        from src.wrappers import base_wrapper
        from src.util.data import ExecutionParams

        # When machine has NUMA and params.numa is an int
        params = ExecutionParams(
            machine=self.numa_machine,
            n_threads=2,
            freq_level=0,
            boost=1,
            place_wideness=0,
            binding=0,
            numa=0,
        )
        base_wrapper.prepare_and_execute(self.numa_machine, NONE, params, [])
        mock_set_numa.assert_called_once_with(self.numa_machine, 0)
        mock_set_numa.reset_mock()

        # When machine has NUMA but params.numa is None
        params_none = ExecutionParams(
            machine=self.numa_machine,
            n_threads=2,
            freq_level=0,
            boost=1,
            place_wideness=0,
            binding=0,
            numa=None,
        )
        base_wrapper.prepare_and_execute(self.numa_machine, NONE, params_none, [])
        mock_set_numa.assert_not_called()

        # When machine does NOT have NUMA even if params.numa is set
        params_non_numa = ExecutionParams(
            machine=self.non_numa_machine,
            n_threads=2,
            freq_level=0,
            boost=1,
            place_wideness=0,
            binding=0,
            numa=0,
        )
        base_wrapper.prepare_and_execute(self.non_numa_machine, NONE, params_non_numa, [])
        mock_set_numa.assert_not_called()

    def test_easy_wrapper_parses_numa_when_present(self):
        from src.wrappers import easy_wrapper
        with tempfile.TemporaryDirectory() as tmpdir:
            in_file = Path(tmpdir) / "input.csv"
            out_file = Path(tmpdir) / "output.csv"
            # 6 parameters: N_THREADS=2, CLK=1, PLACES=0, BINDING=0, BOOST=1, NUMA=0
            in_file.write_text("2,1,0,0,1,0\n")

            with patch("src.wrappers.base_wrapper.prepare_and_execute", return_value={"energy_uj": 100, "time": 1.0}) as mock_pe:
                easy_wrapper.main(NONE, self.numa_machine, input_file=in_file.as_posix(), output_file=out_file.as_posix())
                self.assertTrue(mock_pe.called)
                passed_params = mock_pe.call_args[0][2]
                self.assertEqual(passed_params.numa, 0)

            # 5 parameters: NUMA omitted
            in_file.write_text("2,1,0,0,1\n")
            with patch("src.wrappers.base_wrapper.prepare_and_execute", return_value={"energy_uj": 100, "time": 1.0}) as mock_pe:
                easy_wrapper.main(NONE, self.numa_machine, input_file=in_file.as_posix(), output_file=out_file.as_posix())
                self.assertTrue(mock_pe.called)
                passed_params = mock_pe.call_args[0][2]
                self.assertIsNone(passed_params.numa)

    def test_morris_screening_with_numa(self):
        def eval_fn(pt):
            # NUMA has an effect
            return {"energy_uj": float(pt["N_THREADS"] * 100.0 + pt.get("NUMA", 0) * 50.0)}

        result = energyuq.morris_screen(
            NONE,
            self.numa_machine,
            r=2,
            num_levels=4,
            evaluate_fn=eval_fn,
            default_params_fn=lambda m: energyuq.default_params(m, numa=True),
        )
        self.assertIn("NUMA", result.mu_star)

    def test_create_campaign_numa_option(self):
        import json
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            campaign = energyuq.create_campaign(NONE, self.numa_machine, root, numa=True)
            app = campaign.get_active_app()
            params_dict = json.loads(app["params"].serialize())
            self.assertIn("NUMA", params_dict)
            self.assertIn("NUMA", campaign.active_params)

            campaign_no_numa = energyuq.create_campaign(NONE, self.numa_machine, root / "sub", numa=False)
            app_no_numa = campaign_no_numa.get_active_app()
            params_no_numa_dict = json.loads(app_no_numa["params"].serialize())
            self.assertNotIn("NUMA", params_no_numa_dict)
            self.assertNotIn("NUMA", campaign_no_numa.active_params)

    def test_create_with_numa_flag(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            with patch("src.wrappers.base_wrapper.prepare_and_execute", return_value={"energy_uj": 100.0, "time": 1.0, "EDP": 100.0}):
                c, a = energyuq.create(
                    NONE,
                    self.numa_machine,
                    dir=root.as_posix(),
                    screen_morris=False,
                    active_params=["N_THREADS", "NUMA"],
                    numa=True,
                )
                self.assertIn("NUMA", c.active_params)


class TestSystemAndExecutionUnification(unittest.TestCase):
    def setUp(self):
        self.machine = Machine(
            name="TestMachine",
            freq=[1000, 2000, 3000],
            max_threads=4,
            places=["threads", "cores"],
            proc_bind=["close", "spread"],
            turbo_boost=["0", "1"],
            has_numa=True,
            numactl=["0", "1"],
        )

    def test_try_exec_success(self):
        from src.util.system import try_exec
        self.assertTrue(try_exec([["true"], ["echo", "hello"]]))

    def test_try_exec_failure(self):
        from src.util.system import try_exec
        self.assertFalse(try_exec([["false"]]))
        self.assertFalse(try_exec([["true"], ["false"]]))

    def test_try_exec_with_input(self):
        from src.util.system import try_exec
        self.assertTrue(try_exec([["cat"]], input="data_line"))

    def test_execution_params_from_dict(self):
        from src.util.data import ExecutionParams
        d = {"N_THREADS": 3, "CLK": 1, "PLACES": 0, "BINDING": 1, "BOOST": 0, "NUMA": 1}
        params = ExecutionParams.from_dict(self.machine, d)
        self.assertEqual(params.n_threads, 3)
        self.assertEqual(params.freq_level, 1)
        self.assertEqual(params.place_wideness, 0)
        self.assertEqual(params.binding, 1)
        self.assertEqual(params.boost, 0)
        self.assertEqual(params.numa, 1)

        # Fallback keys (THREADS, CLK_LEVEL)
        d_alt = {"THREADS": 2, "CLK_LEVEL": 0}
        params_alt = ExecutionParams.from_dict(self.machine, d_alt)
        self.assertEqual(params_alt.n_threads, 2)
        self.assertEqual(params_alt.freq_level, 0)

    def test_execution_params_from_args(self):
        from src.util.data import ExecutionParams
        args = ["2", "1", "0", "1", "0", "1", "extra_arg"]
        params = ExecutionParams.from_args(self.machine, args)
        self.assertEqual(params.n_threads, 2)
        self.assertEqual(params.freq_level, 1)
        self.assertEqual(params.place_wideness, 0)
        self.assertEqual(params.binding, 1)
        self.assertEqual(params.boost, 0)
        self.assertEqual(params.numa, 1)

    def test_compute_edp(self):
        from src.util.data import compute_edp
        edp = compute_edp(2_000_000.0, 1.5)
        self.assertAlmostEqual(edp, 3.0)

    def test_to_serializable_primitive(self):
        from src.util.data import to_serializable_primitive
        import numpy as np
        data = {
            "int": np.int64(42),
            "float": np.float64(3.14),
            "arr": np.array([1, 2, 3]),
            "path": Path("/test/path"),
            "tuple": (1, 2),
        }
        res = to_serializable_primitive(data)
        self.assertEqual(res["int"], 42)
        self.assertIsInstance(res["int"], int)
        self.assertEqual(res["float"], 3.14)
        self.assertIsInstance(res["float"], float)
        self.assertEqual(res["arr"], [1, 2, 3])
        self.assertEqual(res["path"], "/test/path")
        self.assertEqual(res["tuple"], [1, 2])

    def test_save_and_load_machine(self):
        from src.machines.machine import save_machine, load_machine
        with tempfile.TemporaryDirectory() as tmpdir:
            p = Path(tmpdir)
            save_machine(self.machine, p)
            self.assertTrue((p / "machine.msgpack").exists())

            loaded = load_machine(p)
            self.assertIsNotNone(loaded)
            self.assertEqual(loaded.name, self.machine.name)
            self.assertEqual(loaded.freq, self.machine.freq)
            self.assertEqual(loaded.max_threads, self.machine.max_threads)

    def test_load_machine_pkl_fallback(self):
        import pickle
        from src.machines.machine import load_machine
        with tempfile.TemporaryDirectory() as tmpdir:
            p = Path(tmpdir)
            with open(p / "machine.pkl", "wb") as f:
                pickle.dump(self.machine, f)

            loaded = load_machine(p)
            self.assertIsNotNone(loaded)
            self.assertEqual(loaded.name, self.machine.name)

    def test_multi_run_derived_qois_edp_only(self):
        import pandas as pd
        from src.util.data import _compute_derived_qois
        df = pd.DataFrame({
            "energy_uj": [1_000_000.0, 2_000_000.0],
            "time": [2.0, 3.0],
        })
        computed = _compute_derived_qois(df)
        self.assertIn("EDP", computed.columns)
        self.assertNotIn("edp_j_s", computed.columns)
        self.assertAlmostEqual(computed["EDP"].iloc[0], 2.0)
        self.assertAlmostEqual(computed["EDP"].iloc[1], 6.0)


if __name__ == "__main__":
    unittest.main()


