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


if __name__ == "__main__":
    unittest.main()
