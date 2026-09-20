import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src import energyuq
from src.machines.machine import Machine
from src.programs import NONE
from src.util.morris import MorrisScreeningResult


class TestSaveResume(unittest.TestCase):
    def setUp(self):
        self.machine = Machine(
            name="TestMachine",
            freq=[1000, 1500, 2000, 2500],
            max_threads=4,
            places=["cores", "threads"],
            proc_bind=["close", "spread"],
            turbo_boost=["0", "1"],
        )

    def mock_execute(self, *args, **kwargs):
        return {"energy_uj": 1234.0, "time": 0.5, "EDP": 617.0}

    def test_checkpoint_files_created_on_save(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            with patch("src.wrappers.base_wrapper.prepare_and_execute", side_effect=self.mock_execute):
                campaign, analysis = energyuq.create(
                    NONE,
                    self.machine,
                    dir=root.as_posix(),
                    screen_morris=False,
                    active_params=["N_THREADS", "CLK"],
                )

                # Fresh create automatically saves initial checkpoint
                checkpoint_path = root / "checkpoint.json"
                self.assertTrue(checkpoint_path.exists())
                with open(checkpoint_path, "r", encoding="utf-8") as f:
                    ckpt = json.load(f)
                self.assertEqual(ckpt["iteration"], 0)
                self.assertEqual(ckpt["status"], "in_progress")
                self.assertEqual(ckpt["converged"], False)

                self.assertTrue((root / "sampler").exists())
                self.assertTrue((root / "analysis").exists())
                self.assertTrue((root / "machine.msgpack").exists())
                self.assertTrue((root / "active_params.msgpack").exists())

    def test_periodic_saving_every_2_iterations(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            with patch("src.wrappers.base_wrapper.prepare_and_execute", side_effect=self.mock_execute):
                campaign, analysis = energyuq.create(
                    NONE,
                    self.machine,
                    dir=root.as_posix(),
                    screen_morris=False,
                    active_params=["N_THREADS", "CLK"],
                )

                # Run 2 iterations with save_every=2
                energyuq.refine_sampling_plan(
                    campaign,
                    analysis,
                    max_number_of_refinements=2,
                    save_every=2,
                    save_dir=root,
                )

                self.assertEqual(len(analysis.adaptation_errors), 2)
                with open(root / "checkpoint.json", "r", encoding="utf-8") as f:
                    ckpt = json.load(f)
                self.assertEqual(ckpt["iteration"], 2)
                self.assertEqual(ckpt["status"], "completed")

    def test_seamless_resumption_continues_iterations_and_samples(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            with patch("src.wrappers.base_wrapper.prepare_and_execute", side_effect=self.mock_execute):
                # Phase 1: create and run 2 refinement iterations
                c1, a1 = energyuq.create(
                    NONE,
                    self.machine,
                    dir=root.as_posix(),
                    screen_morris=False,
                    active_params=["N_THREADS", "CLK"],
                )
                energyuq.refine_sampling_plan(
                    c1,
                    a1,
                    max_number_of_refinements=2,
                    save_every=2,
                    save_dir=root,
                )
                collation_len_phase1 = len(c1.get_collation_result())
                self.assertEqual(len(a1.adaptation_errors), 2)

                # Phase 2: Resume campaign from same directory
                c2, a2 = energyuq.create(
                    NONE,
                    self.machine,
                    dir=root.as_posix(),
                    resume=True,
                )
                self.assertEqual(len(a2.adaptation_errors), 2)
                self.assertEqual(len(c2.get_collation_result()), collation_len_phase1)

                # Run 1 more refinement iteration on resumed campaign
                energyuq.refine_sampling_plan(
                    c2,
                    a2,
                    max_number_of_refinements=1,
                    save_every=2,
                    save_dir=root,
                )

                self.assertEqual(len(a2.adaptation_errors), 3)
                collation_len_phase2 = len(c2.get_collation_result())
                self.assertGreaterEqual(collation_len_phase2, collation_len_phase1)

                with open(root / "checkpoint.json", "r", encoding="utf-8") as f:
                    ckpt = json.load(f)
                self.assertEqual(ckpt["iteration"], 3)
                self.assertEqual(ckpt["total_samples"], collation_len_phase2)

    def test_resume_bypasses_morris_screening(self):
        call_count = 0

        def dummy_eval(point):
            nonlocal call_count
            call_count += 1
            return {
                "energy_uj": float(point["N_THREADS"] * 100.0),
                "time": float(point["N_THREADS"] * 0.1),
            }

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            with patch("src.wrappers.base_wrapper.prepare_and_execute", side_effect=self.mock_execute):
                # Phase 1: create with Morris screening
                c1, a1 = energyuq.create(
                    NONE,
                    self.machine,
                    dir=root.as_posix(),
                    screen_morris=True,
                    morris_r=2,
                    evaluate_fn=dummy_eval,
                )
                initial_evals = call_count
                self.assertGreater(initial_evals, 0)
                self.assertTrue((root / "morris_screening.msgpack").exists())
                self.assertIsInstance(getattr(c1, "morris_screening", None), MorrisScreeningResult)

                # Phase 2: Resume
                call_count = 0  # reset counter
                c2, a2 = energyuq.create(
                    NONE,
                    self.machine,
                    dir=root.as_posix(),
                    resume=True,
                    evaluate_fn=dummy_eval,
                )

                # Morris evaluate_fn should NOT be called at all during resume!
                self.assertEqual(call_count, 0)
                self.assertIsInstance(getattr(c2, "morris_screening", None), MorrisScreeningResult)

    def test_resume_nonexistent_directory_raises(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            non_existent = Path(tmpdir) / "does_not_exist"
            with self.assertRaises(FileNotFoundError):
                energyuq.create(
                    NONE,
                    self.machine,
                    dir=non_existent.as_posix(),
                    resume=True,
                )


if __name__ == "__main__":
    unittest.main()
