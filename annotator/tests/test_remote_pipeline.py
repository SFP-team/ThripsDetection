import importlib.util
import io
import os
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

from annotator import predict_jobs
from annotator.sessions import make_key


class LocalShell:
    """Stands in for a paramiko client: runs the remote command with local bash."""

    def exec_command(self, command: str, get_pty: bool = False):
        proc = subprocess.run(["bash", "-c", command], capture_output=True)
        stdout = io.BytesIO(proc.stdout)
        stdout.channel = mock.Mock()
        stdout.channel.recv_exit_status.return_value = proc.returncode
        return io.BytesIO(b""), stdout, io.BytesIO(proc.stderr)


def status(job: Path, run: Path, pid: int | None) -> dict:
    with mock.patch.object(predict_jobs, "TRAINING_PYTHON", sys.executable):
        return predict_jobs._remote_pipeline_status(LocalShell(), str(job), str(run), pid)


class RemotePipelineStatusTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.job = Path(self.tmp.name) / "job"
        self.run = self.job / "run"
        self.run.mkdir(parents=True)

    def tearDown(self):
        self.tmp.cleanup()

    def test_failed_marker_wins(self):
        (self.job / "FAILED").write_text("Plant cut failed. See pipeline.log.\n")
        payload = status(self.job, self.run, None)
        self.assertTrue(payload["failed"])
        self.assertEqual(payload["detail"], "Plant cut failed. See pipeline.log.")

    def test_done_needs_marker_and_scores(self):
        (self.job / "DONE").write_text("DONE\n")
        self.assertFalse(status(self.job, self.run, None)["done"])
        (self.run / "plant_severity.csv").write_text("image\n")
        self.assertTrue(status(self.job, self.run, None)["done"])

    def test_dead_process_without_markers_fails_the_job(self):
        proc = subprocess.Popen(["sleep", "0"])
        proc.wait()
        payload = status(self.job, self.run, proc.pid)
        self.assertTrue(payload["failed"])
        self.assertIn("stopped before it finished", payload["detail"])

    def test_live_process_keeps_running(self):
        proc = subprocess.Popen(["sleep", "30"])
        try:
            (self.job / "progress.json").write_text('{"step": "segmenting", "detail": "Cutting", "done": false}\n')
            payload = status(self.job, self.run, proc.pid)
        finally:
            proc.kill()
            proc.wait()
        self.assertFalse(payload["failed"])
        self.assertFalse(payload["done"])
        self.assertEqual(payload["step"], "segmenting")

    def test_unknown_pid_fails_only_after_a_long_stall(self):
        progress = self.job / "progress.json"
        progress.write_text('{"step": "scoring_tiles", "detail": "Scoring", "done": false}\n')
        self.assertFalse(status(self.job, self.run, None)["failed"])
        stale = time.time() - predict_jobs.STALL_SECONDS - 60
        os.utime(progress, (stale, stale))
        os.utime(self.job, (stale, stale))
        payload = status(self.job, self.run, None)
        self.assertTrue(payload["failed"])
        self.assertIn("written nothing", payload["detail"])


class RunIdTests(unittest.TestCase):
    def test_key_is_unique_within_the_folder_it_will_live_in(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = make_key("W25 rover", root)
            (root / first).mkdir()
            second = make_key("W25 rover", root)
            self.assertNotEqual(first, second)
            self.assertTrue(second.startswith(first))
            other_root = root / "elsewhere"
            other_root.mkdir()
            self.assertEqual(make_key("W25 rover", other_root), first)


@unittest.skipUnless(
    importlib.util.find_spec("numpy"), "segment_and_tile needs numpy"
)
class SegmentResumeTests(unittest.TestCase):
    def test_partial_tiles_for_unconfirmed_photo_are_dropped(self):
        sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
        import segment_and_tile as seg

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            tile = {"image": "", "tile": "", "x": 0, "y": 0, "width": 512, "height": 512,
                    "mask_frac": 0.5, "seed_method": "s", "mask_used": "selected"}
            rows = [
                {**tile, "image": "a.jpg", "tile": "a_x0_y0.jpg"},
                {**tile, "image": "b.jpg", "tile": "b_x0_y0.jpg"},
            ]
            image = {name: 0 for name in seg.IMAGE_FIELDS}
            image.update(image="a.jpg", tiles_kept=1)
            seg.write_csv(out / "tiles.csv", seg.TILE_FIELDS, rows)
            seg.write_csv(out / "images.csv", seg.IMAGE_FIELDS, [image])
            (out / "tiles.csv.tmp").write_text("torn")

            resumed_rows, resumed_images = seg.load_resume_state(out)
            self.assertEqual([row["image"] for row in resumed_images], ["a.jpg"])
            self.assertEqual([row["tile"] for row in resumed_rows], ["a_x0_y0.jpg"])

            seg.checkpoint(out, resumed_rows, resumed_images, 2, "a.jpg")
            self.assertFalse((out / "tiles.csv.tmp").exists())
            self.assertFalse((out / "images.csv.tmp").exists())
            self.assertIn('"done": 1', (out / "segment_progress.json").read_text())


if __name__ == "__main__":
    unittest.main()
