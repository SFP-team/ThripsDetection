import json
import tempfile
import unittest
from pathlib import Path

from annotator.predict_jobs import (
    TILED_MISSING,
    crop_tile_rect,
    inspect_raw_folder,
    inspect_tiled_folder,
    list_prediction_runs,
    plant_detail,
    remote_media_path,
    resolve_prediction_media,
    start_prediction_from_tiled_upload,
    start_prediction_from_upload,
    unique_basenames,
    uses_remote_media,
)
from annotator.predict_report import (
    EXPORT_FIELDS,
    assemble_plants,
    compare_severity,
    distinct_top_tiles,
    priority_group,
    severity_group,
)


class SeverityDisplayTests(unittest.TestCase):
    def test_bands_follow_handoff_cutpoints(self):
        self.assertEqual(severity_group(1.00), "Minimal visible damage")
        self.assertEqual(severity_group(1.49), "Minimal visible damage")
        self.assertEqual(severity_group(1.50), "Mild visible damage")
        self.assertEqual(severity_group(2.49), "Mild visible damage")
        self.assertEqual(severity_group(3.00), "Moderate visible damage")
        self.assertEqual(severity_group(4.00), "Severe visible damage")
        self.assertEqual(severity_group(4.50), "Very severe visible damage")

    def test_priority_bands_for_ten_plants(self):
        groups = [priority_group(rank, 10) for rank in range(1, 11)]
        self.assertEqual(groups.count("high"), 2)
        self.assertEqual(groups.count("medium"), 3)
        self.assertEqual(groups.count("low"), 5)

    def test_single_plant_is_high_in_that_upload(self):
        self.assertEqual(priority_group(1, 1), "high")


class AssembleTests(unittest.TestCase):
    def test_failed_segment_never_gets_score_one(self):
        plants = assemble_plants(
            ["IMG_GOOD.JPG", "IMG_FAIL.JPG"],
            [
                {
                    "image": "IMG_GOOD.JPG",
                    "predicted_expected": "3.2",
                    "predicted_score": "3",
                    "p_score_1": "0.05",
                    "p_score_2": "0.10",
                    "p_score_3": "0.70",
                    "p_score_4": "0.10",
                    "p_score_5": "0.05",
                    "confidence": "0.70",
                    "needs_review": "false",
                }
            ],
            tile_rows=[{"image": "IMG_GOOD.JPG", "tile": "a.jpg", "x": 0, "y": 0, "p_flush": 0.9, "expected_damage": 1.2}],
            image_rows=[{"image": "IMG_GOOD.JPG"}],
        )
        failed = next(row for row in plants if row["image"] == "IMG_FAIL.JPG")
        scored = next(row for row in plants if row["image"] == "IMG_GOOD.JPG")
        self.assertEqual(failed["status"], "failed")
        self.assertEqual(failed["failure"], "no_plant_detected")
        self.assertIsNone(failed["predicted_score"])
        self.assertIsNone(failed["predicted_expected"])
        self.assertNotEqual(failed["predicted_score"], 1)
        self.assertEqual(scored["rank"], 1)
        self.assertEqual(scored["predicted_score"], 3)

    def test_no_foliage_is_failed_not_healthy(self):
        plants = assemble_plants(
            ["IMG_EMPTY.JPG"],
            [],
            tile_rows=[],
            image_rows=[{"image": "IMG_EMPTY.JPG"}],
        )
        self.assertEqual(plants[0]["failure"], "no_foliage_tiles")
        self.assertIsNone(plants[0]["predicted_score"])

    def test_export_columns_are_complete(self):
        plants = assemble_plants(
            ["IMG_A.JPG"],
            [
                {
                    "image": "IMG_A.JPG",
                    "predicted_expected": "2.2",
                    "predicted_score": "2",
                    "p_score_1": "0.1",
                    "p_score_2": "0.6",
                    "p_score_3": "0.2",
                    "p_score_4": "0.05",
                    "p_score_5": "0.05",
                    "confidence": "0.6",
                    "needs_review": "false",
                }
            ],
            tile_rows=[{"image": "IMG_A.JPG", "tile": "a.jpg", "x": 0, "y": 0}],
            image_rows=[{"image": "IMG_A.JPG"}],
        )
        self.assertTrue(set(EXPORT_FIELDS).issubset(plants[0]))

    def test_low_confidence_is_flagged(self):
        plants = assemble_plants(
            ["IMG_A.JPG"],
            [
                {
                    "image": "IMG_A.JPG",
                    "predicted_expected": "2.0",
                    "predicted_score": "2",
                    "confidence": "0.40",
                    "needs_review": "false",
                    "p_score_1": "0.4",
                    "p_score_2": "0.3",
                    "p_score_3": "0.1",
                    "p_score_4": "0.1",
                    "p_score_5": "0.1",
                }
            ],
            tile_rows=[{"image": "IMG_A.JPG", "tile": "a.jpg"}],
            image_rows=[{"image": "IMG_A.JPG"}],
        )
        self.assertTrue(plants[0]["needs_review"])

    def test_evidence_tiles_prefer_distant_high_flush_damage(self):
        tiles = [
            {"tile": "near_a", "x": 0, "y": 0, "p_flush": 0.9, "expected_damage": 2.0},
            {"tile": "near_b", "x": 10, "y": 10, "p_flush": 0.9, "expected_damage": 1.9},
            {"tile": "far", "x": 800, "y": 0, "p_flush": 0.8, "expected_damage": 1.5},
            {"tile": "mature", "x": 400, "y": 400, "p_flush": 0.05, "expected_damage": 2.0},
        ]
        selected = [row["tile"] for row in distinct_top_tiles(tiles, count=2)]
        self.assertEqual(selected, ["near_a", "far"])


class FolderGuardTests(unittest.TestCase):
    def test_raw_folder_rejects_tiled_marker(self):
        with tempfile.TemporaryDirectory() as raw:
            folder = Path(raw)
            (folder / "tiles_foliage.csv").write_text("image,tile\n")
            (folder / "photo.jpg").write_bytes(b"x")
            with self.assertRaisesRegex(RuntimeError, "already tiled"):
                inspect_raw_folder(folder)

    def test_raw_folder_rejects_duplicate_basenames(self):
        with tempfile.TemporaryDirectory() as raw:
            folder = Path(raw)
            (folder / "a").mkdir()
            (folder / "b").mkdir()
            (folder / "a" / "IMG_1.JPG").write_bytes(b"x")
            (folder / "b" / "IMG_1.JPG").write_bytes(b"y")
            with self.assertRaisesRegex(RuntimeError, "Duplicate filename"):
                inspect_raw_folder(folder)

    def test_tiled_folder_requires_manifest(self):
        with tempfile.TemporaryDirectory() as raw:
            with self.assertRaisesRegex(FileNotFoundError, "tiles_foliage.csv"):
                inspect_tiled_folder(Path(raw))

    def test_unique_basenames_accepts_distinct_files(self):
        paths = [Path("one.JPG"), Path("two.JPG")]
        self.assertEqual(unique_basenames(paths), paths)

    def test_raw_upload_rejects_tiled_marker_without_gpu(self):
        with self.assertRaisesRegex(RuntimeError, "already tiled"):
            start_prediction_from_upload("lab", "x", [("run/tiles_foliage.csv", b"image,tile\n")])

    def test_tiled_upload_requires_marker_without_gpu(self):
        with self.assertRaisesRegex(FileNotFoundError, "tiles_foliage.csv"):
            start_prediction_from_tiled_upload("lab", "x", [("run/photo.jpg", b"x")])
        self.assertIn("tiles_foliage.csv", TILED_MISSING)


class SourceNameTests(unittest.TestCase):
    def test_source_names_union_keeps_failed_plants(self):
        from annotator import predict_jobs

        with tempfile.TemporaryDirectory() as raw:
            original = predict_jobs.PREDICTIONS
            predict_jobs.PREDICTIONS = Path(raw)
            try:
                incoming = Path(raw) / "run" / "incoming"
                incoming.mkdir(parents=True)
                (incoming / "images.csv").write_text(
                    "image,tiles_kept\nGOOD.JPG,2\nFAIL.JPG,0\n",
                    encoding="utf-8",
                )
                (incoming / "tiles_foliage.csv").write_text(
                    "image,tile,decision\nGOOD.JPG,a.jpg,keep\n",
                    encoding="utf-8",
                )
                names = predict_jobs.source_image_names("run")
                self.assertEqual(names, ["GOOD.JPG", "FAIL.JPG"])
            finally:
                predict_jobs.PREDICTIONS = original

    def test_list_hides_underscore_fixture_runs(self):
        from annotator import predict_jobs

        with tempfile.TemporaryDirectory() as raw:
            original = predict_jobs.PREDICTIONS
            predict_jobs.PREDICTIONS = Path(raw)
            try:
                fixture = Path(raw) / "_ui_fixture"
                real = Path(raw) / "20260826-real"
                fixture.mkdir()
                real.mkdir()
                (fixture / "run.json").write_text(
                    '{"run_id":"_ui_fixture","source":"fixture","created_at":"2026-01-01"}',
                    encoding="utf-8",
                )
                (real / "run.json").write_text(
                    '{"run_id":"20260826-real","source":"tiled","created_at":"2026-01-02"}',
                    encoding="utf-8",
                )
                ids = [item["run_id"] for item in list_prediction_runs()]
                self.assertEqual(ids, ["20260826-real"])
            finally:
                predict_jobs.PREDICTIONS = original


class RemoteMediaTests(unittest.TestCase):
    def test_remote_paths_stay_on_the_server(self):
        from annotator import predict_jobs

        with tempfile.TemporaryDirectory() as raw:
            original = predict_jobs.PREDICTIONS
            predict_jobs.PREDICTIONS = Path(raw)
            try:
                run = Path(raw) / "w25"
                run.mkdir()
                (run / "run.json").write_text(
                    json.dumps(
                        {
                            "run_id": "w25",
                            "media_source": "remote",
                            "remote_images": "/home/fpt/ThripsDetection/RoverImages",
                            "remote_run": "/home/fpt/ThripsDetection/prediction_runs/job/run",
                        }
                    ),
                    encoding="utf-8",
                )
                incoming = run / "incoming"
                incoming.mkdir()
                (incoming / "images.csv").write_text(
                    "image,box_x0,box_y0,box_w,box_h\nW25010101R.jpg,10,20,100,200\n",
                    encoding="utf-8",
                )
                (incoming / "plant_severity.csv").write_text(
                    "image,predicted_expected,predicted_score,confidence,needs_review,"
                    "p_score_1,p_score_2,p_score_3,p_score_4,p_score_5\n"
                    "W25010101R.jpg,2.2,2,0.7,false,0.1,0.6,0.2,0.05,0.05\n",
                    encoding="utf-8",
                )
                (incoming / "tiles_foliage.csv").write_text(
                    "image,tile,x,y,width,height,decision\n"
                    "W25010101R.jpg,W25010101R_x10_y20.jpg,10,20,512,512,keep\n",
                    encoding="utf-8",
                )
                self.assertTrue(uses_remote_media("w25"))
                self.assertEqual(
                    remote_media_path("w25", "original", "W25010101R.jpg"),
                    "/home/fpt/ThripsDetection/RoverImages/W25010101R.jpg",
                )
                self.assertEqual(
                    remote_media_path("w25", "crop", "W25010101R.jpg"),
                    "/home/fpt/ThripsDetection/prediction_runs/job/run/plant_crops/W25010101R_plant.jpg",
                )
                source, location = resolve_prediction_media("w25", "crop", "W25010101R.jpg")
                self.assertEqual(source, "remote")
                self.assertEqual(location, remote_media_path("w25", "crop", "W25010101R.jpg"))
                detail = plant_detail("w25", "W25010101R.jpg")
                self.assertTrue(detail["media"]["has_crop"])
                self.assertTrue(detail["media"]["has_original"])
                self.assertTrue(detail["media"]["has_tile_images"])
                self.assertEqual(detail["media"]["source"], "remote")
            finally:
                predict_jobs.PREDICTIONS = original

    def test_remote_dir_rejects_relative_paths(self):
        from annotator.predict_jobs import _safe_remote_dir

        with self.assertRaises(ValueError):
            _safe_remote_dir("RoverImages")
        with self.assertRaises(ValueError):
            _safe_remote_dir("/home/fpt/../etc")
        self.assertEqual(
            _safe_remote_dir("/home/fpt/ThripsDetection/RoverImages"),
            "/home/fpt/ThripsDetection/RoverImages",
        )


class OverlayRectTests(unittest.TestCase):
    def test_tile_maps_into_crop_pixels(self):
        image = {"box_x0": 737, "box_y0": 784, "box_w": 1756, "box_h": 3673}
        top_left = crop_tile_rect(
            {"x": 737, "y": 784, "width": 512, "height": 512},
            image,
            1756,
            3673,
        )
        self.assertEqual(top_left["crop_x"], 0)
        self.assertEqual(top_left["crop_y"], 0)
        self.assertEqual(top_left["crop_w"], 512)
        self.assertEqual(top_left["crop_h"], 512)

    def test_right_edge_tile_stays_inside_crop(self):
        image = {"box_x0": 737, "box_y0": 784, "box_w": 1756, "box_h": 3673}
        rect = crop_tile_rect(
            {"x": 1981, "y": 784, "width": 512, "height": 512},
            image,
            1756,
            3673,
        )
        self.assertAlmostEqual(rect["crop_x"], 1244)
        self.assertAlmostEqual(rect["crop_x"] + rect["crop_w"], 1756)

    def test_tile_outside_crop_is_dropped(self):
        image = {"box_x0": 737, "box_y0": 784, "box_w": 1756, "box_h": 3673}
        self.assertIsNone(
            crop_tile_rect({"x": 0, "y": 0, "width": 512, "height": 512}, image, 1756, 3673)
        )


class CompareTests(unittest.TestCase):
    def test_compare_severity_matches_on_image(self):
        rows = [
            {"image": "A.JPG", "predicted_expected": "2.1000", "predicted_score": "2"},
            {"image": "B.JPG", "predicted_expected": "4.0000", "predicted_score": "4"},
        ]
        result = compare_severity(rows, rows)
        self.assertTrue(result["ok"])
        self.assertEqual(result["matched"], 2)

    def test_compare_severity_flags_drift(self):
        actual = [{"image": "A.JPG", "predicted_expected": "2.5", "predicted_score": "3"}]
        expected = [{"image": "A.JPG", "predicted_expected": "2.1", "predicted_score": "2"}]
        result = compare_severity(actual, expected)
        self.assertFalse(result["ok"])
        self.assertEqual(len(result["mismatches"]), 1)


if __name__ == "__main__":
    unittest.main()
