import unittest

from annotator.config import Settings, public_settings
from annotator.export_merge import decide_export_action, extras_to_append, merged_rows


def row(image: str, tile: str, injury: str = "injured", annotator: str = "a") -> dict:
    return {
        "image": image,
        "tile": tile,
        "rel_y": "0.1",
        "tissue": "flush",
        "injury": injury,
        "curl": "yes",
        "label": injury,
        "annotator": annotator,
        "labeled_at": "2026-08-18T00:00:00+00:00",
    }


class ExportMergeTests(unittest.TestCase):
    def test_empty_local_does_not_write(self):
        action = decide_export_action([], [])
        self.assertEqual(action["action"], "empty")

    def test_create_when_gpu_has_no_csv(self):
        local = [row("plant_a.jpg", "tile_001.jpg")]
        action = decide_export_action(local, [])
        self.assertEqual(action["action"], "create")
        self.assertEqual(action["extras"], local)

    def test_append_only_new_image_tile_pairs(self):
        remote = [row("plant_a.jpg", "tile_001.jpg", "healthy", "gpu")]
        local = [
            row("plant_a.jpg", "tile_001.jpg", "injured", "local"),
            row("plant_a.jpg", "tile_002.jpg", "injured", "local"),
        ]
        extras = extras_to_append(local, remote)
        self.assertEqual([item["tile"] for item in extras], ["tile_002.jpg"])
        action = decide_export_action(local, [("/exports/labels.csv", remote)])
        self.assertEqual(action["action"], "append")
        self.assertEqual(action["remote_path"], "/exports/labels.csv")
        merged = merged_rows(remote, action["extras"])
        self.assertEqual(merged[0]["injury"], "healthy")
        self.assertEqual(merged[0]["annotator"], "gpu")
        self.assertEqual(len(merged), 2)

    def test_same_tile_different_mark_keeps_gpu_row(self):
        remote = [row("plant_a.jpg", "tile_001.jpg", "healthy", "gpu")]
        local = [row("plant_a.jpg", "tile_001.jpg", "injured", "local")]
        extras = extras_to_append(local, remote)
        self.assertEqual(extras, [])
        merged = merged_rows(remote, extras)
        self.assertEqual(merged, remote)

    def test_new_folder_when_images_do_not_overlap(self):
        remote = [row("plant_a.jpg", "tile_001.jpg")]
        local = [row("plant_b.jpg", "tile_009.jpg")]
        action = decide_export_action(local, [("/exports/labels.csv", remote)])
        self.assertEqual(action["action"], "new_folder")

    def test_overlap_on_any_remote_csv_appends_there(self):
        older = [row("plant_z.jpg", "tile_001.jpg")]
        match = [row("plant_a.jpg", "tile_001.jpg")]
        local = [row("plant_a.jpg", "tile_002.jpg")]
        action = decide_export_action(
            local,
            [("/exports/other/labels.csv", older), ("/exports/labels.csv", match)],
        )
        self.assertEqual(action["action"], "append")
        self.assertEqual(action["remote_path"], "/exports/labels.csv")


class PublicSettingsTests(unittest.TestCase):
    def test_password_stays_off_the_api(self):
        settings = Settings(ssh_password="not-for-the-browser")
        payload = public_settings(settings)
        self.assertNotIn("ssh_password", payload)
        self.assertTrue(payload["ssh_password_set"])
        self.assertTrue(payload["gpu_ready"])

    def test_gpu_card_stays_off_without_password(self):
        payload = public_settings(Settings(ssh_password=""))
        self.assertFalse(payload["gpu_ready"])
        self.assertFalse(payload["ssh_password_set"])


if __name__ == "__main__":
    unittest.main()
