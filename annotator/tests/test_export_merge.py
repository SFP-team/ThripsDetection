import unittest

from annotator.config import Settings, public_settings
from annotator.export_merge import decide_export_action, merge_rows


def row(
    image: str,
    tile: str,
    injury: str = "mild",
    annotator: str = "a",
    labeled_at: str = "2026-08-18T00:00:00+00:00",
) -> dict:
    return {
        "image": image,
        "tile": tile,
        "rel_y": "0.1",
        "tissue": "flush",
        "injury": injury,
        "curl": "",
        "label": injury,
        "protocol_version": "visible_new_growth_damage_v2",
        "annotator": annotator,
        "labeled_at": labeled_at,
    }


class ExportMergeTests(unittest.TestCase):
    def test_empty_local_does_not_write(self):
        action = decide_export_action([], [])
        self.assertEqual(action["action"], "empty")

    def test_create_when_gpu_has_no_csv(self):
        local = [row("plant_a.jpg", "tile_001.jpg")]
        action = decide_export_action(local, [])
        self.assertEqual(action["action"], "create")
        self.assertEqual(action["rows"], local)

    def test_append_adds_new_tiles_and_keeps_order(self):
        remote = [row("plant_a.jpg", "tile_001.jpg", "healthy", "gpu")]
        local = [
            row("plant_a.jpg", "tile_001.jpg", "healthy", "local"),
            row("plant_a.jpg", "tile_002.jpg", "severe", "local"),
        ]
        action = decide_export_action(local, [("/exports/labels.csv", remote)])
        self.assertEqual(action["action"], "append")
        self.assertEqual(action["remote_path"], "/exports/labels.csv")
        self.assertEqual([item["tile"] for item in action["rows"]], ["tile_001.jpg", "tile_002.jpg"])
        self.assertEqual([item["tile"] for item in action["added"]], ["tile_002.jpg"])
        self.assertEqual(action["rows"][0]["annotator"], "gpu")

    def test_newer_local_mark_replaces_gpu_row(self):
        remote = [row("plant_a.jpg", "tile_001.jpg", "healthy", "gpu", "2026-08-18T00:00:00+00:00")]
        local = [row("plant_a.jpg", "tile_001.jpg", "severe", "local", "2026-08-25T09:00:00+00:00")]
        merged = merge_rows(local, remote)
        self.assertEqual(merged["rows"][0]["injury"], "severe")
        self.assertEqual(merged["rows"][0]["annotator"], "local")
        self.assertEqual(len(merged["updated"]), 1)
        self.assertEqual(merged["added"], [])
        self.assertEqual(len(merged["rows"]), 1)

    def test_older_or_equal_local_mark_keeps_gpu_row(self):
        remote = [row("plant_a.jpg", "tile_001.jpg", "healthy", "gpu", "2026-08-25T09:00:00+00:00")]
        older = [row("plant_a.jpg", "tile_001.jpg", "severe", "local", "2026-08-18T00:00:00+00:00")]
        same = [row("plant_a.jpg", "tile_001.jpg", "severe", "local", "2026-08-25T09:00:00+00:00")]
        for local in (older, same):
            merged = merge_rows(local, remote)
            self.assertEqual(merged["rows"], remote)
            self.assertEqual(len(merged["kept"]), 1)
            self.assertEqual(merged["updated"], [])

    def test_retry_of_the_same_export_changes_nothing(self):
        local = [row("plant_a.jpg", "tile_001.jpg"), row("plant_a.jpg", "tile_002.jpg")]
        first = merge_rows(local, [])
        second = merge_rows(local, first["rows"])
        self.assertEqual(second["rows"], first["rows"])
        self.assertEqual(second["added"], [])
        self.assertEqual(second["updated"], [])

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
