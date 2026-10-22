"""Offline checks for gallery presentation and viewer navigation."""

import html
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import quote

from PIL import Image

import avatar_design
import avatar_studio as studio
from test_avatar_logic import app_function


class AvatarViewTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = self.temp.name
        self.avatar = studio.create_avatar(self.root, "View QA")
        for index in range(3):
            path = studio.save_gallery_image(self.root, self.avatar["id"], f"view-{index}", Image.new("RGB", (64, 96)))
            studio.record_gallery_item(self.root, self.avatar["id"], {
                "id": f"view-{index}", "generated_path": path, "origin": "console", "status": "unchecked",
                "prompt": '<script>alert("not executable")</script>', "attempt": 1,
            })

    def test_viewer_includes_image_download_and_escaped_collapsed_prompt(self):
        namespace = {"avatar_studio": studio, "SAVE_DIR": self.root, "html": html, "os": os, "quote": quote}
        render = app_function("_avatar_gallery_details", namespace)
        result = render(self.avatar, 1)
        self.assertIn('class="av-image-canvas"', result)
        self.assertIn('alt="Selected avatar image"', result)
        self.assertIn('2 of 3', result)
        self.assertIn('download>Download original</a>', result)
        self.assertIn('data-av-disclosure="prompt-view-1"', result)
        self.assertNotIn('<script>', result)
        self.assertIn('&lt;script&gt;', result)

    def test_viewer_navigation_wraps_without_reordering_saved_items(self):
        namespace = {"avatar_studio": studio, "SAVE_DIR": self.root,
                     "gallery_selection_payload": lambda avatar_id, index: index}
        move = app_function("move_gallery_selection", namespace)
        before = [item["id"] for item in studio.visible_gallery_items(self.root, self.avatar["id"])]
        self.assertEqual(move(self.avatar["id"], 0, -1), 2)
        self.assertEqual(move(self.avatar["id"], 2, 1), 0)
        self.assertIsNone(move(None, None, 1))
        self.assertEqual(before, [item["id"] for item in studio.visible_gallery_items(self.root, self.avatar["id"])])

    def test_idle_gallery_does_not_display_a_wall_of_zero_counters(self):
        result = avatar_design.gallery_pipeline({"job": None, "counts": {"found": 0}})
        self.assertIn("No batch in progress", result)
        self.assertNotIn("av-pipeline-counts", result)

    def test_saved_findings_and_activity_remain_accessible(self):
        result = avatar_design.gallery_pipeline({
            "job": {"id": "batch-1", "status": "failed", "phase": "cleaning", "message": "API <unavailable>"},
            "counts": {"found": 30, "approved": 0},
            "events": [{"time": "2026-10-22T12:00:00", "message": "Saved findings"}],
        })
        self.assertIn("<b>30</b>Found", result)
        self.assertIn('data-av-disclosure="activity-batch-1"', result)
        self.assertIn("Saved findings", result)
        self.assertIn("API &lt;unavailable&gt;", result)


if __name__ == "__main__":
    unittest.main()
