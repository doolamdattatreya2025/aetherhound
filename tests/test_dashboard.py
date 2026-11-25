"""Tests for the terminal dashboard."""
import os
import re
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import dashboard as db   # noqa: E402
import detector as dt    # noqa: E402
import signals as sg     # noqa: E402

ANSI = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")


def run(channel, seconds, seed=2, log=None):
    rng = np.random.default_rng(seed)
    frames, acoustic = db.build_recording(channel, 30, rng)
    dash = db.Dashboard(channel, acoustic, log)
    for f in frames[:int(seconds / sg.FRAME_S)]:
        dash.feed(f)
    return dash, frames


class DashboardTests(unittest.TestCase):
    def test_screen_fits_the_terminal(self):
        dash, _ = run("ultrasonic", 12)
        lines = len(dash.render(100, 30).split("\n"))
        self.assertLessEqual(lines, 30)     # never taller than the terminal
        self.assertGreaterEqual(lines, 28)  # and uses most of it

    def test_calibrating_message_at_start(self):
        dash, _ = run("ultrasonic", 1)
        self.assertIn("Calibrating", ANSI.sub("", dash.render(100, 30)))

    def test_quiet_room_shows_no_alert(self):
        dash, _ = run("ultrasonic", 7)
        text = ANSI.sub("", dash.render(100, 30))
        self.assertIn("No covert channel found", text)
        self.assertNotIn("DETECTED", text)

    def test_every_channel_shows_alert(self):
        for ch in db.CHANNELS:
            with self.subTest(channel=ch):
                dash, _ = run(ch, 22)
                self.assertIn("COVERT CHANNEL DETECTED", ANSI.sub("", dash.render(100, 30)))

    def test_alert_names_the_right_detector(self):
        dash, _ = run("fan", 22)
        text = ANSI.sub("", dash.render(100, 30))
        self.assertRegex(text, r"ALERT Fan-speed")
        self.assertNotRegex(text, r"ALERT Ultrasonic")

    def test_log_file_records_alert(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "det.log")
            run("ultrasonic", 22, log=path)
            with open(path, encoding="utf-8") as f:
                content = f.read()
        self.assertIn("Ultrasonic", content)
        self.assertIn("RAISED", content)
        self.assertIn("18.75 kHz", content)

    def test_downsample_keeps_peak(self):
        row = np.zeros(1024)
        row[500] = 30
        cols = db.downsample(row, 100)
        self.assertEqual(len(cols), 100)
        self.assertEqual(cols.max(), 30)

    def test_display_meets_30_fps(self):
        rng = np.random.default_rng(1)
        frames, acoustic = db.build_recording("ultrasonic", 30, rng)
        dash = db.Dashboard("ultrasonic", acoustic)
        self.assertGreaterEqual(db.bench(dash, frames), 30)

    def test_detector_refactor_matches_batch_run(self):
        rng = np.random.default_rng(4)
        quiet, _ = sg.make_acoustic(None, 8, rng)
        attack, _ = sg.make_acoustic("disk", 20, rng)
        frames = np.vstack([quiet, attack])
        batch = [(e.detector, e.kind) for e in dt.monitor_acoustic(frames)[0]]
        dets, hold = dt.make_acoustic_detectors()
        live = []
        for i, f in enumerate(frames):
            live += [(e.detector, e.kind) for e in dt.step_detectors(dets, f, 9.0, (i + 1) * sg.FRAME_S, hold)]
        self.assertEqual(batch, live)


if __name__ == "__main__":
    unittest.main()
