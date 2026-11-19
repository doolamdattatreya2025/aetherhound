"""Unit tests for the AetherHound signal generators and detectors.

Run from the project folder with:
    python -m unittest discover -s tests -v
"""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import signals as sg                      # noqa: E402
import detector as dt                     # noqa: E402

QUIET_S, ATTACK_S = 8, 25                 # quiet room first, so the baseline is clean


def raised(events):
    return [e for e in events if e.kind == "raised"]


class SignalTests(unittest.TestCase):
    def test_acoustic_frame_shape(self):
        frames, bits = sg.make_acoustic("ultrasonic", 5, np.random.default_rng(0))
        self.assertEqual(frames.shape[1], sg.AUDIO_BINS)
        self.assertGreater(len(bits), 0)

    def test_quiet_room_returns_no_bits(self):
        _, bits = sg.make_acoustic(None, 2, np.random.default_rng(0))
        self.assertIsNone(bits)

    def test_rf_frame_shape(self):
        frames, _ = sg.make_rf("carrier", 5, np.random.default_rng(0))
        self.assertEqual(frames.shape[1], sg.RF_BINS)

    def test_pure_tone_level(self):
        """A tone of amplitude A should read about 20*log10(A) dB."""
        t = np.arange(sg.FS) / sg.FS
        tone = 0.01 * np.sin(2 * np.pi * 3_000 * t)
        frames = sg.acoustic_frames(tone)
        self.assertAlmostEqual(frames[:, round(3_000 / sg.BIN_HZ)].max(), -40, delta=1.5)

    def test_ultrasonic_frequencies(self):
        """The two FSK tones sit 250 Hz either side of the carrier."""
        quiet, _ = sg.make_acoustic(None, 20, np.random.default_rng(1))
        frames, _ = sg.make_acoustic("ultrasonic", 20, np.random.default_rng(1))
        lift = frames.mean(axis=0) - quiet.mean(axis=0)
        top = sorted(np.argsort(lift)[-2:] * sg.BIN_HZ)
        self.assertAlmostEqual(top[0], 18_750, delta=60)
        self.assertAlmostEqual(top[1], 19_250, delta=60)


class DetectionTests(unittest.TestCase):
    def run_acoustic(self, technique, seed, thr=9.0):
        rng = np.random.default_rng(seed)
        quiet, _ = sg.make_acoustic(None, QUIET_S, rng)
        attack, _ = sg.make_acoustic(technique, ATTACK_S, rng)
        return dt.monitor_acoustic(np.vstack([quiet, attack]), thr)

    def run_rf(self, technique, seed, thr=9.0):
        rng = np.random.default_rng(seed)
        quiet, _ = sg.make_rf(None, QUIET_S, rng)
        attack, _ = sg.make_rf(technique, ATTACK_S, rng, fc_mhz=50)
        return dt.monitor_rf(np.vstack([quiet, attack]), thr)

    def assert_detected(self, events, detector_name, max_delay_s=10):
        hits = [e for e in raised(events) if e.detector == detector_name and e.time >= QUIET_S]
        self.assertTrue(hits, f"{detector_name} did not raise an alert")
        self.assertLess(hits[0].time - QUIET_S, max_delay_s)

    def test_ultrasonic_detected(self):
        events, _ = self.run_acoustic("ultrasonic", 11)
        self.assert_detected(events, "Ultrasonic")

    def test_fan_detected(self):
        events, _ = self.run_acoustic("fan", 12)
        self.assert_detected(events, "Fan-speed")

    def test_disk_detected(self):
        events, _ = self.run_acoustic("disk", 13)
        self.assert_detected(events, "Disk-seek")

    def test_carrier_detected(self):
        events, _ = self.run_rf("carrier", 14)
        self.assert_detected(events, "RF")

    def test_video_cable_reports_linked_lines(self):
        events, _ = self.run_rf("video", 15)
        self.assert_detected(events, "RF")
        text = [e.text for e in raised(events) if e.time >= QUIET_S][0]
        self.assertIn("linked", text)
        for line in ("50.00 MHz", "75.00 MHz", "100.00 MHz"):
            self.assertIn(line, text)

    def test_ultrasonic_lines_labelled_fsk(self):
        """The alert can fire on the first tone. Once both tones have appeared
        the detector should label them as an FSK pair."""
        _, detectors = self.run_acoustic("ultrasonic", 11)
        labels = {c["label"] for c in detectors[0].clusters if c["key"]}
        self.assertIn("FSK", labels)

    def test_quiet_room_no_false_alarms(self):
        rng = np.random.default_rng(21)
        acoustic, _ = sg.make_acoustic(None, 90, rng)
        rf, _ = sg.make_rf(None, 90, rng)
        self.assertEqual(raised(dt.monitor_acoustic(acoustic)[0]), [])
        self.assertEqual(raised(dt.monitor_rf(rf)[0]), [])

    def test_steady_tone_is_listed_but_not_alerted(self):
        """A new steady tone is not data, so it must not raise an alert."""
        rng = np.random.default_rng(31)
        samples = sg.room_noise(QUIET_S + ATTACK_S, rng=rng)
        start = QUIET_S * sg.FS
        t = np.arange(len(samples) - start) / sg.FS
        samples[start:] += 0.005 * np.sin(2 * np.pi * 21_000 * t)
        events, detectors = dt.monitor_acoustic(sg.acoustic_frames(samples))
        self.assertEqual(raised(events), [])
        steady = [c for c in detectors[0].clusters if c["kind"] == 1]
        self.assertTrue(steady, "steady tone was not listed")

    def test_lower_sensitivity_misses_weak_emitter(self):
        """A weak emitter is caught at +9 dB but missed at +15 dB."""
        rng = np.random.default_rng(41)
        quiet, _ = sg.make_acoustic(None, QUIET_S, rng)
        attack, _ = sg.make_acoustic("ultrasonic", ATTACK_S, rng, level_dbfs=-58)
        frames = np.vstack([quiet, attack])
        caught = raised(dt.monitor_acoustic(frames, thr=9)[0])
        missed = raised(dt.monitor_acoustic(frames, thr=15)[0])
        self.assertTrue(caught)
        self.assertEqual(missed, [])


if __name__ == "__main__":
    unittest.main()
