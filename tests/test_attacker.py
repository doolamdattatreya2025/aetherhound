"""Tests for the attacker side: framing, sender and receiver."""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import attacker as at   # noqa: E402
import signals as sg    # noqa: E402

SECRET = b"KEY-9F3A-77C1"


class FramingTests(unittest.TestCase):
    def test_bytes_bits_round_trip(self):
        data = bytes(range(0, 256, 7))
        self.assertEqual(at.bits_to_bytes(at.bytes_to_bits(data)), data)

    def test_crc_changes_when_data_changes(self):
        self.assertNotEqual(at.crc8(b"abc"), at.crc8(b"abd"))

    def test_frame_round_trip(self):
        bits = at.build_frame(SECRET)
        self.assertEqual(bits[:len(at.HEADER)], at.HEADER)
        payload, ok = at.parse_frame(bits[len(at.HEADER):])
        self.assertTrue(ok)
        self.assertEqual(payload, SECRET)

    def test_flipped_bit_fails_crc(self):
        bits = at.build_frame(SECRET)
        bits[len(at.HEADER) + 20] ^= 1
        _, ok = at.parse_frame(bits[len(at.HEADER):])
        self.assertFalse(ok)

    def test_empty_payload_rejected(self):
        with self.assertRaises(ValueError):
            at.build_frame(b"")


class ChannelTests(unittest.TestCase):
    def test_secret_arrives_on_every_channel(self):
        rng = np.random.default_rng(3)
        for tech in sg.ACOUSTIC_TECHNIQUES + sg.RF_TECHNIQUES:
            with self.subTest(channel=tech):
                r = at.run_attack(tech, SECRET, rng)
                self.assertTrue(r["success"])
                self.assertEqual(r["bit_errors"], 0)

    def test_data_rate_matches_symbol_length(self):
        rng = np.random.default_rng(3)
        r = at.run_attack("ultrasonic", SECRET, rng)
        self.assertAlmostEqual(r["raw_bps"], 1 / 0.120, places=3)
        self.assertLess(r["goodput_bps"], r["raw_bps"])   # framing costs bits

    def test_very_weak_signal_fails(self):
        rng = np.random.default_rng(3)
        r = at.run_attack("ultrasonic", SECRET, rng, level=-90)
        self.assertFalse(r["success"])
        self.assertGreater(r["ber"], 0.1)

    def test_pure_noise_does_not_decode(self):
        rng = np.random.default_rng(3)
        frames, _ = sg.make_acoustic(None, 20, rng)
        rx = at.receive("ultrasonic", frames)
        self.assertIsNone(rx["payload"])

    def test_detector_sees_the_attack(self):
        import detector as dt
        rng = np.random.default_rng(3)
        r = at.run_attack("ultrasonic", SECRET, rng)
        quiet, _ = sg.make_acoustic(None, 8, rng)      # the detector needs a quiet start
        events, _ = dt.monitor_acoustic(np.vstack([quiet, r["frames"]]))
        self.assertTrue(any(e.kind == "raised" for e in events))


if __name__ == "__main__":
    unittest.main()
