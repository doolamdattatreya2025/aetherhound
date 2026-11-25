"""Detectors for AetherHound.

Python version of the detectors in the browser simulator. They only ever see
spectrum frames (dB per frequency bin), never the emitter.

Two kinds:

NarrowbandDetector  watches a range of bins for lines that switch on and off
                    like data. Used for the ultrasonic, fan and RF channels.
BurstDetector       watches total energy in one band and counts bursts.
                    Used for disk seeks, which are not narrowband.

Both learn a baseline during the first CAL_FRAMES frames, so the signal given
to them should start with a stretch of quiet room.
"""
import math
from dataclasses import dataclass

import numpy as np

from signals import AUDIO_BINS, BIN_HZ, FRAME_S, RF_BIN_HZ, RF_BINS

CAL_FRAMES = 140          # frames used to learn the baseline (about 3 s)


def fmt_khz(hz):
    return f"{hz / 1000:.2f} kHz"


def fmt_hz(hz):
    return f"{hz:.0f} Hz"


def fmt_mhz(hz):
    return f"{hz / 1e6:.2f} MHz"


class NarrowbandDetector:
    """Finds keyed lines (data) and steady lines (no alert) in a bin range.

    Per frame:
      1. A bin is "on" if it is more than thr dB above its baseline.
      2. A bin's flag counts only if it was on in two frames in a row.
      3. The last `window` flags of each bin are kept.
      4. A bin that switched at least min_changes times is keyed.
         A bin that is on for a third of the window with 2 or fewer changes is steady.
      5. Neighbouring bins are grouped into one line. A line containing a
         steady bin is steady.
      6. Keyed lines that take turns are FSK. Keyed lines that switch together
         are linked (like the harmonics of a display cable).
      7. Alert after keyed lines are seen for window/12 checks in a row;
         clear after none for window/4 checks.
    A frame where a quarter of the bins jump at once is a click: the previous
    flags are kept for that frame and the next one.
    """

    def __init__(self, name, lo, hi, unit_hz, fmt, gap, window, min_changes=4):
        self.name, self.lo, self.hi, self.m = name, lo, hi, hi - lo
        self.unit_hz, self.fmt, self.gap = unit_hz, fmt, gap
        self.window, self.min_changes = window, min_changes
        self.hold_min = round(window / 12)
        self.miss_max = round(window / 4)
        self.reset()

    def reset(self):
        m, w = self.m, self.window
        self.mean = np.zeros(m)
        self.ring = np.zeros((m, w), dtype=np.uint8)
        self.prev_raw = np.zeros(m, dtype=bool)
        self.prev_flag = np.zeros(m, dtype=np.uint8)
        self.on_cnt = np.zeros(m, dtype=int)
        self.ex_on = np.zeros(m)
        self.frame = 0
        self.calibrated = 0
        self.guard = 0
        self.broad = False
        self.clusters = []
        self.keyed = 0
        self.hold = self.miss = 0
        self.alert = self.raised = self.cleared = False
        self.impulses = 0

    def step(self, db, thr):
        """Process one frame (a full spectrum in dB)."""
        self.raised = self.cleared = False
        self.broad = False
        x = db[self.lo:self.hi]
        if self.calibrated < CAL_FRAMES:
            self.calibrated += 1
            self.mean += (x - self.mean) / self.calibrated
            return
        excess = x - self.mean
        cur = excess > thr
        steady = self.on_cnt > self.window * 0.85
        raw = np.count_nonzero(cur & ~steady)
        hard = raw > 0.25 * self.m
        broad = hard or self.guard > 0
        self.guard = 1 if hard else 0
        self.broad = broad
        if broad:
            flag = self.prev_flag.copy()
            self.impulses += 1
        else:
            flag = (cur & self.prev_raw).astype(np.uint8)
            quiet = ~cur
            self.mean[quiet] += (x[quiet] - self.mean[quiet]) * 0.003
            on = flag.astype(bool)
            self.ex_on[on] = np.where(self.ex_on[on] > 0,
                                      self.ex_on[on] * 0.7 + excess[on] * 0.3,
                                      excess[on])
            self.prev_raw = cur
        head = self.frame % self.window
        old = self.ring[:, head].astype(int)
        self.ring[:, head] = flag
        self.on_cnt += flag.astype(int) - old
        self.prev_flag = flag
        self.frame += 1
        if self.frame % 3 == 0:
            self._analyse()

    def _analyse(self):
        w = self.window
        order = (self.frame % w + np.arange(w)) % w           # oldest to newest
        history = self.ring[:, order]
        changes = np.count_nonzero(history[:, 1:] != history[:, :-1], axis=1)
        enough = self.on_cnt >= 8
        keyed = enough & (changes >= self.min_changes)
        steady = enough & ~keyed & (self.on_cnt >= w * 0.35) & (changes <= 2)
        cls = np.zeros(self.m, dtype=int)
        cls[steady], cls[keyed] = 1, 2

        clusters = []
        for i in np.flatnonzero(cls):
            if clusters and i - clusters[-1]["last"] <= self.gap + 1:
                c = clusters[-1]
                c["last"] = i
                c["kind"] = max(c["kind"], cls[i])
                c["steady"] |= cls[i] == 1
                if self.ex_on[i] > self.ex_on[c["peak"]]:
                    c["peak"] = i
            else:
                clusters.append({"last": i, "peak": i, "kind": cls[i],
                                 "steady": cls[i] == 1, "label": "", "linked": 0})
        for c in clusters:
            if c["steady"]:         # edge bins of a steady line flicker; ignore them
                c["kind"] = 1
            c["label"] = "keyed" if c["kind"] == 2 else "steady"

        keyed_clusters = [c for c in clusters if c["kind"] == 2][:4]
        for a in range(len(keyed_clusters)):
            for b in range(a + 1, len(keyed_clusters)):
                ha, hb = history[keyed_clusters[a]["peak"]], history[keyed_clusters[b]["peak"]]
                either = np.count_nonzero(ha | hb)
                if either <= 10:
                    continue
                ratio = np.count_nonzero(ha & hb) / either
                if ratio < 0.2:
                    keyed_clusters[a]["label"] = keyed_clusters[b]["label"] = "FSK"
                elif ratio > 0.75:
                    keyed_clusters[a]["linked"] += 1
                    keyed_clusters[b]["linked"] += 1

        for c in clusters:
            p = c["peak"]
            c["where"] = self.fmt((self.lo + p) * self.unit_hz)
            c["key"] = c["kind"] == 2
            c["snr"] = float(self.ex_on[p])
            c["duty"] = self.on_cnt[p] / w
            c["short"] = "FSK" if c["label"] == "FSK" else (
                "keyed" + (f" (x{c['linked'] + 1} linked)" if c["linked"] else ""))
        self.clusters = clusters
        self.keyed = sum(c["key"] for c in clusters)
        _update_alert(self)


class BurstDetector:
    """Counts bursts of energy in one band. A burst is a run above baseline.

    Keyed means three or more bursts in the window while the band is on for
    less than 80% of it. Steady means on for 80% or more (no alert).
    """

    def __init__(self, name, lo, hi, window, where, min_bursts=3):
        self.name, self.lo, self.hi, self.window = name, lo, hi, window
        self.where, self.min_bursts = where, min_bursts
        self.hold_min = round(window / 12)
        self.miss_max = round(window / 4)
        self.reset()

    def reset(self):
        self.ring = np.zeros(self.window, dtype=np.uint8)
        self.n = 0
        self.mean = self.m2 = 0.0
        self.sd = 0.3
        self.frame = 0
        self.prev = 0
        self.ex = 0.0
        self.clusters = []
        self.keyed = 0
        self.hold = self.miss = 0
        self.alert = self.raised = self.cleared = False
        self.impulses = 0
        self.broad = False

    def step(self, db, thr, hold=False):
        """hold=True means a click was seen, so keep the previous flag."""
        self.raised = self.cleared = False
        e = 10 * math.log10(float(np.sum(10 ** (db[self.lo:self.hi] / 10))) + 1e-12)
        if self.n < CAL_FRAMES:
            self.n += 1
            d = e - self.mean
            self.mean += d / self.n
            self.m2 += d * (e - self.mean)
            if self.n == CAL_FRAMES:
                self.sd = max(0.15, math.sqrt(self.m2 / (self.n - 1)))
            return
        limit = self.mean + max(thr * 0.45, 5 * self.sd)
        if hold:
            flag = self.prev
            self.impulses += 1
        else:
            flag = int(e > limit)
            if flag:
                self.ex = self.ex * 0.7 + (e - self.mean) * 0.3 if self.ex > 0 else e - self.mean
            else:
                self.mean += (e - self.mean) * 0.002
        self.ring[self.frame % self.window] = flag
        self.prev = flag
        self.frame += 1
        if self.frame % 3 == 0:
            self._analyse()

    def _analyse(self):
        w = self.window
        seq = self.ring[(self.frame % w + np.arange(w)) % w]
        on = int(seq.sum())
        bursts = int(np.count_nonzero((seq[1:] == 1) & (seq[:-1] == 0)))
        keyed = bursts >= self.min_bursts and on < w * 0.8
        steady = on >= w * 0.8
        self.clusters = []
        if keyed or steady:
            self.clusters.append({"kind": 2 if keyed else 1, "key": keyed, "where": self.where,
                                  "short": "bursts", "snr": self.ex, "bursts": bursts})
        self.keyed = int(keyed)
        _update_alert(self)


def _update_alert(det):
    """Raise after enough consecutive keyed checks, clear after enough quiet ones."""
    if det.keyed:
        det.hold, det.miss = det.hold + 1, 0
    else:
        det.hold, det.miss = 0, det.miss + 1
    if not det.alert and det.hold >= det.hold_min:
        det.alert = det.raised = True
    elif det.alert and det.miss >= det.miss_max:
        det.alert = False
        det.cleared = True


@dataclass
class Event:
    time: float          # seconds from the start of the frames
    detector: str
    kind: str            # "raised" or "cleared"
    text: str            # lines found, for a raised alert


def _describe(det):
    return ", ".join(f"{c['where']} {c['short']}" for c in det.clusters if c["key"])


def step_detectors(detectors, db, thr, t, burst_hold_from=None):
    """Feed one frame to every detector. t is the time in seconds at the end of the frame.

    Returns the events (alerts raised or cleared) this frame caused. A live display
    calls this once per frame; _run below calls it for a whole recording.
    """
    events = []
    for det in detectors:
        if isinstance(det, BurstDetector):
            det.step(db, thr, hold=burst_hold_from.broad if burst_hold_from else False)
        else:
            det.step(db, thr)
        if det.raised:
            events.append(Event(t, det.name, "raised", _describe(det)))
        if det.cleared:
            events.append(Event(t, det.name, "cleared", ""))
    return events


def _run(detectors, frames, thr, burst_hold_from=None):
    events = []
    for i, db in enumerate(frames):
        events += step_detectors(detectors, db, thr, (i + 1) * FRAME_S, burst_hold_from)
    return events, detectors


def make_acoustic_detectors():
    """The ultrasonic, fan and disk detectors. Returns (detectors, ultrasonic)."""
    ultrasonic = NarrowbandDetector("Ultrasonic", math.ceil(16_000 / BIN_HZ), math.floor(23_500 / BIN_HZ),
                                    BIN_HZ, fmt_khz, gap=2, window=96)
    fan = NarrowbandDetector("Fan-speed", math.ceil(250 / BIN_HZ), math.floor(1_200 / BIN_HZ),
                             BIN_HZ, fmt_hz, gap=1, window=192)
    disk = BurstDetector("Disk-seek", math.ceil(1_500 / BIN_HZ), math.floor(6_000 / BIN_HZ),
                         window=192, where="1.5-6 kHz band")
    return [ultrasonic, fan, disk], ultrasonic


def make_rf_detectors():
    """The RF detector. Returns (detectors, None)."""
    return [NarrowbandDetector("RF", 4, RF_BINS, RF_BIN_HZ, fmt_mhz, gap=1, window=96)], None


def monitor_acoustic(frames, thr=9.0):
    """Run the ultrasonic, fan and disk detectors over acoustic frames."""
    detectors, ultrasonic = make_acoustic_detectors()
    return _run(detectors, frames, thr, burst_hold_from=ultrasonic)


def monitor_rf(frames, thr=9.0):
    """Run the RF detector over RF frames."""
    detectors, _ = make_rf_detectors()
    return _run(detectors, frames, thr)


if __name__ == "__main__":
    from signals import ACOUSTIC_TECHNIQUES, RF_TECHNIQUES, make_acoustic, make_rf

    rng = np.random.default_rng(7)
    QUIET_S, ATTACK_S = 8, 25

    def first_alert(events, start):
        for e in events:
            if e.kind == "raised" and e.time >= start:
                return f"{e.detector} at {e.time - start:.1f} s ({e.text})"
        return "NOT DETECTED"

    def false_alarms(events, start):
        return sum(e.kind == "raised" and e.time < start for e in events)

    for tech in ACOUSTIC_TECHNIQUES:
        quiet, _ = make_acoustic(None, QUIET_S, rng)
        attack, _ = make_acoustic(tech, ATTACK_S, rng)
        events, _ = monitor_acoustic(np.vstack([quiet, attack]))
        print(f"{tech:10s} {first_alert(events, QUIET_S)}   false alarms before attack: {false_alarms(events, QUIET_S)}")
    for tech in RF_TECHNIQUES:
        quiet, _ = make_rf(None, QUIET_S, rng)
        attack, _ = make_rf(tech, ATTACK_S, rng, fc_mhz=50)
        events, _ = monitor_rf(np.vstack([quiet, attack]))
        print(f"{tech:10s} {first_alert(events, QUIET_S)}   false alarms before attack: {false_alarms(events, QUIET_S)}")

    quiet_a, _ = make_acoustic(None, 120, rng)
    quiet_r, _ = make_rf(None, 120, rng)
    print("quiet room, 120 s: acoustic false alarms =", sum(e.kind == "raised" for e in monitor_acoustic(quiet_a)[0]),
          " RF false alarms =", sum(e.kind == "raised" for e in monitor_rf(quiet_r)[0]))
