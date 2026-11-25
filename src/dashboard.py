"""Terminal dashboard for AetherHound.

A live spectrum waterfall in the terminal, with one status line per detector and
a flashing red banner when a covert channel is found. It uses only the Python
standard library plus numpy, so it runs in any terminal that understands ANSI
colours (Windows Terminal, VS Code terminal, Linux, macOS).

The room is simulated: a quiet start so the detectors can learn the baseline,
then a covert emitter switches on. Frames are fed to the detectors one by one,
at the speed a real microphone would deliver them.

Run from the repository root:
    python src/dashboard.py                       # ultrasonic channel
    python src/dashboard.py --channel fan
    python src/dashboard.py --channel video --log detections.log
    python src/dashboard.py --bench               # check the display speed (30+ fps goal)
    python src/dashboard.py --snapshot 15         # print one frame at t = 15 s, no animation

Press Ctrl+C to quit.
"""
import argparse
import os
import shutil
import sys
import time
from collections import deque
from datetime import datetime

import numpy as np

import detector as dt
import signals as sg

CHANNELS = sg.ACOUSTIC_TECHNIQUES + sg.RF_TECHNIQUES
QUIET_S = 8                 # quiet start, so the detectors can learn the room
FC_MHZ = 50                 # where the simulated RF emitters sit
MAX_EVENT_LINES = 4
BARS = " ▁▂▃▄▅▆▇█"

# 256-colour palette from cold (at the baseline) to hot (far above it).
PALETTE = [17, 18, 19, 25, 31, 37, 43, 79, 115, 151, 187, 223, 220, 214, 208, 202, 196]
BG = [f"\x1b[48;5;{c}m " for c in PALETTE]      # a coloured cell
FG = [f"\x1b[38;5;{c}m" for c in PALETTE]       # coloured text for the spectrum bar
RESET = "\x1b[0m"
BOLD = "\x1b[1m"
RED_BANNER = "\x1b[1;97;41m"
DIM = "\x1b[2m"
GREEN = "\x1b[32m"
RED = "\x1b[1;31m"
YELLOW = "\x1b[33m"

FLOOR_DB = 5.0              # lifts below this are shown as the coldest colour (hides room noise)
DB_PER_STEP = 1.2           # each palette step is this many dB above the floor


def build_recording(channel, seconds, rng):
    """Quiet room for QUIET_S seconds, then the covert emitter. Returns (frames, is_acoustic)."""
    acoustic = channel in sg.ACOUSTIC_TECHNIQUES
    if acoustic:
        quiet, _ = sg.make_acoustic(None, QUIET_S, rng)
        attack, _ = sg.make_acoustic(channel, seconds - QUIET_S, rng)
    else:
        quiet, _ = sg.make_rf(None, QUIET_S, rng)
        attack, _ = sg.make_rf(channel, seconds - QUIET_S, rng, fc_mhz=FC_MHZ)
    return np.vstack([quiet, attack]), acoustic


def downsample(values, width):
    """Shrink a row of bins to `width` columns, keeping the strongest bin of each group."""
    n = len(values)
    edges = np.linspace(0, n, width + 1).astype(int)
    edges[1:] = np.maximum(edges[1:], edges[:-1] + 1)
    edges = np.minimum(edges, n)
    return np.maximum.reduceat(values, edges[:-1].clip(max=n - 1))


def to_colour_index(lift_db):
    """Map dB above baseline to a palette index."""
    return np.clip(((lift_db - FLOOR_DB) / DB_PER_STEP).astype(int), 0, len(PALETTE) - 1)


class Dashboard:
    """Holds the detectors and the picture state. Call feed() per frame, render() per screen."""

    def __init__(self, channel, acoustic, log_path=None, thr=9.0):
        self.channel, self.acoustic, self.thr = channel, acoustic, thr
        if acoustic:
            self.detectors, self.hold_from = dt.make_acoustic_detectors()
            self.bins, self.unit_hz, self.title = sg.AUDIO_BINS, sg.BIN_HZ, "ACOUSTIC 0-24 kHz"
        else:
            self.detectors, self.hold_from = dt.make_rf_detectors()
            self.bins, self.unit_hz, self.title = sg.RF_BINS, sg.RF_BIN_HZ, "RF 0-200 MHz"
        self.frame_count = 0
        self.baseline = np.zeros(self.bins)
        self.cal = []
        self.latest = np.zeros(self.bins)          # lift of the newest frame, in dB
        self.history = deque(maxlen=200)           # recent lift rows for the waterfall
        self.events = deque(maxlen=MAX_EVENT_LINES)
        self.log_path = log_path
        self.alert_started = None

    @property
    def t(self):
        return self.frame_count * sg.FRAME_S

    @property
    def calibrating(self):
        return self.frame_count < dt.CAL_FRAMES

    def feed(self, db):
        """Give one spectrum frame to the display and to every detector."""
        self.frame_count += 1
        if self.frame_count <= dt.CAL_FRAMES:
            self.cal.append(db)
            if self.frame_count == dt.CAL_FRAMES:
                self.baseline = np.mean(self.cal, axis=0)
                self.cal = []
        self.latest = db - self.baseline
        self.history.append(self.latest)
        for e in dt.step_detectors(self.detectors, db, self.thr, self.t, self.hold_from):
            self.events.append(e)
            if e.kind == "raised" and self.alert_started is None:
                self.alert_started = self.t
            self._log(e)
        if not any(d.alert for d in self.detectors):
            self.alert_started = None

    def _log(self, e):
        if not self.log_path:
            return
        stamp = datetime.now().isoformat(timespec="seconds")
        line = f"{stamp}  t={e.time:6.1f}s  {e.detector:<10s} {e.kind.upper():<7s} {e.text}\n"
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(line)

    # ---- drawing ----------------------------------------------------------
    def axis_line(self, width):
        """Frequency labels under the spectrum."""
        top = self.bins * self.unit_hz
        marks = 5
        out = [" "] * width
        for i in range(marks):
            pos = int(i * (width - 1) / (marks - 1))
            hz = top * pos / (width - 1)
            label = f"{hz / 1000:.0f}k" if self.acoustic else f"{hz / 1e6:.0f}M"
            start = min(pos, width - len(label))
            for j, ch in enumerate(label):
                out[start + j] = ch
        return "".join(out)

    def spectrum_bar(self, width):
        cols = downsample(self.latest, width)
        idx = to_colour_index(cols)
        levels = np.clip(((cols - FLOOR_DB) / (DB_PER_STEP * len(PALETTE)) * (len(BARS) - 1)).astype(int),
                         0, len(BARS) - 1)
        return "".join(FG[i] + BARS[l] for i, l in zip(idx, levels)) + RESET

    def status_lines(self):
        lines = []
        for d in self.detectors:
            if self.calibrating:
                lines.append(f" {YELLOW}...{RESET} {d.name:<11s} learning the room")
            elif d.alert:
                text = ", ".join(c["short"] if isinstance(d, dt.BurstDetector) else
                                 f"{c['where']} {c['short']}" for c in d.clusters if c["key"])
                lines.append(f" {RED}ALERT{RESET} {d.name:<11s} {text}")
            else:
                steady = [c for c in d.clusters if not c["key"]]
                note = f"  ({len(steady)} steady line{'s' * (len(steady) != 1)} listed)" if steady else ""
                lines.append(f" {GREEN} ok {RESET} {d.name:<11s} quiet{note}")
        return lines

    def render(self, width, height, flash=False):
        """The whole screen as one string, ready to print after a cursor-home code."""
        width = max(40, width)
        height = max(16, height)
        alerting = any(d.alert for d in self.detectors) and not self.calibrating
        out = []
        head = f" AETHERHOUND  {self.title}  channel: {self.channel}  t = {self.t:5.1f} s"
        out.append(BOLD + head.ljust(width) + RESET)
        if alerting:
            secs = self.t - (self.alert_started or self.t)
            text = f" !!  COVERT CHANNEL DETECTED  !!   alert for {secs:4.1f} s"
            style = RED_BANNER if flash else "\x1b[1;91;40m"
            out.append(style + text.ljust(width) + RESET)
        elif self.calibrating:
            out.append(YELLOW + f" Calibrating baseline... {100 * self.frame_count // dt.CAL_FRAMES}%".ljust(width) + RESET)
        else:
            out.append(GREEN + " Monitoring. No covert channel found.".ljust(width) + RESET)
        out.append(self.spectrum_bar(width))
        out.append(DIM + self.axis_line(width) + RESET)
        status = self.status_lines()
        events = [f" {e.time:6.1f}s  {e.detector:<10s} {e.kind.upper():<7s} {e.text}" for e in self.events]
        events += [""] * (MAX_EVENT_LINES - len(events))
        fixed = len(out) + len(status) + 1 + len(events) + 2
        rows = max(3, height - fixed)
        out.append(DIM + " waterfall: newest at the bottom, colour = dB above the room baseline" + RESET)
        recent = list(self.history)[-rows:]
        blank = [BG[0] * width]
        for row in [None] * (rows - len(recent)) + recent:
            if row is None:
                out.append(blank[0] + RESET)
            else:
                idx = to_colour_index(downsample(row, width))
                out.append("".join(BG[i] for i in idx) + RESET)
        out += status
        out.append(DIM + " recent events" + RESET)
        out += events
        return "\n".join(out)


def enable_ansi():
    """Make sure the terminal understands colour codes (needed on older Windows consoles)."""
    if os.name == "nt":
        os.system("")
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass


def run_live(dash, frames, fps, speed):
    """Replay the recording in real time, drawing `fps` screens per second."""
    enable_ansi()
    out = sys.stdout
    out.write("\x1b[?1049h\x1b[?25l")                 # alternate screen, hide cursor
    start = time.perf_counter()
    fed = 0
    tick = 0
    try:
        while fed < len(frames):
            now = time.perf_counter()
            due = min(len(frames), int((now - start) * speed / sg.FRAME_S))
            while fed < due:
                dash.feed(frames[fed])
                fed += 1
            size = shutil.get_terminal_size((100, 30))
            out.write("\x1b[H" + dash.render(size.columns, size.lines, flash=tick % 10 < 5).replace("\n", "\x1b[K\n") + "\x1b[K")
            out.flush()
            tick += 1
            sleep = start + tick / fps - time.perf_counter()
            if sleep > 0:
                time.sleep(sleep)
    finally:
        out.write("\x1b[?25h\x1b[?1049l" + RESET)       # show cursor, leave alternate screen
        out.flush()


def bench(dash, frames, width=120, height=40):
    """Measure how fast the display can be drawn. Returns screens per second."""
    for db in frames[:dt.CAL_FRAMES + 400]:
        dash.feed(db)
    count = 300
    t0 = time.perf_counter()
    for i in range(count):
        dash.feed(frames[(dt.CAL_FRAMES + 400 + i) % len(frames)])
        dash.render(width, height, flash=i % 2 == 0)
    return count / (time.perf_counter() - t0)


def main():
    ap = argparse.ArgumentParser(description="AetherHound terminal dashboard")
    ap.add_argument("--channel", choices=CHANNELS, default="ultrasonic")
    ap.add_argument("--seconds", type=float, default=40, help="length of the simulated recording")
    ap.add_argument("--fps", type=float, default=30, help="screen updates per second")
    ap.add_argument("--speed", type=float, default=1.0, help="1 = real time, 2 = twice as fast")
    ap.add_argument("--log", help="write every alert to this file")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--bench", action="store_true", help="measure display speed and exit")
    ap.add_argument("--snapshot", type=float, metavar="SECONDS",
                    help="print one screen at this time and exit")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    frames, acoustic = build_recording(args.channel, max(args.seconds, QUIET_S + 5), rng)
    dash = Dashboard(args.channel, acoustic, args.log)

    if args.bench:
        fps = bench(dash, frames)
        print(f"Drew {fps:.0f} screens per second "
              f"({'meets' if fps >= 30 else 'misses'} the 30 fps goal)")
        return
    if args.snapshot is not None:
        enable_ansi()
        for db in frames[:int(args.snapshot / sg.FRAME_S)]:
            dash.feed(db)
        print(dash.render(100, 30, flash=True))
        return
    try:
        run_live(dash, frames, args.fps, args.speed)
    except KeyboardInterrupt:
        pass
    print(f"Stopped at t = {dash.t:.1f} s. Alerts raised: "
          f"{sum(e.kind == 'raised' for e in dash.events)} in the last events shown.")


if __name__ == "__main__":
    main()
