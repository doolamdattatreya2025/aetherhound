"""Evaluation of the AetherHound detectors.

Runs many simulated attacks and measures how well the detectors do:

  1. Emitter level:  detection rate and time to detect as the emitter gets weaker.
  2. Room noise:     detection rate as the room gets noisier.
  3. Sensitivity:    the trade-off between catching weak emitters and false alarms.

Each trial is a quiet room (so the detector can learn its baseline) followed by
one covert emitter. A trial counts as detected when the detector for that
channel raises an alert after the emitter starts, or is already in alert when it
starts. It counts as a false alarm if any detector raises an alert before the
emitter starts, or if the wrong detector raises one.

Run from the project folder:
    python evaluation/run_evaluation.py              (20 trials per condition, a few minutes)
    python evaluation/run_evaluation.py --quick      (6 trials, about a minute)
Results are written to evaluation/results/ as CSV files and PNG plots.
"""
import argparse
import csv
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import detector as dt                      # noqa: E402
import signals as sg                       # noqa: E402

TECHNIQUES = ["ultrasonic", "fan", "disk", "carrier", "video"]
DETECTOR_FOR = {"ultrasonic": "Ultrasonic", "fan": "Fan-speed", "disk": "Disk-seek",
                "carrier": "RF", "video": "RF"}
IS_ACOUSTIC = {t: t in sg.ACOUSTIC_TECHNIQUES for t in TECHNIQUES}
DEFAULT_LEVEL = {"ultrasonic": -54, "fan": -42, "disk": -52, "carrier": -88, "video": -88}
DEFAULT_NOISE = {True: -42, False: -100}   # keyed by IS_ACOUSTIC
QUIET_S, ATTACK_S = 6, 20
LEVEL_OFFSETS = [-16, -12, -8, -4, 0, 4]   # dB relative to the default emitter level
NOISE_OFFSETS = [0, 3, 6, 9, 12]           # dB louder than the default room
THRESHOLDS = [5, 7, 8, 9, 10, 11, 13, 15]        # detector sensitivity, dB above baseline
WEAK_OFFSET = -8                           # emitter level used in the sensitivity sweep
QUIET_MINUTES = 30                         # quiet-room time per threshold (6 when --quick)


def run_trial(args):
    """One simulated attack. Returns (technique, delay or None, false alarm count)."""
    technique, level, noise, thr, seed = args
    rng = np.random.default_rng(seed)
    if IS_ACOUSTIC[technique]:
        quiet, _ = sg.make_acoustic(None, QUIET_S, rng, noise_dbfs=noise)
        attack, _ = sg.make_acoustic(technique, ATTACK_S, rng, noise_dbfs=noise, level_dbfs=level)
        events, _ = dt.monitor_acoustic(np.vstack([quiet, attack]), thr)
    else:
        quiet, _ = sg.make_rf(None, QUIET_S, rng, noise_dbm=noise)
        attack, _ = sg.make_rf(technique, ATTACK_S, rng, noise_dbm=noise, fc_mhz=50, level_dbm=level)
        events, _ = dt.monitor_rf(np.vstack([quiet, attack]), thr)
    want = DETECTOR_FOR[technique]
    raised = [e for e in events if e.kind == "raised"]
    hits = [e for e in raised if e.detector == want and e.time >= QUIET_S]
    # An alert already on when the emitter starts (raised earlier by noise) never
    # raises again, so it counts as detected with no delay. The earlier alarm is
    # still counted as a false alarm below.
    active = False
    for e in events:
        if e.detector == want and e.time < QUIET_S:
            active = e.kind == "raised"
    delay = hits[0].time - QUIET_S if hits else (0.0 if active else None)
    false = sum(1 for e in raised if e.time < QUIET_S or e.detector != want)
    return technique, delay, false


def run_quiet(args):
    """One quiet-room run. Returns (acoustic alerts, rf alerts)."""
    thr, seed, seconds = args
    rng = np.random.default_rng(seed)
    acoustic, _ = sg.make_acoustic(None, seconds, rng)
    rf, _ = sg.make_rf(None, seconds, rng)
    n_a = sum(e.kind == "raised" for e in dt.monitor_acoustic(acoustic, thr)[0])
    n_r = sum(e.kind == "raised" for e in dt.monitor_rf(rf, thr)[0])
    return n_a, n_r


def summarise(results):
    """Reduce trial results to detection rate, mean delay and false alarm trials."""
    n = len(results)
    delays = [d for _, d, _ in results if d is not None]
    return {"trials": n,
            "detection_rate": round(len(delays) / n, 3),
            "mean_delay_s": round(float(np.mean(delays)), 2) if delays else "",
            "trials_with_false_alarm": sum(1 for _, _, f in results if f > 0)}


def write_csv(path, rows):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def sweep(pool, jobs_by_condition):
    """Run every condition's trials in parallel and summarise each condition."""
    flat = [(cond, job) for cond, jobs in jobs_by_condition.items() for job in jobs]
    out = list(pool.map(run_trial, [job for _, job in flat], chunksize=4))
    grouped = {}
    for (cond, _), res in zip(flat, out):
        grouped.setdefault(cond, []).append(res)
    return {cond: summarise(res) for cond, res in grouped.items()}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--trials", type=int, default=20, help="trials per condition")
    ap.add_argument("--quick", action="store_true", help="6 trials per condition")
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "results"))
    args = ap.parse_args()
    trials = 6 if args.quick else args.trials
    os.makedirs(args.out, exist_ok=True)
    started = time.time()
    seed = 1000

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        # 1. emitter level
        jobs = {}
        for t in TECHNIQUES:
            for off in LEVEL_OFFSETS:
                jobs[(t, off)] = [(t, DEFAULT_LEVEL[t] + off, DEFAULT_NOISE[IS_ACOUSTIC[t]], 9, seed + i)
                                  for i in range(trials)]
        res = sweep(pool, jobs)
        level_rows = [{"technique": t, "level_offset_db": off, "level": DEFAULT_LEVEL[t] + off, **res[(t, off)]}
                      for t in TECHNIQUES for off in LEVEL_OFFSETS]
        write_csv(os.path.join(args.out, "level_sweep.csv"), level_rows)
        print(f"level sweep done ({time.time() - started:.0f} s)")

        # 2. room noise
        jobs = {}
        for t in TECHNIQUES:
            base = DEFAULT_NOISE[IS_ACOUSTIC[t]]
            for off in NOISE_OFFSETS:
                jobs[(t, off)] = [(t, DEFAULT_LEVEL[t], base + off, 9, seed + 500 + i) for i in range(trials)]
        res = sweep(pool, jobs)
        noise_rows = [{"technique": t, "noise_offset_db": off, "noise": DEFAULT_NOISE[IS_ACOUSTIC[t]] + off,
                       **res[(t, off)]} for t in TECHNIQUES for off in NOISE_OFFSETS]
        write_csv(os.path.join(args.out, "noise_sweep.csv"), noise_rows)
        print(f"noise sweep done ({time.time() - started:.0f} s)")

        # 3. sensitivity: weak emitter detection and false alarms in a quiet room
        jobs = {}
        for t in TECHNIQUES:
            for thr in THRESHOLDS:
                jobs[(t, thr)] = [(t, DEFAULT_LEVEL[t] + WEAK_OFFSET, DEFAULT_NOISE[IS_ACOUSTIC[t]], thr,
                                   seed + 900 + i) for i in range(trials)]
        res = sweep(pool, jobs)
        runs = 6 if args.quick else QUIET_MINUTES
        sens_rows = []
        for thr in THRESHOLDS:
            quiet = list(pool.map(run_quiet, [(thr, seed + 7000 + i, 60) for i in range(runs)]))
            hours = runs * 60 / 3600
            fa_ac = sum(a for a, _ in quiet) / hours
            fa_rf = sum(r for _, r in quiet) / hours
            for t in TECHNIQUES:
                sens_rows.append({"technique": t, "threshold_db": thr, "weak_level": DEFAULT_LEVEL[t] + WEAK_OFFSET,
                                  **res[(t, thr)],
                                  "false_alarms_per_hour": round(fa_ac if IS_ACOUSTIC[t] else fa_rf, 1)})
        write_csv(os.path.join(args.out, "sensitivity_sweep.csv"), sens_rows)
        print(f"sensitivity sweep done ({time.time() - started:.0f} s)")

    make_plots(args.out, level_rows, noise_rows, sens_rows)
    print_summary(level_rows, noise_rows, sens_rows, trials)
    print(f"\nResults written to {args.out}  (total {time.time() - started:.0f} s)")


def print_summary(level_rows, noise_rows, sens_rows, trials):
    print(f"\n=== Summary ({trials} trials per condition) ===")
    print("\nDetection rate at the default emitter level and noise:")
    for r in level_rows:
        if r["level_offset_db"] == 0:
            print(f"  {r['technique']:10s} {r['detection_rate'] * 100:5.0f}%   mean time to detect {r['mean_delay_s']} s")
    print("\nWeakest emitter detected in at least 90% of trials (dB below default):")
    for t in TECHNIQUES:
        ok = [r["level_offset_db"] for r in level_rows if r["technique"] == t and r["detection_rate"] >= 0.9]
        print(f"  {t:10s} {min(ok) if ok else 'none in tested range'}")
    print("\nSensitivity trade-off (all five techniques averaged, weak emitter).")
    print("False alarms come from minutes of quiet room, so 0.0 means none seen, not none ever.")
    for thr in THRESHOLDS:
        rows = [r for r in sens_rows if r["threshold_db"] == thr]
        det = np.mean([r["detection_rate"] for r in rows])
        fa = np.mean([r["false_alarms_per_hour"] for r in rows])
        print(f"  +{thr:2d} dB  detection {det * 100:5.1f}%   false alarms per hour {fa:5.1f}")


def make_plots(out, level_rows, noise_rows, sens_rows):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not installed, so the plots were skipped. Run: pip install matplotlib")
        return

    def line_plot(rows, xkey, xlabel, title, fname):
        fig, ax = plt.subplots(figsize=(7, 4.2))
        for t in TECHNIQUES:
            pts = [(r[xkey], r["detection_rate"] * 100) for r in rows if r["technique"] == t]
            ax.plot(*zip(*pts), marker="o", label=t)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Detection rate (%)")
        ax.set_ylim(-3, 103)
        ax.set_title(title)
        ax.grid(alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(out, fname), dpi=150)
        plt.close(fig)

    line_plot(level_rows, "level_offset_db", "Emitter level relative to default (dB)",
              "Detection rate vs emitter level", "detection_vs_level.png")
    line_plot(noise_rows, "noise_offset_db", "Room noise above default (dB)",
              "Detection rate vs room noise", "detection_vs_noise.png")

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4.2))
    for t in TECHNIQUES:
        pts = [(r["threshold_db"], r["detection_rate"] * 100) for r in sens_rows if r["technique"] == t]
        a1.plot(*zip(*pts), marker="o", label=t)
    a1.set_xlabel("Sensitivity threshold (dB above baseline)")
    a1.set_ylabel(f"Detection rate of weak emitter ({WEAK_OFFSET} dB) (%)")
    a1.set_ylim(-3, 103)
    a1.grid(alpha=0.3)
    a1.legend()
    for name, flag in (("acoustic", True), ("RF", False)):
        pts = [(r["threshold_db"], r["false_alarms_per_hour"]) for r in sens_rows
               if r["technique"] == ("ultrasonic" if flag else "carrier")]
        a2.plot(*zip(*pts), marker="o", label=name)
    a2.set_xlabel("Sensitivity threshold (dB above baseline)")
    a2.set_ylabel("False alarms per hour of quiet room")
    a2.grid(alpha=0.3)
    a2.legend()
    fig.suptitle("Sensitivity trade-off")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "sensitivity_tradeoff.png"), dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
