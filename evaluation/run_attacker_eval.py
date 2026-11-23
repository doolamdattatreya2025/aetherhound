"""Attacker-side evaluation for AetherHound.

Three questions:
  1. Speed: how fast can the attacker send before the receiver starts making errors?
  2. Power: how quiet can the emitter be before the secret no longer arrives?
  3. Race: how much of a 16-byte secret has already leaked when the detector raises its alert?

Run from the repository root:
    python evaluation/run_attacker_eval.py            # full run
    python evaluation/run_attacker_eval.py --quick    # fewer trials

Results go to evaluation/results/ as CSV files and PNG plots.
"""
import argparse
import csv
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import attacker as at   # noqa: E402
import detector as dt   # noqa: E402
import signals as sg    # noqa: E402

TECHNIQUES = ["ultrasonic", "fan", "disk", "carrier", "video"]
DETECTOR_FOR = {"ultrasonic": "Ultrasonic", "fan": "Fan-speed", "disk": "Disk-seek",
                "carrier": "RF", "video": "RF"}
DEFAULT_LEVEL = {"ultrasonic": -54, "fan": -42, "disk": -52, "carrier": -88, "video": -88}
FC_MHZ = 50                                   # where the RF emitters sit
SPEED_FACTORS = [0.25, 0.5, 0.75, 1.0, 1.5]   # symbol length relative to the default
LEVEL_OFFSETS = [-12, -8, -4, 0]              # dB relative to the default emitter level
SECRET = bytes(range(0x41, 0x41 + 16))        # a 16-byte "key"
QUIET_S = 8                                   # quiet start so the detector can calibrate


def attack_trial(args):
    technique, factor, level_offset, seed = args
    rng = np.random.default_rng(seed)
    sym_ms = at.DEFAULT_SYM_MS[technique] * factor
    r = at.run_attack(technique, SECRET, rng, sym_ms=sym_ms, fc_mhz=FC_MHZ,
                      level=DEFAULT_LEVEL[technique] + level_offset)
    return r["success"], r["ber"]


def race_trial(args):
    technique, seed = args
    rng = np.random.default_rng(seed)
    r = at.run_attack(technique, SECRET, rng, fc_mhz=FC_MHZ)
    if technique in sg.ACOUSTIC_TECHNIQUES:
        quiet, _ = sg.make_acoustic(None, QUIET_S, rng)
        events, _ = dt.monitor_acoustic(np.vstack([quiet, r["frames"]]))
    else:
        quiet, _ = sg.make_rf(None, QUIET_S, rng)
        events, _ = dt.monitor_rf(np.vstack([quiet, r["frames"]]))
    hits = [e for e in events if e.kind == "raised" and e.detector == DETECTOR_FOR[technique]
            and e.time >= QUIET_S]
    if not hits:
        return None, 1.0
    delay = hits[0].time - QUIET_S
    sym_s = at.symbol_seconds(technique)
    # The first 32 bits are header and length, so secret bits start after them.
    leaked = min(max(int(delay / sym_s) - 32, 0), 8 * len(SECRET))
    return delay, leaked / (8 * len(SECRET))


def write_csv(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=10)
    ap.add_argument("--quick", action="store_true", help="3 trials per condition")
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "results"))
    args = ap.parse_args()
    n = 3 if args.quick else args.trials
    os.makedirs(args.out, exist_ok=True)
    t0 = time.time()

    with ProcessPoolExecutor(args.workers) as pool:
        # 1. speed sweep
        speed_rows = []
        for tech in TECHNIQUES:
            for f in SPEED_FACTORS:
                jobs = [(tech, f, 0, 1000 * i + 7) for i in range(n)]
                res = list(pool.map(attack_trial, jobs))
                sym = at.DEFAULT_SYM_MS[tech] * f
                speed_rows.append({
                    "technique": tech, "symbol_ms": round(sym, 1), "raw_bps": round(1000 / sym, 2),
                    "trials": n,
                    "secret_received_rate": round(sum(s for s, _ in res) / n, 3),
                    "mean_ber": round(float(np.mean([b for _, b in res])), 4)})
        write_csv(os.path.join(args.out, "attacker_speed.csv"), speed_rows)
        print(f"speed sweep done ({time.time() - t0:.0f} s)")

        # 2. power sweep
        power_rows = []
        for tech in TECHNIQUES:
            for off in LEVEL_OFFSETS:
                jobs = [(tech, 1.0, off, 1000 * i + 11) for i in range(n)]
                res = list(pool.map(attack_trial, jobs))
                power_rows.append({
                    "technique": tech, "level_offset_db": off, "trials": n,
                    "secret_received_rate": round(sum(s for s, _ in res) / n, 3),
                    "mean_ber": round(float(np.mean([b for _, b in res])), 4)})
        write_csv(os.path.join(args.out, "attacker_power.csv"), power_rows)
        print(f"power sweep done ({time.time() - t0:.0f} s)")

        # 3. race against the detector
        race_rows = []
        for tech in TECHNIQUES:
            res = list(pool.map(race_trial, [(tech, 1000 * i + 13) for i in range(n)]))
            delays = [d for d, _ in res if d is not None]
            race_rows.append({
                "technique": tech, "trials": n,
                "detected_rate": round(len(delays) / n, 3),
                "mean_alert_delay_s": round(float(np.mean(delays)), 2) if delays else "",
                "secret_bits_leaked_before_alert_pct": round(100 * float(np.mean([f for _, f in res])), 1),
                "full_transmission_s": round(
                    (24 + 8 + 8 * len(SECRET) + 8) * at.symbol_seconds(tech), 1)})
        write_csv(os.path.join(args.out, "attacker_race.csv"), race_rows)
        print(f"race done ({time.time() - t0:.0f} s)")

    plot(speed_rows, power_rows, race_rows, args.out)

    print("\n=== Summary ===")
    print("Default speed, default power:")
    for r in speed_rows:
        if r["symbol_ms"] == at.DEFAULT_SYM_MS[r["technique"]]:
            print(f"  {r['technique']:11s} {r['raw_bps']:5.2f} bps raw  secret received {r['secret_received_rate'] * 100:3.0f}%")
    print("Fastest speed that still delivers the secret in at least 90% of trials:")
    for tech in TECHNIQUES:
        ok = [r for r in speed_rows if r["technique"] == tech and r["secret_received_rate"] >= 0.9]
        print(f"  {tech:11s} {max(r['raw_bps'] for r in ok):5.2f} bps" if ok else f"  {tech:11s} none")
    print("Secret bits already leaked when the detector alerts:")
    for r in race_rows:
        print(f"  {r['technique']:11s} {r['secret_bits_leaked_before_alert_pct']:5.1f}%  "
              f"(alert after {r['mean_alert_delay_s']} s, full send takes {r['full_transmission_s']} s)")
    print(f"\nResults written to {args.out}  (total {time.time() - t0:.0f} s)")


def plot(speed_rows, power_rows, race_rows, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"ultrasonic": "#1f77b4", "fan": "#d95f02", "disk": "#7570b3",
              "carrier": "#1b9e77", "video": "#a6761d"}
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for tech in TECHNIQUES:
        rows = sorted((r for r in speed_rows if r["technique"] == tech), key=lambda r: r["raw_bps"])
        ax[0].plot([r["raw_bps"] for r in rows], [100 * r["secret_received_rate"] for r in rows],
                   marker="o", label=tech, color=colors[tech])
        ax[1].plot([r["raw_bps"] for r in rows], [100 * r["mean_ber"] for r in rows],
                   marker="o", label=tech, color=colors[tech])
    ax[0].set_ylabel("secret received intact (%)")
    ax[1].set_ylabel("bit error rate (%)")
    for a in ax:
        a.set_xlabel("attacker speed (bits per second on the air)")
        a.grid(alpha=0.3)
    ax[0].legend(frameon=False)
    ax[0].set_title("Faster sending breaks the secret")
    ax[1].set_title("Errors rise with speed")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "attacker_speed.png"), dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for tech in TECHNIQUES:
        rows = sorted((r for r in power_rows if r["technique"] == tech), key=lambda r: r["level_offset_db"])
        ax[0].plot([r["level_offset_db"] for r in rows], [100 * r["secret_received_rate"] for r in rows],
                   marker="o", label=tech, color=colors[tech])
    ax[0].set_xlabel("emitter level vs default (dB)")
    ax[0].set_ylabel("secret received intact (%)")
    ax[0].set_title("A quieter emitter loses the secret")
    ax[0].grid(alpha=0.3)
    ax[0].legend(frameon=False)
    names = [r["technique"] for r in race_rows]
    x = np.arange(len(names))
    alert = [float(r["mean_alert_delay_s"] or 0) for r in race_rows]
    start = [32 * at.symbol_seconds(t) for t in names]
    total = [r["full_transmission_s"] for r in race_rows]
    ax[1].bar(x - 0.27, alert, 0.27, label="detector alert", color="#d62728")
    ax[1].bar(x, start, 0.27, label="secret starts", color="#999999")
    ax[1].bar(x + 0.27, total, 0.27, label="full send ends", color="#1f77b4")
    ax[1].set_xticks(x)
    ax[1].set_xticklabels(names)
    ax[1].set_ylabel("seconds after the attack starts")
    ax[1].set_title("The race: the alert comes before the secret")
    ax[1].legend(frameon=False)
    ax[1].grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "attacker_power_and_race.png"), dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    main()
