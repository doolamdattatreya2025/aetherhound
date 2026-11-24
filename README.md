# AetherHound: Air-Gapped RF & Acoustic Covert Channel Detector

Internship project at Infotact Solutions (Cyber Security, Security Admin Associate L1).
Author: Doolam Dattatreya

## Overview

An air-gapped computer has no network connection, but malware on it can still leak data
through side channels such as sound and radio emissions. AetherHound is a software-only
lab that simulates an air-gapped room, simulates covert emitters in it, and detects them.
No hardware is needed. Everything runs in a virtual environment.

## Goals

1. Simulate a room with a microphone and an RF receiver, including realistic background noise.
2. Simulate covert emitters that leak data through acoustic and RF channels.
3. Build detectors that find these channels with few false alarms.
4. Measure detector performance and document the results.

## Covert channels covered

| Channel | Medium | How data is carried |
|---|---|---|
| Ultrasonic FSK | Acoustic | Two tones near 19 kHz, above most adult hearing |
| Fan speed | Acoustic | Fan blade tone shifts between two speeds |
| Disk seeks | Acoustic | Short mechanical bursts in the 1.5 to 6 kHz band |
| Keyed carrier | RF | One narrow radio line switched on and off |
| Video-cable leakage | RF | Harmonic lines of a display cable that switch together |

## Detection approach

1. **Baseline.** Learn the normal level of every frequency bin during a short calibration.
2. **Threshold.** Flag bins that rise above their baseline by a set number of dB for consecutive frames.
3. **Keying.** A bin that switches on and off repeatedly is treated as data. Steady tones are listed but not alerted.
4. **Pairing and linking.** Keyed bins that take turns indicate FSK. Keyed bins that switch together indicate linked lines such as cable harmonics.
5. **Burst watch.** Total energy in the disk-noise band is tracked, and bursts are counted.
6. **Impulse rejection.** Broadband clicks that raise most bins at once are ignored.

## Repository structure

```
aetherhound/
  README.md
  requirements.txt
  simulator/      browser-based lab (index.html, no install needed)
  src/            signals.py (room and emitters), detector.py, attacker.py
  tests/          unit tests
  evaluation/     sweep scripts and results (CSV tables and PNG plots)
  docs/           threat model, architecture, report
```

## How to run

```
pip install -r requirements.txt
python src/signals.py                          # check the signal generators
python src/detector.py                         # demo: detect all five channels
python src/attacker.py                         # demo: send a secret and read it back
python -m unittest discover -s tests -v        # run all tests
python evaluation/run_evaluation.py --quick    # detector evaluation (fast)
python evaluation/run_attacker_eval.py --quick # attacker evaluation (fast)
```

To use the browser simulator, open `simulator/index.html` in a browser.

## Project plan

| Week | Focus |
|---|---|
| 1 | Threat model, repository setup, simulator, signal generators, detectors, tests, first evaluation |
| 2 | Attacker side (sender and receiver), attacker evaluation, terminal dashboard with live spectrum waterfall |
| 3 | Learned baseline (autoencoder trained on normal room sound), alerting logic, tuning |
| 4 | Detection log files, hardening and evasion tests, report, presentation, demo |

## Status

- [x] Repository created
- [x] Threat model
- [x] Simulator (browser)
- [x] Signal generators for five covert channels
- [x] Detectors
- [x] Evaluation of detection rate and false alarms
- [x] Attacker sender and receiver, with data rate and bit error rate
- [ ] Terminal dashboard with live spectrum waterfall
- [ ] Learned baseline (autoencoder)
- [ ] Detection log files
- [ ] Hardening and evasion tests
- [ ] Report and presentation

## Simulation note

The project brief describes an SDR and ultrasonic microphones. This version replaces the
hardware with simulated room signals so that it runs anywhere with no hardware. The detector
works on spectrum frames, so the same detector design applies to real recordings.

## Limitations

The signals are synthetic and simplified. Detection thresholds would need retuning
against real microphone and software-defined-radio captures before any real use.

## License

To be decided.
