# Threat Model: Covert Channels from an Air-Gapped Computer

Project: AetherHound
Scope: acoustic and RF covert channels, defended by passive monitoring

## 1. Context

An air-gapped computer has no network connection. It is used where the data is highly
sensitive, for example key material, internal documents or industrial control. The air gap
removes network attack paths, but it does not remove physical side effects. Any running
computer makes sound and radio emissions, and malware on the machine can control some of
them to carry data out.

This document describes the threat the project defends against, the channels we simulate,
and what the detector should look for.

## 2. Assets

- Data stored or processed on the air-gapped machine, such as keys, passwords and documents.
- Confidence that the isolated environment is truly isolated.

## 3. Adversary

**Capabilities**
- Has already placed malware on the air-gapped machine (for example through removable media
  or a compromised supply chain). How the malware arrived is out of scope.
- The malware can read data on the machine and can control fans, disk activity, speakers
  and the display output.
- Has a receiver nearby, such as a compromised phone, a microphone or a radio receiver.

**Limits**
- No network path out of the machine.
- The receiver must be physically close enough to pick up the signal.
- The channel is slow, so the adversary leaks small, high-value data such as keys.

## 4. Channels in scope

| # | Channel | Medium | How data is carried | What a defender can observe |
|---|---|---|---|---|
| 1 | Ultrasonic FSK | Acoustic | Two tones near 19 kHz, one per bit value, played through a speaker | Two keyed narrow lines above 16 kHz that take turns |
| 2 | Fan speed | Acoustic | Fan blade tone switches between two speeds | Two keyed lines in the low hundreds of Hz, switching slowly |
| 3 | Disk seeks | Acoustic | Mechanical seek noise bursts, a burst is a 1 and silence is a 0 | Repeated bursts of energy in roughly 1.5 to 6 kHz |
| 4 | Keyed carrier | RF | One narrow radio line switched on and off | One keyed line that does not appear in the baseline |
| 5 | Video-cable leakage | RF | Display cable harmonics are switched by changing what is drawn | Several lines at regular spacing that switch together |

Related published research, to be read and cited in the final report:
- Hanspach and Goetz, work on covert acoustic networking between computers (ultrasonic).
- Guri et al., Fansmitter (fan speed), DiskFiltration (disk seeks) and AirHopper (display
  cable emissions received on FM).

Check each citation (authors, year, venue) against the original papers before the final
report.

## 5. Attack flow

1. Malware reads the target data.
2. Malware encodes it into bits and a framing format.
3. Malware drives the chosen channel (speaker, fan, disk or display) to send the bits.
4. A nearby receiver records the signal.
5. The adversary decodes the recording offline.

AetherHound monitors the room during step 3 and 4, before data is lost.

## 6. Detection strategy

The detector learns what the room normally sounds and radiates like, then looks for changes
that behave like data.

- **Baseline per frequency bin.** Hum, fans and broadcast carriers become part of normal.
- **Keying.** Data channels switch on and off repeatedly. Steady new tones are only listed.
- **Pairing.** Two keyed lines that take turns indicate FSK.
- **Linking.** Several keyed lines that switch together indicate a harmonic source such as a cable.
- **Burst watch.** Total energy in the disk-noise band is counted over time.
- **Impulse rejection.** Clicks that raise most bins at once are ignored.

## 7. Risk summary

Ratings are qualitative and will be revisited after evaluation.

| Channel | Likelihood | Impact | Notes |
|---|---|---|---|
| Ultrasonic FSK | Medium | High | Needs a speaker and a microphone, which many machines have |
| Fan speed | Low | High | Very slow, works without a speaker |
| Disk seeks | Low | High | Needs a mechanical disk |
| Keyed carrier | Low | High | Needs the machine to produce a controllable radio line |
| Video-cable leakage | Medium | High | Needs a display output, which most machines have |

## 8. Mitigations (context for the report)

- Acoustic: remove or disable speakers and microphones, filter ultrasonic output, add masking
  noise, increase the distance to any receiver.
- RF: shielded cables and enclosures, shielded rooms, keeping receivers away from the room.
- Monitoring: the approach in this project, plus policy controls on devices brought into the room.

## 9. Assumptions and limits of the simulation

- Signals are synthetic and simplified. Real rooms have more varied noise.
- The room has a fixed noise model with a steady hum, a slowly drifting low-frequency noise
  floor and a few fixed radio carriers.
- Thresholds are tuned for the simulator and need retuning on real recordings.
- A channel already running during calibration enters the baseline and can be missed.
- Only passive detection is simulated. No jamming or active countermeasures.

## 10. Next steps

1. Build the simulator with all five channels.
2. Implement the detectors in Python and test them.
3. Measure detection rate and false alarms across emitter level and noise.
4. Add attacker-side data rate and error rate.
5. Write the report and presentation.
6. 
