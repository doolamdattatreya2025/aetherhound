"""Signal generators for AetherHound.

Python version of the signal side of the browser simulator. It builds a
simulated room (background noise plus one covert emitter) and turns it into
spectrum frames that a detector can analyse.

Acoustic channels are generated as audio samples and then analysed with an FFT.
RF channels are generated directly as spectrum frames, because the detector only
ever sees spectrum.

Every emitter also returns the bits it sent. The detector never sees them; the
evaluation code uses them later to measure bit error rate.
"""
import numpy as np
from scipy.signal import lfilter

# ---- constants (same values as the browser simulator) ----------------------
FS = 48_000                  # audio sample rate, Hz
N = 2048                     # FFT size
HOP = 1024                   # samples between frames
FRAME_S = HOP / FS           # one frame is about 21.3 ms
BIN_HZ = FS / N              # width of one audio bin, about 23.4 Hz
AUDIO_BINS = N // 2          # bins kept per frame (0 to 24 kHz)
RF_BINS = 800                # RF spectrum bins
RF_BIN_HZ = 250_000          # width of one RF bin (0 to 200 MHz in total)

ACOUSTIC_TECHNIQUES = ("ultrasonic", "fan", "disk")
RF_TECHNIQUES = ("carrier", "video")


def db_to_amp(db):
    return 10 ** (db / 20)


def db_to_pow(db):
    return 10 ** (db / 10)


def random_bits(count, rng):
    """Random data for an emitter to send."""
    return rng.integers(0, 2, size=count)


def _bits_for(bits, count, rng):
    """Use the bits the attacker gave us (padded with zeros), or make random ones."""
    if bits is None:
        return random_bits(count, rng)
    bits = np.asarray(bits, dtype=int)
    return np.concatenate([bits, np.zeros(max(0, count - len(bits)), dtype=int)])


def _hold(bits, sym_samples, n):
    """Repeat each bit for one symbol and cut the result to n samples."""
    return np.repeat(bits, sym_samples)[:n]


# ---- acoustic room ---------------------------------------------------------
def room_noise(seconds, noise_dbfs=-42, rng=None):
    """Background sound of the room, as audio samples.

    White noise, a slow low-frequency drift, mains hum at 120 and 240 Hz,
    and a faint whine at 16.6 kHz (below the ultrasonic band we watch).
    """
    rng = rng or np.random.default_rng()
    n = int(seconds * FS)
    sigma = db_to_amp(noise_dbfs)
    white = sigma * rng.standard_normal(n)
    drift = lfilter([0.08], [1, -0.995], rng.standard_normal(n))
    t = np.arange(n) / FS
    hum = 0.012 * np.sin(2 * np.pi * 120 * t) + 0.005 * np.sin(2 * np.pi * 240 * t)
    whine = 0.0015 * np.sin(2 * np.pi * 16_600 * t)
    return white + 3 * sigma * drift + hum + whine


# ---- acoustic emitters -----------------------------------------------------
def ultrasonic_fsk(seconds, level_dbfs=-54, fc=19_000, sym_ms=120, rng=None, bits=None):
    """Two tones 500 Hz apart around fc. A 1 is the higher tone, a 0 the lower."""
    rng = rng or np.random.default_rng()
    n = int(seconds * FS)
    sym = int(sym_ms / 1000 * FS)
    bits = _bits_for(bits, n // sym + 1, rng)
    freq = fc + np.where(_hold(bits, sym, n) == 1, 250, -250)
    phase = 2 * np.pi * np.cumsum(freq) / FS
    return db_to_amp(level_dbfs) * np.sin(phase), bits


def fan_speed(seconds, level_dbfs=-42, fc=520, sym_ms=400, rng=None, bits=None):
    """A fan tone that moves between two speeds, 160 Hz apart.

    A real fan cannot jump between speeds, so the tone glides with a time
    constant of about 60 ms.
    """
    rng = rng or np.random.default_rng()
    n = int(seconds * FS)
    sym = int(sym_ms / 1000 * FS)
    bits = _bits_for(bits, n // sym + 1, rng)
    target = fc + np.where(_hold(bits, sym, n) == 1, 80, -80)
    alpha = 1 / (0.06 * FS)
    freq, _ = lfilter([alpha], [1, -(1 - alpha)], target, zi=[(1 - alpha) * fc])
    phase = 2 * np.pi * np.cumsum(freq) / FS
    return db_to_amp(level_dbfs) * np.sin(phase), bits


def disk_seeks(seconds, level_dbfs=-52, sym_ms=300, rng=None, bits=None):
    """Seek noise from a hard disk. A burst is a 1, silence is a 0.

    Each burst is 40 ms of noise with a smooth start and end, passed through a
    resonance near 3 kHz, which is roughly where a mechanical disk is loud.
    """
    rng = rng or np.random.default_rng()
    n = int(seconds * FS)
    sym = int(sym_ms / 1000 * FS)
    burst = int(0.040 * FS)
    bits = _bits_for(bits, n // sym + 1, rng)
    excite = np.zeros(n)
    envelope = np.sin(np.pi * np.arange(burst) / burst)
    for slot, bit in enumerate(bits):
        start = slot * sym
        if bit and start + burst <= n:
            excite[start:start + burst] = 1.4 * envelope * rng.standard_normal(burst)
    r, theta = 0.9, 2 * np.pi * 3_000 / FS
    resonator = lfilter([1], [1, -2 * r * np.cos(theta), r * r], excite)
    return db_to_amp(level_dbfs) * resonator, bits


# ---- audio to spectrum frames ----------------------------------------------
_WINDOW = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(N) / N)


def acoustic_frames(samples):
    """Cut audio into frames and return the level of every bin in dB.

    Result shape is (frames, 1024). A pure tone of amplitude A reads 20*log10(A).
    """
    count = (len(samples) - N) // HOP + 1
    idx = np.arange(N)[None, :] + HOP * np.arange(count)[:, None]
    spectrum = np.fft.rfft(samples[idx] * _WINDOW, axis=1)[:, :AUDIO_BINS]
    return 20 * np.log10(np.abs(spectrum) / (N / 4) + 1e-9)


def make_acoustic(technique, seconds, rng=None, noise_dbfs=-42, **emitter_args):
    """Room plus one emitter, returned as dB frames.

    technique is "ultrasonic", "fan", "disk" or None for a quiet room.
    Returns (frames_db, bits). bits is None when no emitter is on.
    """
    rng = rng or np.random.default_rng()
    samples = room_noise(seconds, noise_dbfs, rng)
    bits = None
    if technique is not None:
        emitter = {"ultrasonic": ultrasonic_fsk, "fan": fan_speed, "disk": disk_seeks}[technique]
        extra, bits = emitter(seconds, rng=rng, **emitter_args)
        samples = samples + extra
    return acoustic_frames(samples), bits


# ---- RF ---------------------------------------------------------------------
# Fixed radio carriers always present in the room: (MHz, dBm).
BASE_CARRIERS = [(88.5, -72), (91.1, -79), (94.9, -70), (98.3, -76), (101.7, -74),
                 (104.5, -82), (27, -84), (144.5, -86), (33, -91), (66, -91),
                 (99, -91), (132, -91), (165, -91)]


def _add_line(power, k, p):
    """Put power p on RF bin k, with a little leaking into the neighbours."""
    power[:, k] += p
    if k > 0:
        power[:, k - 1] += 0.2 * p
    if k < RF_BINS - 1:
        power[:, k + 1] += 0.2 * p


def rf_background(frames, noise_dbm=-100, rng=None):
    """Noise floor plus the fixed carriers, as linear power (frames, 800)."""
    rng = rng or np.random.default_rng()
    power = db_to_pow(noise_dbm) * rng.exponential(size=(frames, RF_BINS))
    for mhz, dbm in BASE_CARRIERS:
        k = round(mhz * 1e6 / RF_BIN_HZ)
        wobble = 1 + (rng.random(frames) - 0.5) * 0.1
        line = np.zeros_like(power)
        line[:, k] = db_to_pow(dbm) * wobble
        power += line + 0.2 * np.roll(line, 1, axis=1) + 0.2 * np.roll(line, -1, axis=1)
    return power


def add_rf_emitter(power, technique, fc_mhz, level_dbm=-88, bit_ms=160, rng=None, bits=None):
    """Switch an RF emitter on and off by data bit. Returns the bits sent.

    "carrier": one line at fc_mhz.
    "video":   three lines 25 MHz apart that switch together, 1 dB weaker each,
               like the harmonics of a display cable.
    """
    rng = rng or np.random.default_rng()
    frames = power.shape[0]
    slot = (np.arange(frames) * FRAME_S / (bit_ms / 1000)).astype(int)
    bits = _bits_for(bits, slot.max() + 1, rng)
    on = bits[slot] == 1
    lines = [(0, 0)] if technique == "carrier" else [(0, 0), (100, -1), (200, -2)]
    k0 = round(fc_mhz * 1e6 / RF_BIN_HZ)
    for offset, db in lines:
        k = k0 + offset
        if 1 <= k < RF_BINS - 1:
            p = np.zeros_like(power)
            p[on, k] = db_to_pow(level_dbm + db) * (1 + (rng.random(on.sum()) - 0.5) * 0.1)
            power += p + 0.25 * np.roll(p, 1, axis=1) + 0.25 * np.roll(p, -1, axis=1)
    return bits


def make_rf(technique, seconds, rng=None, noise_dbm=-100, fc_mhz=71.5, **emitter_args):
    """RF room plus one emitter, returned as dB frames and the bits sent."""
    rng = rng or np.random.default_rng()
    frames = int(seconds / FRAME_S)
    power = rf_background(frames, noise_dbm, rng)
    bits = None
    if technique is not None:
        bits = add_rf_emitter(power, technique, fc_mhz, rng=rng, **emitter_args)
    return 10 * np.log10(power), bits


if __name__ == "__main__":
    # Quick look: strongest bins that stand out from a quiet room, per technique.
    rng = np.random.default_rng(1)
    quiet, _ = make_acoustic(None, 20, rng)
    base = quiet.mean(axis=0)
    for tech in ACOUSTIC_TECHNIQUES:
        frames, bits = make_acoustic(tech, 20, rng)
        lift = frames.mean(axis=0) - base
        top = np.argsort(lift)[-3:][::-1]
        print(f"{tech:10s} bits sent: {len(bits):3d}  most raised bins:",
              ", ".join(f"{b * BIN_HZ:7.0f} Hz (+{lift[b]:.0f} dB)" for b in top))
    quiet_rf, _ = make_rf(None, 20, rng)
    base_rf = quiet_rf.mean(axis=0)
    for tech in RF_TECHNIQUES:
        frames, bits = make_rf(tech, 20, rng, fc_mhz=50)
        lift = frames.mean(axis=0) - base_rf
        top = np.argsort(lift)[-3:][::-1]
        print(f"{tech:10s} bits sent: {len(bits):3d}  most raised bins:",
              ", ".join(f"{b * RF_BIN_HZ / 1e6:6.2f} MHz (+{lift[b]:.0f} dB)" for b in top))
