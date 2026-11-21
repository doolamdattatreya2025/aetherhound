"""Attacker side of AetherHound: a sender and a receiver.

The attacker's malware (sender) turns a secret into bits, wraps them in a frame,
and plays them through one of the five covert channels. The attacker's receiver
only gets what a nearby microphone or radio would get: the spectrum frames.
It finds the start of the frame, reads the bits back and checks a CRC.

This lets us measure what the attacker achieves (data rate, bit error rate,
whether the secret arrives intact) and compare it with what the detector sees.

Frame layout, in bits:
    preamble  16   1010...10     lets the receiver find the symbol timing
    sync       8   11010011      marks where the data starts
    length     8   payload size in bytes
    payload    8 * length
    crc        8   CRC-8 over length and payload
"""
import math

import numpy as np

import signals as sg

PREAMBLE = [1, 0] * 8
SYNC = [1, 1, 0, 1, 0, 0, 1, 1]
HEADER = PREAMBLE + SYNC            # 24 known bits at the start of every frame

# Default symbol length per channel, in milliseconds. A longer symbol is slower
# but easier to read. These match the defaults of the emitters in signals.py.
DEFAULT_SYM_MS = {"ultrasonic": 120, "fan": 400, "disk": 300, "carrier": 160, "video": 160}


# ---- framing ---------------------------------------------------------------
def crc8(data):
    """CRC-8 (polynomial 0x07) of a bytes object."""
    crc = 0
    for byte in data:
        crc ^= byte
        for _ in range(8):
            crc = ((crc << 1) ^ 0x07) & 0xFF if crc & 0x80 else (crc << 1) & 0xFF
    return crc


def bytes_to_bits(data):
    return [(b >> i) & 1 for b in data for i in range(7, -1, -1)]


def bits_to_bytes(bits):
    return bytes(
        sum(bit << (7 - i) for i, bit in enumerate(bits[k:k + 8]))
        for k in range(0, len(bits) - len(bits) % 8, 8)
    )


def build_frame(payload):
    """Secret bytes in, list of bits out."""
    if not 0 < len(payload) < 256:
        raise ValueError("payload must be 1 to 255 bytes")
    body = bytes([len(payload)]) + payload
    return HEADER + bytes_to_bits(body) + bytes_to_bits(bytes([crc8(body)]))


def parse_frame(bits):
    """Bits after the sync word in, (payload, crc_ok). payload is None if too short."""
    if len(bits) < 8:
        return None, False
    length = bits_to_bytes(bits[:8])[0]
    need = 8 + 8 * length + 8
    if length == 0 or len(bits) < need:
        return None, False
    body = bits_to_bytes(bits[:8 + 8 * length])
    crc = bits_to_bytes(bits[8 + 8 * length:need])[0]
    return body[1:], crc == crc8(body)


# ---- sender ----------------------------------------------------------------
def symbol_seconds(technique, sym_ms=None):
    return (sym_ms or DEFAULT_SYM_MS[technique]) / 1000


def send(technique, payload, rng, noise=None, fc_mhz=71.5, sym_ms=None, level=None, tail_s=1.0):
    """Play the framed payload through one channel in a simulated room.

    noise is dBFS for acoustic channels and dBm for RF ones (room default if None).
    level is the emitter level in the same unit (emitter default if None).
    Returns (spectrum_frames, frame_bits, seconds).
    """
    bits = build_frame(payload)
    sym_s = symbol_seconds(technique, sym_ms)
    seconds = len(bits) * sym_s + tail_s
    args = {}
    if technique in sg.ACOUSTIC_TECHNIQUES:
        args["sym_ms"] = sym_s * 1000
        if noise is not None:
            args["noise_dbfs"] = noise
        if level is not None:
            args["level_dbfs"] = level
        frames, _ = sg.make_acoustic(technique, seconds, rng, bits=bits, **args)
    else:
        args["bit_ms"] = sym_s * 1000
        if noise is not None:
            args["noise_dbm"] = noise
        if level is not None:
            args["level_dbm"] = level
        frames, _ = sg.make_rf(technique, seconds, rng, fc_mhz=fc_mhz, bits=bits, **args)
    return frames, bits, seconds


# ---- receiver --------------------------------------------------------------
def _bin(freq_hz):
    return int(round(freq_hz / sg.BIN_HZ))


def soft_signal(technique, frames, fc_mhz=71.5):
    """One number per frame. High means the sender is probably sending a 1.

    The receiver knows which channel the sender uses and where it sits.
    """
    if technique == "ultrasonic":
        return frames[:, _bin(19_250)] - frames[:, _bin(18_750)]
    if technique == "fan":
        return frames[:, _bin(600)] - frames[:, _bin(440)]
    if technique == "disk":
        lo, hi = _bin(1_500), _bin(6_000)
        return 10 * np.log10((10 ** (frames[:, lo:hi] / 10)).sum(axis=1))
    k0 = int(round(fc_mhz * 1e6 / sg.RF_BIN_HZ))
    if technique == "carrier":
        return frames[:, k0]
    if technique == "video":
        return frames[:, [k0, k0 + 100, k0 + 200]].mean(axis=1)
    raise ValueError(technique)


def _symbol_values(technique, soft, start, sym_frames, count):
    """One value per symbol, starting at a (possibly fractional) frame position."""
    vals = np.empty(count)
    for i in range(count):
        a = start + i * sym_frames
        lo = int(round(a))
        # A seek is a short burst at the start of its slot, so for the disk we
        # only look at the first 4 frames, which keeps the next slot's burst out.
        span = min(4, sym_frames) if technique == "disk" else sym_frames
        hi = max(lo + 1, int(round(a + span)))
        if hi > len(soft):
            vals[i] = np.nan
            continue
        window = soft[max(lo, 0):hi]
        if len(window) == 0:
            vals[i] = np.nan
            continue
        if technique == "disk":
            vals[i] = window.max()          # take the peak of the burst
        else:                               # use the middle of the symbol, where it is settled
            m = len(window)
            mid = window[m // 5: max(m // 5 + 1, m - m // 5)]
            vals[i] = mid.mean()
    return vals


def receive(technique, frames, fc_mhz=71.5, sym_ms=None):
    """Find the frame in the recording and decode it.

    Returns a dict with: payload (bytes or None), crc_ok, bits (decoded frame bits
    from the start of the header), start_frame, and quality (gap between the
    levels of 1s and 0s in the header, in the unit of the soft signal).
    """
    soft = soft_signal(technique, frames, fc_mhz)
    sym_frames = symbol_seconds(technique, sym_ms) / sg.FRAME_S
    known = np.array(HEADER)
    # Try start positions across two symbols; keep the one where the 24 known
    # header bits split into the cleanest high and low groups.
    best = None
    for start in np.arange(-2, 2 * sym_frames, 0.25):
        v = _symbol_values(technique, soft, start, sym_frames, len(HEADER))
        if np.isnan(v).any():
            continue
        gap = v[known == 1].mean() - v[known == 0].mean()
        if best is None or gap > best[0]:
            best = (gap, start)
    if best is None:
        return {"payload": None, "crc_ok": False, "bits": [], "start_frame": None, "quality": 0.0}
    gap, start = best
    total = int((len(soft) - start) / sym_frames)
    v = _symbol_values(technique, soft, start, sym_frames, total)
    v = v[:np.argmax(np.isnan(v))] if np.isnan(v).any() else v
    hv = v[:len(HEADER)]
    thr = (hv[known == 1].mean() + hv[known == 0].mean()) / 2
    bits = [int(x > thr) for x in v]
    payload, ok = (None, False)
    if bits[:len(HEADER)] == HEADER:
        payload, ok = parse_frame(bits[len(HEADER):])
    return {"payload": payload, "crc_ok": ok, "bits": bits, "start_frame": start, "quality": gap}


# ---- measuring -------------------------------------------------------------
def bit_errors(sent_bits, got_bits):
    """Count wrong bits over the sent frame. Missing bits count as errors."""
    got = list(got_bits[:len(sent_bits)]) + [1 - b for b in sent_bits[len(got_bits):]]
    return sum(a != b for a, b in zip(sent_bits, got))


def run_attack(technique, payload, rng, **kwargs):
    """Send a secret, receive it, and report how it went."""
    fc = kwargs.get("fc_mhz", 71.5)
    sym_ms = kwargs.get("sym_ms")
    frames, bits, seconds = send(technique, payload, rng, **kwargs)
    rx = receive(technique, frames, fc, sym_ms)
    sym_s = symbol_seconds(technique, sym_ms)
    errors = bit_errors(bits, rx["bits"])
    airtime = len(bits) * sym_s
    return {
        "technique": technique,
        "frames": frames,
        "sent_bits": bits,
        "received": rx,
        "success": rx["crc_ok"] and rx["payload"] == payload,
        "bit_errors": errors,
        "ber": errors / len(bits),
        "raw_bps": 1 / sym_s,                            # bits on the air per second
        "goodput_bps": 8 * len(payload) / airtime,       # secret bits delivered per second
        "airtime_s": airtime,
    }


if __name__ == "__main__":
    rng = np.random.default_rng(7)
    secret = b"KEY-9F3A-77C1-B2D4"
    print(f"Secret: {secret!r} ({len(secret)} bytes)\n")
    print(f"{'channel':11s} {'raw bps':>8s} {'goodput':>8s} {'airtime':>8s} {'BER':>6s}  result")
    for tech in sg.ACOUSTIC_TECHNIQUES + sg.RF_TECHNIQUES:
        r = run_attack(tech, secret, rng)
        got = r["received"]["payload"]
        print(f"{tech:11s} {r['raw_bps']:8.2f} {r['goodput_bps']:8.2f} {r['airtime_s']:7.1f}s "
              f"{r['ber']:6.3f}  {'OK ' + repr(got) if r['success'] else 'FAILED'}")
