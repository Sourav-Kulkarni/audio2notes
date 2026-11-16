"""Minimal pure-numpy SoundFont 2 (.sf2) sample player.

Plays real recorded instrument samples from an .sf2 bank: zone selection by key/velocity,
root key + tuning, sample loops, volume envelope (DAHDSR), stereo pan, attenuation.
(No filters/modulators/LFOs - enough for natural sounding melody playback.)
"""
import struct
from functools import lru_cache
from pathlib import Path

import numpy as np

# generator operator ids (SF2 spec section 8.1.2)
G_START_OFS, G_END_OFS, G_LSTART_OFS, G_LEND_OFS, G_START_C = 0, 1, 2, 3, 4
G_END_C, G_PAN = 12, 17
G_DELAY, G_ATTACK, G_HOLD, G_DECAY, G_SUSTAIN, G_RELEASE = 33, 34, 35, 36, 37, 38
G_INSTRUMENT, G_KEYRANGE, G_VELRANGE, G_LSTART_C = 41, 43, 44, 45
G_ATTEN, G_LEND_C, G_COARSE, G_FINE, G_SAMPLE, G_MODES, G_SCALE, G_ROOT = 48, 50, 51, 52, 53, 54, 56, 58
RANGE_GENS = (G_KEYRANGE, G_VELRANGE)
# generators that are NOT additive at preset level (spec 8.5): sample/address/mode/root
INST_ONLY = {G_START_OFS, G_END_OFS, G_LSTART_OFS, G_LEND_OFS, G_START_C, G_END_C,
             G_LSTART_C, G_LEND_C, G_SAMPLE, G_MODES, G_ROOT, 57}

DEFAULTS = {G_DELAY: -12000, G_ATTACK: -12000, G_HOLD: -12000, G_DECAY: -12000,
            G_SUSTAIN: 0, G_RELEASE: -12000, G_ATTEN: 0, G_PAN: 0, G_COARSE: 0, G_FINE: 0,
            G_SCALE: 100, G_ROOT: -1, G_MODES: 0}


def _chunks(buf, off, end):
    while off < end:
        cid, size = struct.unpack_from("<4sI", buf, off)
        yield cid, off + 8, size
        off += 8 + size + (size & 1)


class SoundFont:
    def __init__(self, path):
        data = Path(path).read_bytes()
        assert data[:4] == b"RIFF" and data[8:12] == b"sfbk", "not an sf2 file"
        self.smpl = None
        pd = {}
        for cid, off, size in _chunks(data, 12, len(data)):
            if cid != b"LIST":
                continue
            kind = data[off:off + 4]
            for sid, so, ss in _chunks(data, off + 4, off + size):
                if kind == b"sdta" and sid == b"smpl":
                    self.smpl = np.frombuffer(data, dtype="<i2", count=ss // 2, offset=so).astype(np.float32) / 32768
                elif kind == b"pdta":
                    pd[sid.decode()] = data[so:so + ss]
        self.presets = self._parse(pd)

    @staticmethod
    def _gens(raw, a, b):
        out = {}
        for i in range(a, b):
            op, lo, hi = struct.unpack_from("<HBB", raw, i * 4)
            out[op] = (lo, hi) if op in RANGE_GENS else struct.unpack_from("<h", raw, i * 4 + 2)[0]
        return out

    def _zones(self, bags, gens, a, b, terminal):
        """Returns (global_gens, [zone_gens...]) for bag index range [a, b)."""
        zones = []
        for z in range(a, b):
            g0 = struct.unpack_from("<H", bags, z * 4)[0]
            g1 = struct.unpack_from("<H", bags, (z + 1) * 4)[0]
            zones.append(self._gens(gens, g0, g1))
        glob = {}
        if zones and terminal not in zones[0]:
            glob = zones.pop(0)
        return glob, [z for z in zones if terminal in z]

    def _parse(self, pd):
        shdr = pd["shdr"]
        self.samples = []
        for i in range(len(shdr) // 46 - 1):
            name, s, e, ls, le, sr, pitch, corr, link, typ = struct.unpack_from("<20sIIIIIBbHH", shdr, i * 46)
            self.samples.append(dict(name=name.split(b"\0")[0].decode("latin1"), start=s, end=e,
                                     ls=ls, le=le, sr=sr, pitch=pitch, corr=corr, type=typ))
        inst = pd["inst"]
        self.instruments = []
        n_inst = len(inst) // 22 - 1
        for i in range(n_inst):
            name, b0 = struct.unpack_from("<20sH", inst, i * 22)
            b1 = struct.unpack_from("<20sH", inst, (i + 1) * 22)[1]
            glob, zones = self._zones(pd["ibag"], pd["igen"], b0, b1, G_SAMPLE)
            self.instruments.append((glob, zones))
        phdr = pd["phdr"]
        presets = {}
        for i in range(len(phdr) // 38 - 1):
            name, prog, bank, b0 = struct.unpack_from("<20sHHH", phdr, i * 38)
            b1 = struct.unpack_from("<20sHHH", phdr, (i + 1) * 38)[3]
            glob, zones = self._zones(pd["pbag"], pd["pgen"], b0, b1, G_INSTRUMENT)
            presets[(bank, prog)] = dict(name=name.split(b"\0")[0].decode("latin1"), glob=glob, zones=zones)
        return presets

    def list_presets(self):
        return {k: v["name"] for k, v in sorted(self.presets.items())}

    # ------------------------------------------------------------- note rendering
    def _regions(self, bank, prog, key, vel):
        p = self.presets.get((bank, prog)) or self.presets.get((0, prog)) or next(iter(self.presets.values()))
        inr = lambda g, k, v: (g.get(G_KEYRANGE, (0, 127))[0] <= k <= g.get(G_KEYRANGE, (0, 127))[1] and
                               g.get(G_VELRANGE, (0, 127))[0] <= v <= g.get(G_VELRANGE, (0, 127))[1])
        for pz in p["zones"]:
            pg = {**p["glob"], **pz}
            if not inr(pg, key, vel):
                continue
            iglob, izones = self.instruments[pg[G_INSTRUMENT]]
            for iz in izones:
                ig = {**DEFAULTS, **iglob, **iz}
                if not inr(ig, key, vel):
                    continue
                g = dict(ig)
                for op, val in pg.items():  # preset-level generators are offsets
                    if op in INST_ONLY or op in RANGE_GENS or op == G_INSTRUMENT:
                        continue
                    g[op] = g.get(op, 0) + val
                yield g

    def render_note(self, key, vel, dur, sr=44100, bank=0, prog=0, tune_cents=0.0):
        out = None
        for g in self._regions(bank, prog, key, vel):
            sm = self.samples[g[G_SAMPLE]]
            s = sm["start"] + g.get(G_START_OFS, 0) + 32768 * g.get(G_START_C, 0)
            e = sm["end"] + g.get(G_END_OFS, 0) + 32768 * g.get(G_END_C, 0)
            ls = sm["ls"] + g.get(G_LSTART_OFS, 0) + 32768 * g.get(G_LSTART_C, 0) - s
            le = sm["le"] + g.get(G_LEND_OFS, 0) + 32768 * g.get(G_LEND_C, 0) - s
            wav = self.smpl[s:e]
            root = g[G_ROOT] if g[G_ROOT] >= 0 else sm["pitch"]
            cents = (key - root) * g[G_SCALE] + g[G_COARSE] * 100 + g[G_FINE] + sm["corr"] + tune_cents
            ratio = 2 ** (cents / 1200) * sm["sr"] / sr

            tc = lambda op: 2 ** (g[op] / 1200)  # timecents -> seconds
            rel = min(tc(G_RELEASE), 1.5)
            n = int((dur + rel) * sr)
            pos = np.arange(n) * ratio
            loop = (g[G_MODES] & 1) and le - ls > 32 and 0 <= ls < le <= len(wav)
            if loop:
                over = pos >= le
                pos[over] = ls + np.mod(pos[over] - ls, le - ls)
            valid = pos < len(wav) - 1
            i0 = np.minimum(pos.astype(np.int64), len(wav) - 2)
            fr = pos - i0
            y = (wav[i0] * (1 - fr) + wav[i0 + 1] * fr) * valid

            # volume envelope: delay, attack (linear), hold, decay (in dB), sustain, release
            t = np.arange(n) / sr
            d, a, h, dc = tc(G_DELAY), tc(G_ATTACK), tc(G_HOLD), tc(G_DECAY)
            sus_db = -min(max(g[G_SUSTAIN], 0), 1440) / 10
            # SF2 decay = time to fall 100 dB, linear in dB, stopping at the sustain level
            env_db = np.where(t < d + a + h, 0.0, np.maximum(sus_db, -(t - d - a - h) / max(dc, 1e-3) * 100))
            env = 10 ** (env_db / 20)
            env = np.where(t < d, 0, np.where(t < d + a, (t - d) / max(a, 1e-4), 1)) * env
            off = t >= dur
            if off.any():
                lvl = env[np.argmax(off)]
                env[off] = lvl * 10 ** (-(t[off] - dur) / max(rel, 1e-3) * 100 / 20)  # -100 dB over release
            att_db = -0.04 * g[G_ATTEN] - 40 * np.log10(127 / max(vel, 1)) * 0.5  # EMU-style 0.4 factor + velocity curve
            y = y * env * 10 ** (att_db / 20)
            pan = np.clip(g[G_PAN] / 1000 + 0.5, 0, 1)
            st = np.stack([y * np.cos(pan * np.pi / 2), y * np.sin(pan * np.pi / 2)])
            if out is None:
                out = st
            else:
                m = max(out.shape[1], n)
                out = np.pad(out, ((0, 0), (0, m - out.shape[1])))
                out[:, :n] += st
        return out if out is not None else np.zeros((2, int(dur * sr)))


@lru_cache(maxsize=4)
def load_soundfont(path):
    return SoundFont(path)


def _reverb_ir(sr, seconds=1.4, seed=1):
    rng = np.random.default_rng(seed)
    n = int(seconds * sr)
    t = np.arange(n) / sr
    ir = rng.normal(0, 1, (2, n)) * np.exp(-t * 6.9 / seconds)   # -60 dB at `seconds`
    ir[:, : int(0.012 * sr)] = 0                                   # pre-delay
    # darken the tail (one-pole low-pass)
    from scipy.signal import lfilter
    ir = lfilter([0.25], [1, -0.75], ir, axis=1)
    return ir / np.sqrt((ir ** 2).sum(axis=1, keepdims=True))


def render_notes(notes, total_s, sf_path, bank=0, prog=0, sr=44100, transpose=0,
                 legato_s=0.04, reverb=0.18, progress=None, tune_cents=0.0):
    """notes: iterable with .start .end .midi .velocity. Returns stereo (2, n) float32."""
    from scipy.signal import fftconvolve
    sf = load_soundfont(str(sf_path))
    out = np.zeros((2, int((total_s + 2) * sr)), dtype=np.float32)
    notes = list(notes)
    for k, n in enumerate(notes):
        # slight overlap into the next note for a connected, singing line
        dur = n.end - n.start + legato_s
        y = sf.render_note(int(n.midi + transpose), int(np.clip(n.velocity, 1, 127)), dur, sr, bank, prog, tune_cents)
        i = int(n.start * sr)
        j = min(out.shape[1], i + y.shape[1])
        out[:, i:j] += y[:, : j - i]
        if progress and k % 8 == 0:
            progress(k / max(1, len(notes)))
    if reverb > 0:
        ir = _reverb_ir(sr)
        wet = np.stack([fftconvolve(out[c], ir[c])[: out.shape[1]] for c in range(2)])
        out = out * (1 - reverb) + wet * reverb * 1.2
    out = out[:, : int(total_s * sr)]
    peak = np.abs(out).max()
    return (out / peak * 0.85 if peak > 0 else out).astype(np.float32)
