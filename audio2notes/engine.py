"""GUI-facing engine: a Session holds one loaded song; steps work on a selected time range.

  load()            decode the whole file (+ a light preview wav for the waveform view)
  extract(a, b)     separate vocals for [a, b] and track pitch      (cached per selection)
  notate(params)    notes / tonic / notation + instrument renders  (fast, re-runnable)
"""
import json
import time
import uuid
from pathlib import Path

import numpy as np

from . import audio_io as aio
from . import notes as nt
from . import render as rd
from .transcribe import TranscribeParams, transcribe

SR = 44100


def _f(x, nd=3):
    """json-safe float list (nan -> None)."""
    return [None if not np.isfinite(v) else round(float(v), nd) for v in x]


class Session:
    def __init__(self, src_path, root):
        self.id = uuid.uuid4().hex[:10]
        self.src = Path(src_path)
        self.dir = Path(root) / self.id
        self.dir.mkdir(parents=True, exist_ok=True)
        self.audio = None
        self.duration = 0.0
        self.sel = None        # dict for the current extracted selection
        self.notation = None

    # ------------------------------------------------------------------ load
    def load(self):
        self.audio = aio.load_audio(self.src, sr=SR, channels=2)
        self.duration = self.audio.shape[1] / SR
        import librosa
        prev = librosa.resample(aio.to_mono(self.audio), orig_sr=SR, target_sr=22050)
        aio.write_wav(self.dir / "preview.wav", prev, 22050)
        return dict(sid=self.id, name=self.src.name, duration=self.duration,
                    preview=f"/media/{self.id}/preview.wav")

    # ------------------------------------------------------------------ extract
    def extract(self, start, end, model="htdemucs_ft", progress=lambda f, s: None):
        start = max(0.0, float(start))
        end = min(self.duration, float(end))
        if end - start < 1.0:
            raise ValueError("selection must be at least 1 second long")
        key = f"sel_{start:.2f}_{end:.2f}_{model}"
        sdir = self.dir / key
        sdir.mkdir(exist_ok=True)
        clip = self.audio[:, int(start * SR):int(end * SR)]

        if not (sdir / "vocals.wav").exists():
            from .separate import separate_vocals
            progress(0.0, "Loading separation model")
            t0 = time.time()
            stems = separate_vocals(clip, SR, model_name=model,
                                    progress=lambda f: progress(0.02 + 0.78 * f, "Separating vocals from the music"))
            aio.write_wav(sdir / "selection.wav", clip, SR)
            aio.write_wav(sdir / "vocals.wav", stems["vocals"], SR)
            aio.write_wav(sdir / "accompaniment.wav", stems["accompaniment"], SR)
            (sdir / "timing.json").write_text(json.dumps({"separation_s": time.time() - t0}))
        vocals, _ = aio.read_wav(sdir / "vocals.wav")

        if not (sdir / "pitch.npz").exists():
            from .pitch import track_pitch
            progress(0.8, "Tracking vocal pitch")
            pt = track_pitch(aio.to_mono(vocals), SR,
                             progress=lambda f: progress(0.8 + 0.2 * f, "Tracking vocal pitch"))
            np.savez_compressed(sdir / "pitch.npz", **pt)
        progress(1.0, "Done")
        self.sel = dict(start=start, end=end, model=model, dir=sdir, key=key)
        self.notation = None
        return self.pitch_payload()

    def _pitch(self, periodicity=0.5, silence_db=-40.0):
        from .pitch import apply_voicing
        pt = dict(np.load(self.sel["dir"] / "pitch.npz"))
        apply_voicing(pt, periodicity, silence_db)
        return pt

    def pitch_payload(self, periodicity=0.5, silence_db=-40.0):
        pt = self._pitch(periodicity, silence_db)
        off = self.sel["start"]
        midi = nt.hz_to_midi(pt["f0"])
        raw = nt.hz_to_midi(np.maximum(pt["f0_raw"], 1e-3))
        base = f"/media/{self.id}/{self.sel['key']}"
        return dict(start=self.sel["start"], end=self.sel["end"],
                    time=_f(pt["time"] + off, 3), midi=_f(midi, 3), midi_raw=_f(raw, 3),
                    periodicity=_f(pt["periodicity"], 3), voiced_pct=float(np.mean(pt["voiced"]) * 100),
                    audio=dict(selection=f"{base}/selection.wav", vocals=f"{base}/vocals.wav",
                               accompaniment=f"{base}/accompaniment.wav"))

    # ------------------------------------------------------------------ notate
    def _split_params(self, params):
        params = dict(params or {})
        extra = dict(instrument=params.pop("instrument", "harmonium"),
                     transpose=int(params.pop("transpose", 0) or 0),
                     periodicity=float(params.pop("periodicity", 0.5)),
                     silence_db=float(params.pop("silence_db", -40.0)))
        return TranscribeParams.from_dict(params), extra

    def _onsets(self, delta):
        key = round(float(delta), 3)
        cache = self.sel.setdefault("onsets", {})
        if key not in cache:
            if "vocals_mono" not in self.sel:
                voc, _ = aio.read_wav(self.sel["dir"] / "vocals.wav")
                self.sel["vocals_mono"] = aio.to_mono(voc)
            cache[key] = nt.detect_onsets(self.sel["vocals_mono"], SR, delta=key)
        return cache[key]

    def _note_json(self, x, tr):
        off = self.sel["start"]
        return dict(start=round(float(x.start) + off, 3), end=round(float(x.end) + off, 3), midi=int(x.midi),
                    sung=round(float(x.midi_float + tr.tuning), 3),  # actual sung pitch (uncorrected)
                    cents=int(round((x.midi_float - x.midi) * 100)), velocity=int(x.velocity),
                    onset=bool(getattr(x, "onset", False)),
                    western=nt.midi_name(x.midi), sargam=nt.note_to_sargam(x.midi, tr.sa_midi))

    def _write_variant(self, name, nl, tr, instrument, transpose, progress):
        """Notation files + instrument renders for a list of Note (selection-relative times)."""
        from .instruments import render as render_instrument
        sdir = self.sel["dir"]
        ndir = sdir / "notation"
        ndir.mkdir(exist_ok=True)
        dur = self.sel["end"] - self.sel["start"]
        if "clip_m" not in self.sel:
            clip, _ = aio.read_wav(sdir / "selection.wav")
            voc, _ = aio.read_wav(sdir / "vocals.wav")
            self.sel["clip_m"] = aio.to_mono(clip) / (np.abs(clip).max() + 1e-9)
            self.sel["voc_m"] = aio.to_mono(voc) / (np.abs(voc).max() + 1e-9)
        clip_m, voc_m = self.sel["clip_m"], self.sel["voc_m"]
        base = f"/media/{self.id}/{self.sel['key']}/notation"
        stamp = uuid.uuid4().hex[:6]  # cache-buster for the browser
        rd.write_notes_csv(nl, ndir / f"notes_{name}.csv", tr.sa_midi)
        rd.write_midi(nl, ndir / f"notes_{name}.mid", transpose=transpose)
        rd.write_notation(nl, ndir / f"notation_{name}.txt", tr.sa_midi, tr.info())
        y = render_instrument(nl, dur, instrument, SR, transpose=transpose, progress=progress,
                              tune_cents=tr.tuning * 100)
        n = min(y.shape[1], len(voc_m))
        aio.write_wav(ndir / f"inst_{name}.wav", y, SR)
        aio.write_wav(ndir / f"vocals+inst_{name}.wav", np.clip(voc_m[:n] * 0.75 + y[:, :n] * 0.55, -1, 1), SR)
        aio.write_wav(ndir / f"original+inst_{name}.wav", np.clip(clip_m[:n] * 0.65 + y[:, :n] * 0.5, -1, 1), SR)
        return dict(
            notes=[self._note_json(x, tr) for x in nl],
            notation=(ndir / f"notation_{name}.txt").read_text(encoding="utf-8"),
            audio={"instrument": f"{base}/inst_{name}.wav?v={stamp}",
                   "vocals+instrument": f"{base}/vocals+inst_{name}.wav?v={stamp}",
                   "original+instrument": f"{base}/original+inst_{name}.wav?v={stamp}"},
            files={"MIDI": f"{base}/notes_{name}.mid?v={stamp}", "CSV": f"{base}/notes_{name}.csv?v={stamp}",
                   "Notation (txt)": f"{base}/notation_{name}.txt?v={stamp}"})

    def notate(self, params=None, progress=lambda f, s: None):
        if self.sel is None:
            raise ValueError("extract vocals first")
        tp, extra = self._split_params(params)
        progress(0.02, "Finding syllable onsets & segmenting notes")
        pt = self._pitch(extra["periodicity"], extra["silence_db"])
        onsets = self._onsets(tp.onset_delta)
        tr = transcribe(pt, tp, onsets=onsets if tp.split_onsets else None)
        self.tr, self.tp, self.extra = tr, tp, extra
        instrument = extra["instrument"]
        out = dict(sa=nt.NOTE_NAMES[tr.sa_pc], sa_midi=tr.sa_midi, tuning_cents=tr.tuning * 100,
                   key=tr.key_str, scale=tr.scale_str, raw_count=tr.raw_count,
                   sa_candidates=[[nt.NOTE_NAMES[pc], round(s, 3)] for s, pc in tr.sa_rank[:4]],
                   key_candidates=[[f"{nt.NOTE_NAMES[t]} {m}", round(s, 3)] for s, t, m in tr.keys[:4]],
                   hist={nt.NOTE_NAMES[i]: round(float(h), 3) for i, h in enumerate(tr.hist)},
                   onsets=[round(float(t) + self.sel["start"], 3) for t in onsets],
                   instrument=instrument, variants={})
        for vi, (name, nl) in enumerate({"detailed": tr.notes, "simple": tr.simple}.items()):
            cb = lambda f, vi=vi, name=name: progress(0.1 + 0.85 * (vi + f) / 2, f"Rendering {name} notes on {instrument}")
            out["variants"][name] = self._write_variant(name, nl, tr, instrument, extra["transpose"], cb)
        progress(1.0, "Done")
        self.notation = out
        return out

    # ------------------------------------------------------------------ refine (step 4)
    def refine(self, start, end, params=None):
        """Re-transcribe only [start, end] (song time) with its own detail parameters, keeping
        the tuning, Sa and scale of the current notation. Fast; returns note dicts."""
        if getattr(self, "tr", None) is None:
            raise ValueError("extract notation first")
        tp, extra = self._split_params(params)
        off = self.sel["start"]
        a, b = float(start) - off, float(end) - off
        pt = self._pitch(extra["periodicity"], extra["silence_db"])
        sel = (pt["time"] >= a) & (pt["time"] < b)
        if sel.sum() < 5:
            return dict(notes=[])
        n = len(pt["time"])
        sub = {k: (v[sel] if isinstance(v, np.ndarray) and v.ndim == 1 and len(v) == n else v)
               for k, v in pt.items()}
        onsets = self._onsets(tp.onset_delta)
        onsets = onsets[(onsets >= a) & (onsets < b)]
        try:
            tr = transcribe(sub, tp, onsets=onsets if tp.split_onsets else None,
                            tuning=self.tr.tuning, sa_midi=self.tr.sa_midi, scale=self.tr.scale)
        except ValueError:
            return dict(notes=[])
        return dict(notes=[self._note_json(x, self.tr) for x in tr.notes])

    def render_edited(self, notes, instrument=None, transpose=None, progress=lambda f, s: None):
        """Render a user-edited note list (song-time dicts with start/end/midi[/sung/velocity])."""
        if getattr(self, "tr", None) is None:
            raise ValueError("extract notation first")
        off = self.sel["start"]
        instrument = instrument or self.extra["instrument"]
        transpose = self.extra["transpose"] if transpose is None else int(transpose)
        nl = []
        for d in sorted(notes, key=lambda d: d["start"]):
            a, b = float(d["start"]) - off, float(d["end"]) - off
            if b - a < 0.01:
                continue
            m = int(d["midi"])
            sung = d.get("sung")
            mf = float(sung) - self.tr.tuning if sung is not None else float(m)
            nl.append(nt.Note(a, b, m, mf, int(d.get("velocity", 96)), bool(d.get("onset", False))))
        if not nl:
            raise ValueError("no notes to render")
        progress(0.05, f"Rendering edited notes on {instrument}")
        res = self._write_variant("edited", nl, self.tr, instrument, transpose,
                                  lambda f: progress(0.05 + 0.9 * f, f"Rendering edited notes on {instrument}"))
        progress(1.0, "Done")
        return res
