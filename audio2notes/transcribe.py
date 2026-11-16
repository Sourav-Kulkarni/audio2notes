"""Pitch track -> notes (detailed + simple), tuning, tonic, key, scale.
Shared by the CLI pipeline and the GUI."""
from dataclasses import dataclass, field

import numpy as np

from . import notes as nt


@dataclass
class TranscribeParams:
    hysteresis: float = 0.2
    min_note: float = 0.08
    ornament: float = 0.15
    merge_gap: float = 0.12
    no_tuning: bool = False
    tonic: str = ""            # "" = auto, else e.g. "C#"
    scale_min: float = 0.03
    snap_max: float = 0.4
    no_snap: bool = False
    simple_min: float = 0.25
    smooth: float = 0.05        # median smoothing of the contour (s)
    glide_max: float = 0.2      # short notes with a moving pitch below this are glides
    split_onsets: bool = True   # split notes at syllable onsets
    onset_delta: float = 0.12   # onset sensitivity (lower = more onsets)

    @classmethod
    def from_dict(cls, d):
        names = cls.__dataclass_fields__
        return cls(**{k: type(getattr(cls, k))(v) if not isinstance(getattr(cls, k), bool) else bool(v)
                      for k, v in (d or {}).items() if k in names and v is not None})


# Detail level (1 = coarse ... 5 = very fine) -> segmentation parameters.  Level 3 = defaults.
# Single source of truth: the GUI fetches these from /api/config, the CLI uses --detail.
DETAIL_LEVELS = ["Coarse", "Relaxed", "Normal", "Fine", "Very fine"]
DETAIL_PRESETS = {
    "min_note":   [0.25, 0.14, 0.08, 0.05, 0.03],
    "ornament":   [0.30, 0.20, 0.15, 0.07, 0.00],
    "hysteresis": [0.35, 0.28, 0.20, 0.12, 0.06],
    "glide_max":  [0.30, 0.25, 0.20, 0.10, 0.00],
    "smooth":     [0.09, 0.07, 0.05, 0.03, 0.02],
}


def detail_params(level):
    level = int(min(5, max(1, level)))
    return {k: v[level - 1] for k, v in DETAIL_PRESETS.items()}


def parse_tonic(s):
    s = s.strip().upper().replace("♯", "#").replace("SHARP", "#")
    flats = {"DB": "C#", "EB": "D#", "GB": "F#", "AB": "G#", "BB": "A#"}
    s = flats.get(s, s)
    if s not in nt.NOTE_NAMES:
        raise ValueError(f"bad tonic '{s}', use e.g. C, C#, Db")
    return nt.NOTE_NAMES.index(s)


@dataclass
class Transcription:
    notes: list            # detailed (scale-snapped)
    simple: list
    raw_count: int
    midi_corr: np.ndarray  # tuning-corrected fractional midi per frame (nan unvoiced)
    smooth: np.ndarray
    tuning: float          # semitones; sung pitch = corrected + tuning
    hist: np.ndarray
    keys: list
    sa_rank: list
    sa_pc: int
    sa_midi: int
    key_str: str
    scale_str: str
    scale: set = field(default_factory=set)

    def info(self):
        return dict(key=self.key_str, tuning_cents=self.tuning * 100, scale=self.scale_str)


def transcribe(pt, p: TranscribeParams, onsets=None, tuning=None, sa_midi=None, scale=None):
    """tuning/sa_midi/scale may be fixed (used for re-transcribing a sub-range)."""
    hop_s = float(pt["hop_s"])
    midi_f = nt.hz_to_midi(pt["f0"])
    if tuning is None:
        tuning = 0.0 if p.no_tuning else nt.estimate_tuning(midi_f, hop_s)
    midi_c = midi_f - tuning
    notes, smooth = nt.segment_notes(pt["time"], midi_c, pt["loudness_db"], hop_s,
                                     hysteresis=p.hysteresis, min_note_s=p.min_note,
                                     ornament_s=p.ornament, merge_gap_s=p.merge_gap,
                                     smooth_s=p.smooth, glide_max_s=p.glide_max)
    if not notes:
        raise ValueError("no notes detected - check the vocals and the voicing thresholds")
    if p.split_onsets and onsets is not None and len(onsets):
        notes = nt.split_at_onsets(notes, onsets, pt["time"], midi_c, hop_s, min_piece=max(p.min_note, 0.05))
    hist = nt.pitch_class_hist(notes)
    keys = nt.estimate_key(hist)
    sa_rank = nt.estimate_sa(hist, notes)
    if sa_midi is not None:
        sa_pc = sa_midi % 12
    else:
        sa_pc = parse_tonic(p.tonic) if p.tonic else sa_rank[0][1]
    key_str = f"{nt.NOTE_NAMES[keys[0][1]]} {keys[0][2]}"
    # madhya Sa octave: the Sa nearest a 5th below the median sung note
    med = float(np.median([n.midi for n in notes]))
    if sa_midi is None:
        sa_midi = int(min((m for m in range(24, 96) if m % 12 == sa_pc), key=lambda m: abs(m - (med - 5))))

    if scale is None:
        scale = nt.scale_from_hist(hist, sa_pc, p.scale_min)
    raw_count = len(notes)
    if not p.no_snap:
        notes = nt.snap_to_scale(notes, scale, max_dur=p.snap_max, merge_gap_s=p.merge_gap)
    simple = nt.simplify(notes, min_dur=p.simple_min, merge_gap_s=p.merge_gap)
    scale_str = " ".join(nt.SARGAM[(pc - sa_pc) % 12] for pc in sorted(scale, key=lambda q: (q - sa_pc) % 12))
    return Transcription(notes, simple, raw_count, midi_c, smooth, tuning, hist, keys, sa_rank,
                         sa_pc, sa_midi, key_str, scale_str, scale)
