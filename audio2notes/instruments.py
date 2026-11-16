"""Instrument voices for rendering notes to audio (real samples from SoundFonts)."""
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SF_GM = ROOT / "third_party" / "soundfonts" / "GeneralUser_GS.sf2"

# name -> (label, [(soundfont, bank, program, transpose, gain), ...])
INSTRUMENTS = {
    "harmonium": ("Harmonium (layered free reeds)", [(SF_GM, 0, 21, 0, 1.0), (SF_GM, 0, 20, 0, 0.45)]),
    "accordion": ("Accordion (free reed)", [(SF_GM, 0, 21, 0, 1.0)]),
    "reed_organ": ("Reed organ", [(SF_GM, 0, 20, 0, 1.0)]),
    "shehnai": ("Shehnai", [(SF_GM, 0, 111, 0, 1.0)]),
    "violin": ("Violin", [(SF_GM, 0, 40, 0, 1.0)]),
    "flute": ("Flute", [(SF_GM, 0, 73, 0, 1.0)]),
    "piano": ("Grand piano", [(SF_GM, 0, 0, 0, 1.0)]),
    "sitar": ("Sitar", [(SF_GM, 0, 104, 0, 1.0)]),
    "synth": ("Basic synth (no samples)", []),
}
DEFAULT = "harmonium"


def available():
    out = {}
    for k, (label, layers) in INSTRUMENTS.items():
        if all(Path(l[0]).exists() for l in layers):
            out[k] = label
    return out


def render(notes, total_s, name=DEFAULT, sr=44100, transpose=0, progress=None, tune_cents=0.0):
    """Returns stereo float32 (2, n). tune_cents detunes the instrument to match the recording."""
    label, layers = INSTRUMENTS.get(name, INSTRUMENTS[DEFAULT])
    if not layers or not all(Path(l[0]).exists() for l in layers):
        from .render import synth_harmonium
        mono = synth_harmonium(notes, total_s, sr, transpose=transpose + tune_cents / 100)
        return np.stack([mono, mono]).astype(np.float32)
    from .sf2synth import render_notes
    mix = None
    for i, (sf, bank, prog, tr, gain) in enumerate(layers):
        cb = (lambda f, i=i: progress((i + f) / len(layers))) if progress else None
        y = render_notes(notes, total_s, sf, bank, prog, sr, transpose + tr, progress=cb, tune_cents=tune_cents) * gain
        mix = y if mix is None else mix[:, :y.shape[1]] + y[:, :mix.shape[1]]
    peak = np.abs(mix).max()
    return (mix / peak * 0.85 if peak > 0 else mix).astype(np.float32)
