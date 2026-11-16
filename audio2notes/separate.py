"""Vocal isolation with Demucs (Hybrid Transformer Demucs)."""
from functools import lru_cache

import torch


@lru_cache(maxsize=2)
def _load(model_name):
    from demucs.pretrained import get_model
    model = get_model(model_name)
    model.eval()
    return model


def separate_vocals(audio, sr, model_name="htdemucs_ft", device=None, shifts=1, progress=None):
    """audio: (2, samples) float32 at model samplerate (44.1k).
    progress: optional callable(fraction 0..1).
    Returns dict of stems {name: (2, samples)} incl. 'vocals' and 'accompaniment'."""
    from demucs.apply import apply_model

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = _load(model_name)
    if sr != model.samplerate:
        raise ValueError(f"expected {model.samplerate} Hz input, got {sr}")

    length = audio.shape[1]
    seg_len = None

    def cb(d):
        nonlocal seg_len
        if progress is None or d.get("state") != "end":
            return
        if seg_len is None:  # stride between segments ~ segment length * (1 - overlap)
            sub = model.models[0] if hasattr(model, "models") else model
            seg_len = int(float(sub.segment) * sr * 0.75)
        within = min(1.0, (d["segment_offset"] + seg_len) / length)
        frac = (d["model_idx_in_bag"] + (d.get("shift_idx", 0) + within) / shifts) / d["models"]
        progress(min(frac, 1.0))

    wav = torch.from_numpy(audio)
    ref = wav.mean(0)
    mean, std = ref.mean(), ref.std() + 1e-8
    wav = (wav - mean) / std
    with torch.no_grad():
        out = apply_model(model, wav[None], device=device, shifts=shifts, split=True,
                          overlap=0.25, progress=progress is None, num_workers=0, callback=cb)[0]
    out = out * std + mean
    stems = {name: out[i].cpu().numpy() for i, name in enumerate(model.sources)}
    stems["accompaniment"] = sum(v for k, v in stems.items() if k != "vocals")
    return stems
