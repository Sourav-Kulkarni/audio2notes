"""Vocal f0 tracking with CREPE (torchcrepe) + voicing / loudness gating."""
import numpy as np
import librosa
import torch
import torchcrepe
from scipy.ndimage import median_filter

CREPE_SR = 16000


def track_pitch(vocals_mono, sr, hop_s=0.01, fmin=65.0, fmax=1100.0,
                model="full", periodicity_thresh=0.5, silence_db=-40.0,
                device=None, batch_size=512, progress=None):
    """Returns dict of per-frame arrays: time, f0 (Hz, nan when unvoiced),
    f0_raw, periodicity, loudness_db, voiced (bool)."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    x = librosa.resample(vocals_mono.astype(np.float32), orig_sr=sr, target_sr=CREPE_SR)
    hop = int(round(hop_s * CREPE_SR))
    torch.set_num_threads(max(1, torch.get_num_threads()))
    # same as torchcrepe.predict, unrolled so we can report progress per batch
    audio_t = torch.from_numpy(x)[None]
    total_frames = 1 + len(x) // hop
    n_batches = max(1, -(-total_frames // batch_size))
    f0s, pers = [], []
    with torch.no_grad():
        for b, frames in enumerate(torchcrepe.preprocess(audio_t, CREPE_SR, hop, batch_size, device, True)):
            probs = torchcrepe.infer(frames, model, device)
            probs = probs.reshape(1, -1, torchcrepe.PITCH_BINS).transpose(1, 2)
            f, p = torchcrepe.postprocess(probs, fmin, fmax, torchcrepe.decode.viterbi,
                                          return_periodicity=True)
            f0s.append(f.cpu())
            pers.append(p.cpu())
            if progress:
                progress(min(1.0, (b + 1) / n_batches))
    f0 = torch.cat(f0s, 1)[0].numpy()
    per = torch.cat(pers, 1)[0].numpy()

    n = len(f0)
    rms = librosa.feature.rms(y=x, frame_length=2 * hop * 2, hop_length=hop, center=True)[0]
    rms = np.pad(rms, (0, max(0, n - len(rms))))[:n]
    loud_db = 20 * np.log10(rms + 1e-9)
    loud_db -= np.percentile(loud_db, 99)  # relative to (near) peak loudness

    pt = dict(time=np.arange(n) * hop_s, f0_raw=f0, periodicity_raw=per,
              loudness_db=loud_db, hop_s=hop_s)
    apply_voicing(pt, periodicity_thresh, silence_db)
    return pt


def _runs(mask):
    d = np.diff(np.concatenate([[0], mask.astype(int), [0]]))
    return list(zip(np.where(d == 1)[0], np.where(d == -1)[0]))


def apply_voicing(pt, periodicity_thresh=0.5, silence_db=-40.0, smooth_s=0.07,
                  max_gap_s=0.12, min_island_s=0.05, max_jump_st=1.0):
    """Compute pt['periodicity'], pt['voiced'], pt['f0'] (in place) from raw tracks.
    CREPE periodicity on separated vocals is jittery, so: median-smooth it, threshold,
    then bridge short unvoiced gaps when the raw pitch is continuous across them, and
    drop tiny voiced islands."""
    hop_s = float(pt["hop_s"])
    f0, loud = pt["f0_raw"], pt["loudness_db"]
    win = max(3, int(round(smooth_s / hop_s)) | 1)
    per = median_filter(pt["periodicity_raw"], size=win, mode="nearest")
    v = (per >= periodicity_thresh) & (loud >= silence_db)
    m = 69 + 12 * np.log2(np.maximum(f0, 1e-3) / 440)

    max_gap = int(round(max_gap_s / hop_s))
    for a, b in _runs(~v):
        if a == 0 or b == len(v) or b - a > max_gap:
            continue
        seg = m[a - 1:b + 1]
        if (np.abs(np.diff(seg)).max() <= max_jump_st and abs(seg[-1] - seg[0]) <= max_jump_st
                and (loud[a:b] >= silence_db).all()):
            v[a:b] = True

    min_isl = int(round(min_island_s / hop_s))
    for a, b in _runs(v):
        if b - a < min_isl:
            v[a:b] = False

    pt["periodicity"] = per
    pt["voiced"] = v
    pt["f0"] = np.where(v, f0, np.nan)
    return pt
