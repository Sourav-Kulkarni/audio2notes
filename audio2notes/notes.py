"""Turn an f0 contour into discrete notes; estimate tuning offset, tonic (Sa) and key."""
from dataclasses import dataclass, asdict

import numpy as np
from scipy.ndimage import median_filter

NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
# semitone offset from Sa -> sargam symbol (lowercase = komal, M = tivra Ma)
SARGAM = ["S", "r", "R", "g", "G", "m", "M", "P", "d", "D", "n", "N"]

KS_MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
KS_MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])


@dataclass
class Note:
    start: float
    end: float
    midi: int          # tuning-corrected, quantized
    midi_float: float  # median sung pitch (tuning-corrected)
    velocity: int
    onset: bool = False  # starts at a syllable onset -> never merged into the previous note

    @property
    def dur(self):
        return self.end - self.start


def hz_to_midi(f):
    return 69 + 12 * np.log2(f / 440.0)


def midi_name(m):
    return f"{NOTE_NAMES[m % 12]}{m // 12 - 1}"


def voiced_runs(mask):
    """List of (start, end_exclusive) index pairs of True runs."""
    d = np.diff(np.concatenate([[0], mask.astype(int), [0]]))
    return list(zip(np.where(d == 1)[0], np.where(d == -1)[0]))


def estimate_tuning(midi_f, hop_s, max_slope=1.5):
    """Circular mean of deviation from 12-TET over *stable* frames.
    Returns offset in semitones (-0.5..0.5); e.g. +0.2 => singer is 20 cents sharp of A440."""
    m = midi_f.copy()
    slope = np.abs(np.gradient(m)) / hop_s  # semitones/sec
    ok = np.isfinite(m) & np.isfinite(slope) & (slope < max_slope)
    if ok.sum() < 20:
        ok = np.isfinite(m)
    if ok.sum() == 0:
        return 0.0
    ang = 2 * np.pi * m[ok]
    return float(np.angle(np.mean(np.exp(1j * ang))) / (2 * np.pi))


def smooth_contour(midi_f, win):
    """Median filter inside each voiced run only (never across silences)."""
    out = np.full_like(midi_f, np.nan)
    for a, b in voiced_runs(np.isfinite(midi_f)):
        seg = midi_f[a:b]
        out[a:b] = median_filter(seg, size=min(win, len(seg)), mode="nearest") if len(seg) >= 3 else seg
    return out


def segment_notes(time, midi_f, loud_db, hop_s, hysteresis=0.2, min_note_s=0.08,
                  merge_gap_s=0.12, smooth_s=0.05, glide_max_s=0.2, glide_travel=0.6, ornament_s=0.15):
    """midi_f: tuning-corrected fractional midi per frame (nan = unvoiced)."""
    win = max(1, int(round(smooth_s / hop_s)) | 1)
    m = smooth_contour(midi_f, win)
    q = np.full(len(m), -1, dtype=int)
    for a, b in voiced_runs(np.isfinite(m)):
        cur = int(round(m[a]))
        for i in range(a, b):
            if abs(m[i] - cur) > 0.5 + hysteresis:
                cur = int(round(m[i]))
            q[i] = cur

    # raw segments: runs of identical quantized pitch
    segs = []  # [start_idx, end_idx_excl, pitch]
    i = 0
    while i < len(q):
        if q[i] < 0:
            i += 1
            continue
        j = i
        while j < len(q) and q[j] == q[i]:
            j += 1
        segs.append([i, j, q[i]])
        i = j

    min_frames = int(round(min_note_s / hop_s))
    # absorb too-short segments into an adjacent touching neighbour (glides / ornaments)
    changed = True
    while changed:
        changed = False
        for k, (a, b, p) in enumerate(segs):
            if b - a >= min_frames:
                continue
            prev = segs[k - 1] if k > 0 and segs[k - 1][1] == a else None
            nxt = segs[k + 1] if k + 1 < len(segs) and segs[k + 1][0] == b else None
            cands = [s for s in (prev, nxt) if s is not None]
            if cands:
                tgt = min(cands, key=lambda s: (abs(s[2] - p), -(s[1] - s[0])))
                tgt[0], tgt[1] = min(tgt[0], a), max(tgt[1], b)
            segs.pop(k)
            changed = True
            break

    # glide (meend) filter: a short segment whose pitch keeps moving through it is a
    # transition between notes, not a note -> give its time to a touching neighbour
    glide_frames = int(round(glide_max_s / hop_s))
    k = 0
    while k < len(segs):
        a, b, p = segs[k]
        if b - a < glide_frames and b - a >= 3:
            third = max(1, (b - a) // 3)
            travel = abs(np.nanmedian(m[b - third:b]) - np.nanmedian(m[a:a + third]))
            if travel >= glide_travel:
                prev = segs[k - 1] if k > 0 and segs[k - 1][1] >= a - 1 else None
                nxt = segs[k + 1] if k + 1 < len(segs) and segs[k + 1][0] <= b + 1 else None
                if prev is not None:
                    prev[1] = b
                elif nxt is not None:
                    nxt[0] = a
                segs.pop(k)
                continue
        k += 1

    # merge same-pitch neighbours separated by short gaps
    merged = []
    gap_frames = int(round(merge_gap_s / hop_s))
    for s in segs:
        if merged and merged[-1][2] == s[2] and s[0] - merged[-1][1] <= gap_frames:
            merged[-1][1] = s[1]
        else:
            merged.append(list(s))

    merged = _absorb_ornaments(merged, int(round(ornament_s / hop_s)), int(round(0.15 / hop_s)), gap_frames)

    notes = []
    peak =np.nanmax(loud_db) if np.isfinite(loud_db).any() else 0
    for a, b, p in merged:
        mf = float(np.nanmedian(midi_f[a:b])) if np.isfinite(midi_f[a:b]).any() else float(p)
        ld = float(np.nanmean(loud_db[a:b]))
        vel = int(np.clip(64 + (ld - peak) * 2 + 40, 30, 120))
        notes.append(Note(float(time[a]), float(time[b - 1] + hop_s), int(p), mf, vel))
    return notes, m


def _absorb_ornaments(segs, short, near, gap_frames):
    """Remove short notes that are (a) passing tones strictly between their neighbours'
    pitches or (b) excursions flanked by the same pitch on both sides (vibrato/kan).
    Their time goes to the neighbour; then same-pitch neighbours are re-merged."""
    if short <= 0:
        return segs
    changed = True
    while changed:
        changed = False
        for k in range(1, len(segs) - 1):
            a, b, p = segs[k]
            pv, nx = segs[k - 1], segs[k + 1]
            if b - a >= short or a - pv[1] > near or nx[0] - b > near:
                continue
            passing = min(pv[2], nx[2]) < p < max(pv[2], nx[2])
            excursion = pv[2] == nx[2] and abs(p - pv[2]) <= 2
            if passing or excursion:
                pv[1] = b if (pv[1] - pv[0]) >= (nx[1] - nx[0]) or excursion else pv[1]
                if pv[1] != b:
                    nx[0] = a
                segs.pop(k)
                changed = True
                break
    out = []
    for s in segs:
        if out and out[-1][2] == s[2] and s[0] - out[-1][1] <= gap_frames:
            out[-1][1] = max(out[-1][1], s[1])
        else:
            out.append(list(s))
    return out


def _merge_same(notes, gap_s):
    out = []
    for n in notes:
        if out and out[-1].midi == n.midi and n.start - out[-1].end <= gap_s and not n.onset:
            p = out[-1]
            w = p.dur + n.dur
            out[-1] = Note(p.start, n.end, p.midi, (p.midi_float * p.dur + n.midi_float * n.dur) / w,
                           max(p.velocity, n.velocity), p.onset)
        else:
            out.append(n)
    return out


def scale_from_hist(hist, sa_pc, min_share=0.03):
    """Pitch classes the singer actually dwells on (Sa and Pa always included)."""
    pcs = {i for i in range(12) if hist[i] >= min_share}
    return pcs | {sa_pc, (sa_pc + 7) % 12}


def snap_to_scale(notes, scale_pcs, max_dur=0.4, merge_gap_s=0.12):
    """Short notes outside the scale are passing tones of glides: move them to the nearest
    in-scale semitone (in the direction the sung pitch leans). Long ones are kept as
    genuine accidentals."""
    out = []
    for n in notes:
        if n.midi % 12 in scale_pcs or n.dur > max_dur:
            out.append(n)
            continue
        lean = 1 if n.midi_float >= n.midi else -1
        for step in (lean, -lean, 2 * lean, -2 * lean):
            if (n.midi + step) % 12 in scale_pcs:
                out.append(Note(n.start, n.end, n.midi + step, n.midi_float, n.velocity, n.onset))
                break
        else:
            out.append(n)
    return _merge_same(out, merge_gap_s)


def simplify(notes, min_dur=0.25, join_gap_s=0.15, merge_gap_s=0.12):
    """Harmonium-friendly reduction: repeatedly fold the shortest note shorter than min_dur
    into its touching neighbour (closest in pitch, then longest). Isolated short blips are
    dropped if under half min_dur."""
    notes = list(notes)
    while True:
        cands = [(n.dur, i) for i, n in enumerate(notes) if n.dur < min_dur]
        if not cands:
            break
        progressed = False
        for _, i in sorted(cands):
            n = notes[i]
            nb = []
            if i > 0 and n.start - notes[i - 1].end <= join_gap_s:
                nb.append(i - 1)
            if i + 1 < len(notes) and notes[i + 1].start - n.end <= join_gap_s:
                nb.append(i + 1)
            if not nb:
                if n.dur < min_dur / 2:
                    notes.pop(i)
                    progressed = True
                    break
                continue
            j = min(nb, key=lambda j: (abs(notes[j].midi - n.midi), -notes[j].dur))
            t = notes[j]
            notes[j] = Note(min(t.start, n.start), max(t.end, n.end), t.midi, t.midi_float, t.velocity,
                            t.onset if t.start <= n.start else n.onset)
            notes.pop(i)
            notes = _merge_same(notes, merge_gap_s)
            progressed = True
            break
        if not progressed:
            break
    return _merge_same(notes, merge_gap_s)


def detect_onsets(vocals_mono, sr, delta=0.12, min_gap_s=0.08):
    """Syllable / articulation onsets in the isolated vocal (seconds)."""
    import librosa
    y = librosa.resample(vocals_mono.astype(np.float32), orig_sr=sr, target_sr=22050)
    hop = 220  # 10 ms
    env = librosa.onset.onset_strength(y=y, sr=22050, hop_length=hop, aggregate=np.median)
    return librosa.onset.onset_detect(onset_envelope=env, sr=22050, hop_length=hop, units="time",
                                      delta=delta, wait=max(1, int(min_gap_s * 100)), backtrack=False)


def split_at_onsets(notes, onsets, time, midi_c, hop_s, min_piece=0.08, near=0.04):
    """Split notes at syllable onsets that fall inside them; each piece gets its own
    median pitch. Notes that start at an onset are flagged so later merging keeps them."""
    onsets = np.asarray(onsets, dtype=float)
    out = []
    for n in notes:
        cuts = [t for t in onsets if n.start + min_piece <= t <= n.end - min_piece]
        pruned = []
        for t in cuts:
            if not pruned or t - pruned[-1] >= min_piece:
                pruned.append(t)
        starts_at_onset = bool(len(onsets) and np.min(np.abs(onsets - n.start)) <= near)
        bounds = [n.start] + pruned + [n.end]
        for k in range(len(bounds) - 1):
            a, b = bounds[k], bounds[k + 1]
            sel = (time >= a) & (time < b)
            seg = midi_c[sel]
            seg = seg[np.isfinite(seg)]
            if len(seg) >= 2:
                mf = float(np.median(seg))
                m = int(round(mf))
            else:
                mf, m = n.midi_float, n.midi
            out.append(Note(a, b, m, mf, n.velocity, onset=(k > 0) or starts_at_onset or n.onset))
    return out


def pitch_class_hist(notes):
    h = np.zeros(12)
    for n in notes:
        h[n.midi % 12] += n.dur
    return h / h.sum() if h.sum() > 0 else h


def estimate_key(hist):
    """Krumhansl-Schmuckler. Returns list of (score, tonic_pc, mode) best first."""
    res = []
    for t in range(12):
        r = np.roll(hist, -t)
        res.append((float(np.corrcoef(r, KS_MAJOR)[0, 1]), t, "major"))
        res.append((float(np.corrcoef(r, KS_MINOR)[0, 1]), t, "minor"))
    return sorted(res, reverse=True)


def estimate_sa(hist, notes):
    """Heuristic tonic (Sa) for Indian melodies: Sa & Pa are fixed drone notes and
    heavily used; Sa is never komal/tivra so its semitone neighbours tend to be weak
    and the final phrase tends to resolve on Sa. Returns ranked [(score, pc)]."""
    last_pc = None
    if notes:
        tail = [n for n in notes if n.end > notes[-1].end - 3.0]
        last_pc = max(tail, key=lambda n: n.dur).midi % 12
    out = []
    for t in range(12):
        s = (hist[t] + 0.6 * hist[(t + 7) % 12] + 0.25 * hist[(t + 5) % 12]
             - 0.3 * hist[(t + 6) % 12] - 0.2 * hist[(t + 1) % 12])
        if last_pc == t:
            s += 0.05
        out.append((float(s), t))
    return sorted(out, reverse=True)


def note_to_sargam(midi, sa_midi):
    """sa_midi: midi number of the reference (madhya) Sa. Returns e.g. 'S', 'n.', "R'"."""
    rel = midi - sa_midi
    sym = SARGAM[rel % 12]
    octv = rel // 12
    if octv < 0:
        return sym + "." * (-octv)    # mandra saptak (lower octave): dot(s) after
    return sym + "'" * octv            # taar saptak (upper octave): apostrophe(s)


def notes_to_dicts(notes):
    return [asdict(n) for n in notes]
