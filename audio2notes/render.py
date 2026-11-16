"""Outputs: MIDI, notation text/CSV, harmonium synthesis, diagnostic plots."""
import csv

import numpy as np
import pretty_midi

from .notes import midi_name, note_to_sargam, NOTE_NAMES, SARGAM


# ----------------------------------------------------------------- MIDI
def write_midi(notes, path, tempo=120.0, program=20, transpose=0):
    """program 20 = GM 'Reed Organ' (closest GM patch to harmonium)."""
    pm = pretty_midi.PrettyMIDI(initial_tempo=float(tempo))
    inst = pretty_midi.Instrument(program=program, name="Harmonium")
    for n in notes:
        inst.notes.append(pretty_midi.Note(n.velocity, n.midi + transpose, n.start, n.end))
    pm.instruments.append(inst)
    pm.write(str(path))


# ----------------------------------------------------------------- text
def write_notes_csv(notes, path, sa_midi):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["start_s", "end_s", "dur_s", "midi", "western", "sargam",
                    "cents_off_from_key", "velocity"])
        for n in notes:
            w.writerow([f"{n.start:.3f}", f"{n.end:.3f}", f"{n.dur:.3f}", n.midi,
                        midi_name(n.midi), note_to_sargam(n.midi, sa_midi),
                        f"{(n.midi_float - n.midi) * 100:+.0f}", n.velocity])


def fmt_time(t):
    return f"{int(t // 60):02d}:{t % 60:05.2f}"


def phrases(notes, gap_s=0.6):
    out, cur = [], []
    for n in notes:
        if cur and n.start - cur[-1].end > gap_s:
            out.append(cur)
            cur = []
        cur.append(n)
    if cur:
        out.append(cur)
    return out


def write_notation(notes, path, sa_midi, info, gap_s=0.3, hold_s=0.5, max_per_line=10):
    """Human-readable phrase-by-phrase notation. Notes held longer than hold_s get a
    '-' per extra hold_s (like the dash convention in sargam notation)."""
    sa_name = midi_name(sa_midi)
    lines = [
        "audio2notes - harmonium notation",
        "=" * 60,
        f"Tonic (Sa) = {NOTE_NAMES[sa_midi % 12]}   (madhya Sa = harmonium key {sa_name})",
        f"Western key estimate: {info['key']}",
        f"Singer tuning offset vs A440: {info['tuning_cents']:+.0f} cents "
        f"(notes below are corrected for it)",
        f"Scale (notes used): {info.get('scale', '?')}",
        f"Range: {midi_name(min(n.midi for n in notes))} .. {midi_name(max(n.midi for n in notes))}",
        "",
        "Sargam legend: S r R g G m M P d D n N  (lowercase = komal, M = tivra Ma)",
        "  N. = lower octave (mandra),  S' = upper octave (taar),  '-' = held",
        "",
    ]
    ref = "  ".join(f"{SARGAM[i]}={NOTE_NAMES[(sa_midi + i) % 12]}" for i in range(12))
    lines += ["Sargam -> key mapping: " + ref, "", "-" * 60]
    for ph in phrases(notes, gap_s):
        # wrap long phrases into lines of max_per_line notes, columns aligned
        for c in range(0, len(ph), max_per_line):
            chunk = ph[c:c + max_per_line]
            sarg, west = [], []
            for n in chunk:
                holds = int(n.dur // hold_s)
                s = note_to_sargam(n.midi, sa_midi) + " -" * holds
                w = midi_name(n.midi) + " -" * holds
                width = max(len(s), len(w))
                sarg.append(s.ljust(width))
                west.append(w.ljust(width))
            lines.append(f"[{fmt_time(chunk[0].start)} - {fmt_time(chunk[-1].end)}]")
            lines.append("  sargam : " + "  ".join(sarg).rstrip())
            lines.append("  keys   : " + "  ".join(west).rstrip())
            lines.append("")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


# ----------------------------------------------------------------- synthesis
def harmonium_tone(freq, dur, sr, vel=1.0, attack=0.04, release=0.12):
    """Simple reed-organ voice: band-limited, sawtooth-like additive tone, two slightly
    detuned reeds (harmonium 'coupler' beating), soft attack, short release."""
    n = int((dur + release) * sr)
    t = np.arange(n) / sr
    y = np.zeros(n)
    for detune in (1.0, 2 ** (4 / 1200)):  # +4 cents second reed
        f = freq * detune
        k_max = int(min(30, (sr / 2 * 0.9) // f))
        for k in range(1, k_max + 1):
            amp = 1.0 / k ** 0.9 * (1.15 if k % 2 else 0.85)  # reeds: odd-ish emphasis
            y += amp * np.sin(2 * np.pi * f * k * t + 0.3 * k)
    env = np.ones(n)
    a = int(attack * sr)
    r = int(release * sr)
    s_end = max(a, n - r)
    env[:a] = np.linspace(0, 1, a)
    env[s_end:] = np.linspace(1, 0, n - s_end)
    return y * env * vel


def synth_harmonium(notes, total_s, sr=44100, transpose=0):
    out = np.zeros(int((total_s + 1) * sr))
    for n in notes:
        f = 440.0 * 2 ** ((n.midi + transpose - 69) / 12)
        tone = harmonium_tone(f, n.dur, sr, vel=n.velocity / 127)
        i = int(n.start * sr)
        j = min(len(out), i + len(tone))
        out[i:j] += tone[: j - i]
    peak = np.abs(out).max()
    return (out / peak * 0.8 if peak > 0 else out)[: int(total_s * sr)]


# ----------------------------------------------------------------- plots
def plot_pitch_pages(pt, midi_corr, smooth, notes, sa_midi, out_dir, page_s=30.0, simple=None, onsets=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    t = pt["time"]
    total = t[-1] if len(t) else 0
    valid = midi_corr[np.isfinite(midi_corr)]
    if len(valid) == 0:
        return []
    lo, hi = np.floor(np.percentile(valid, 1)) - 3, np.ceil(np.percentile(valid, 99)) + 3
    raw_m = np.where(pt["f0_raw"] > 0, 69 + 12 * np.log2(np.maximum(pt["f0_raw"], 1e-3) / 440), np.nan)
    raw_m = raw_m - (pt.get("tuning_semitones", 0.0))
    paths = []
    start = 0.0
    while start < total:
        end = start + page_s
        sel = (t >= start) & (t < end)
        fig, (ax, ax2) = plt.subplots(2, 1, figsize=(18, 8), sharex=True,
                                      gridspec_kw=dict(height_ratios=[4, 1]))
        ax.plot(t[sel], raw_m[sel], color="0.85", lw=0.8, label="CREPE raw (incl. unvoiced)")
        ax.plot(t[sel], midi_corr[sel], color="tab:blue", lw=1.0, label="voiced f0")
        ax.plot(t[sel], smooth[sel], color="tab:orange", lw=0.8, alpha=0.7, label="smoothed")
        for n in notes:
            if n.end < start or n.start > end:
                continue
            ax.add_patch(Rectangle((n.start, n.midi - 0.4), n.dur, 0.8, color="tab:green", alpha=0.35))
            ax.text(n.start, n.midi + 0.5, note_to_sargam(n.midi, sa_midi), fontsize=7, color="darkgreen")
        for n in simple or []:
            if n.end < start or n.start > end:
                continue
            ax.add_patch(Rectangle((n.start, n.midi - 0.55), n.dur, 1.1, fill=False, ec="purple", lw=1.2))
        for t_on in (onsets if onsets is not None else []):
            if start <= t_on < end:
                ax.plot([t_on, t_on], [lo, lo + 0.8], color="0.45", lw=1.2)  # syllable onset tick
        ticks = np.arange(int(lo), int(hi) + 1)
        ax.set_yticks(ticks)
        ax.set_yticklabels([f"{midi_name(m)}  {note_to_sargam(m, sa_midi)}" for m in ticks], fontsize=7)
        for m in ticks:
            if (m - sa_midi) % 12 == 0:
                ax.axhline(m, color="red", lw=0.6, alpha=0.5)
        ax.set_ylim(lo, hi)
        ax.grid(axis="y", alpha=0.3)
        ax.set_ylabel("pitch (tuning-corrected) / sargam")
        ax.set_title(f"Vocal pitch & notes {fmt_time(start)}-{fmt_time(end)}   green = detailed, purple outline = simple, red line = Sa")
        ax.legend(loc="upper right", fontsize=7)
        ax2.plot(t[sel], pt["periodicity"][sel], lw=0.7, label="periodicity")
        ax2.plot(t[sel], np.clip(pt["loudness_db"][sel] / 60 + 1, 0, 1), lw=0.7, label="loudness (norm)")
        ax2.fill_between(t[sel], 0, pt["voiced"][sel].astype(float), color="tab:green", alpha=0.15, label="voiced")
        ax2.set_ylim(0, 1.05)
        ax2.legend(loc="upper right", fontsize=7)
        ax2.set_xlabel("time (s)")
        ax2.set_xlim(start, end)
        p = out_dir / f"pitch_{int(start):04d}s.png"
        fig.tight_layout()
        fig.savefig(p, dpi=110)
        plt.close(fig)
        paths.append(p)
        start = end
    return paths


def plot_histogram(hist, sa_pc, key_str, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 3.5))
    colors = ["red" if i == sa_pc else "tab:blue" for i in range(12)]
    ax.bar(range(12), hist, color=colors)
    ax.set_xticks(range(12))
    ax.set_xticklabels([f"{NOTE_NAMES[i]}\n{SARGAM[(i - sa_pc) % 12]}" for i in range(12)])
    ax.set_title(f"Pitch-class time distribution  (red = Sa = {NOTE_NAMES[sa_pc]};  key est. {key_str})")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
