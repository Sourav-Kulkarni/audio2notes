"""End-to-end pipeline with cached, inspectable intermediate outputs (same engine as the GUI).

Stages (each writes into the output folder; later runs reuse earlier results):
  1 decode    -> 01_input.wav
  2 separate  -> 02_vocals.wav, 02_accompaniment.wav, 02_stem_*.wav
  3 pitch     -> 03_pitch.npz, 03_pitch.csv
  4 notes     -> 04_onsets.csv, 04_notes_{detailed,simple}.{csv,mid}, 04_notation_*.txt, plots/*.png
  5 render    -> 05_harmonium_*.wav, 05_vocals+harmonium_*.wav, 05_original+harmonium_*.wav

Entry points for intermediate inputs:
  input = a GUI selection folder (outputs/gui/<session>/sel_...)   -> reuses its audio, vocals, pitch
  --vocals FILE        use your own isolated vocal track instead of running separation
  --no-separation      the input itself is a clean vocal
  --notes-csv FILE     skip transcription; render a (hand-edited) notes CSV
  --redo-from STAGE    recompute from STAGE on;  --stop-after STAGE  stop after STAGE
"""
import argparse
import os
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
import csv
import json
import shutil
import time
from dataclasses import fields
from pathlib import Path

import numpy as np

from . import audio_io as aio
from . import notes as nt
from . import render as rd
from .transcribe import TranscribeParams, transcribe, detail_params, DETAIL_PRESETS

STAGES = ["decode", "separate", "pitch", "notes", "render"]
SR = 44100


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def estimate_tempo(audio, sr):
    """Tempo (BPM) of the accompaniment; cosmetic (MIDI bar lines only)."""
    try:
        import librosa
        y = librosa.resample(aio.to_mono(audio), orig_sr=sr, target_sr=22050)
        return float(np.atleast_1d(librosa.beat.beat_track(y=y, sr=22050)[0])[0])
    except Exception as e:
        log(f"tempo estimate failed: {e}")
        return 120.0


def read_notes_csv(path):
    """Read a notes CSV (as written by this tool / downloaded from the GUI; hand edits welcome).
    Needs start_s, end_s and midi columns (or a 'western' column like F#4); velocity optional."""
    out = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            a, b = float(row["start_s"]), float(row["end_s"])
            if row.get("midi", "").strip():
                m = int(float(row["midi"]))
            else:
                w = row["western"].strip()
                m = nt.NOTE_NAMES.index(w[:-1].upper()) + 12 * (int(w[-1]) + 1)
            vel = int(float(row.get("velocity") or 96))
            cents = float(row.get("cents_off_from_key") or 0)
            out.append(nt.Note(a, b, m, m + cents / 100, vel))
    return sorted(out, key=lambda n: n.start)


def import_gui_selection(sel_dir, out):
    """Map a GUI selection folder onto pipeline stage files (no recomputation)."""
    sel_dir = Path(sel_dir)
    pairs = [("selection.wav", "01_input.wav"), ("vocals.wav", "02_vocals.wav"),
             ("accompaniment.wav", "02_accompaniment.wav"), ("pitch.npz", "03_pitch.npz")]
    for src, dst in pairs:
        if (sel_dir / src).exists() and not (out / dst).exists():
            shutil.copy2(sel_dir / src, out / dst)
    log(f"imported GUI selection {sel_dir.name}")


def resolve_params(args):
    """--detail preset, overridden by any explicitly given segmentation flag."""
    d = detail_params(args.detail)
    for k in DETAIL_PRESETS:
        if getattr(args, k) is None:
            setattr(args, k, d[k])
    names = {f.name for f in fields(TranscribeParams)}
    return TranscribeParams.from_dict({k: v for k, v in vars(args).items() if k in names})


def run(args):
    inp = Path(args.input)
    gui_sel = inp.is_dir()
    if gui_sel and not (inp / "vocals.wav").exists():
        raise SystemExit(f"{inp} is a folder but not a GUI selection folder (no vocals.wav)")
    default_name = f"{inp.parent.parent.name}_{inp.name}" if gui_sel else inp.stem
    out = Path(args.out) if args.out else Path("outputs") / default_name
    out.mkdir(parents=True, exist_ok=True)
    (out / "plots").mkdir(exist_ok=True)
    if gui_sel:
        import_gui_selection(inp, out)
    first = STAGES.index(args.redo_from) if args.redo_from else len(STAGES)
    redo = lambda st: STAGES.index(st) >= first
    stop = STAGES.index(args.stop_after) if args.stop_after else len(STAGES) - 1
    done = lambda st: STAGES.index(st) >= stop
    tp = resolve_params(args)
    summary = {"input": str(inp), "params": vars(args)}

    # ---------------------------------------------------------- 1 decode
    p_in = out / "01_input.wav"
    if (redo("decode") and not gui_sel) or not p_in.exists():
        log(f"decode {inp}")
        audio = aio.load_audio(inp, sr=SR, channels=2)
        if args.start or args.duration:
            a = int(args.start * SR)
            b = a + int(args.duration * SR) if args.duration else None
            audio = audio[:, a:b]
        aio.write_wav(p_in, audio, SR)
        first = min(first, STAGES.index("separate"))  # input changed -> everything downstream
    else:
        audio, _ = aio.read_wav(p_in)
    total_s = audio.shape[1] / SR
    log(f"input: {total_s:.1f}s  -> {p_in}")
    if done("decode"):
        return out

    # ---------------------------------------------------------- 2 separate
    p_voc, p_acc = out / "02_vocals.wav", out / "02_accompaniment.wav"
    if args.vocals:
        log(f"using supplied vocals {args.vocals}")
        vocals = aio.load_audio(args.vocals, sr=SR, channels=2)[:, :audio.shape[1]]
        vocals = np.pad(vocals, ((0, 0), (0, audio.shape[1] - vocals.shape[1])))
        acc = audio - vocals
        aio.write_wav(p_voc, vocals, SR)
        aio.write_wav(p_acc, acc, SR)
        first = min(first, STAGES.index("pitch"))
    elif redo("separate") or not p_voc.exists():
        if args.no_separation:
            log("separation skipped (--no-separation): using input as vocals")
            vocals, acc = audio, np.zeros_like(audio)
        else:
            from .separate import separate_vocals
            log(f"separate vocals with demucs '{args.sep_model}' (CPU can take a few minutes)")
            t0 = time.time()
            stems = separate_vocals(audio, SR, model_name=args.sep_model, shifts=args.shifts)
            log(f"separation done in {time.time() - t0:.0f}s")
            vocals, acc = stems["vocals"], stems["accompaniment"]
            for k, v in stems.items():
                if k not in ("vocals", "accompaniment"):
                    aio.write_wav(out / f"02_stem_{k}.wav", v, SR)
        aio.write_wav(p_voc, vocals, SR)
        aio.write_wav(p_acc, acc, SR)
        first = min(first, STAGES.index("pitch"))
    else:
        vocals, _ = aio.read_wav(p_voc)
        acc, _ = aio.read_wav(p_acc) if p_acc.exists() else (np.zeros_like(audio), SR)
    log(f"vocals -> {p_voc}")
    if done("separate"):
        return out

    # ---------------------------------------------------------- 3 pitch
    from .pitch import track_pitch, apply_voicing
    p_pitch = out / "03_pitch.npz"
    if redo("pitch") or not p_pitch.exists():
        log(f"pitch tracking (CREPE '{args.crepe_model}')")
        t0 = time.time()
        pt = track_pitch(aio.to_mono(vocals), SR, model=args.crepe_model,
                         periodicity_thresh=args.periodicity, silence_db=args.silence_db,
                         fmin=args.fmin, fmax=args.fmax)
        log(f"pitch done in {time.time() - t0:.0f}s; voiced {pt['voiced'].mean() * 100:.0f}% of frames")
        np.savez_compressed(p_pitch, **pt)
    else:
        pt = dict(np.load(p_pitch))
        apply_voicing(pt, args.periodicity, args.silence_db)  # gating params may have changed
    with open(out / "03_pitch.csv", "w") as f:
        f.write("time_s,f0_hz,f0_raw_hz,periodicity,loudness_db,voiced\n")
        for row in zip(pt["time"], pt["f0"], pt["f0_raw"], pt["periodicity"], pt["loudness_db"], pt["voiced"]):
            f.write("%.3f,%.2f,%.2f,%.3f,%.1f,%d\n" % row)
    log(f"pitch -> {p_pitch}")
    if done("pitch"):
        return out

    # ---------------------------------------------------------- 4 notes (cheap -> always run)
    log(f"syllable onsets / tuning / segmentation / tonic   (detail {args.detail})")
    onsets = nt.detect_onsets(aio.to_mono(vocals), SR, delta=tp.onset_delta)
    with open(out / "04_onsets.csv", "w") as f:
        f.write("onset_s\n" + "".join(f"{t:.3f}\n" for t in onsets))
    tr = transcribe(pt, tp, onsets=onsets if tp.split_onsets else None)
    tuning, sa_pc, sa_midi = tr.tuning, tr.sa_pc, tr.sa_midi
    pt["tuning_semitones"] = tuning
    info = tr.info()
    log(f"tuning offset {tuning * 100:+.0f} cents | key {tr.key_str} | Sa = {nt.NOTE_NAMES[sa_pc]} "
        f"(candidates: {', '.join(f'{nt.NOTE_NAMES[pc]}:{s:.2f}' for s, pc in tr.sa_rank[:3])}) | "
        f"{len(onsets)} onsets")
    if args.notes_csv:
        edited = read_notes_csv(args.notes_csv)
        variants = {"edited": edited}
        log(f"rendering {len(edited)} notes from {args.notes_csv} (transcription skipped)")
    else:
        variants = {"detailed": tr.notes, "simple": tr.simple}
        log(f"scale used: {tr.scale_str} | notes raw {tr.raw_count} -> detailed {len(tr.notes)} -> simple {len(tr.simple)}")

    tempo = estimate_tempo(acc if np.abs(acc).max() > 1e-4 else audio, SR)
    for name, nl in variants.items():
        rd.write_notes_csv(nl, out / f"04_notes_{name}.csv", sa_midi)
        rd.write_midi(nl, out / f"04_notes_{name}.mid", tempo=tempo, transpose=args.transpose)
        rd.write_notation(nl, out / f"04_notation_{name}.txt", sa_midi, info)
    for old in (out / "plots").glob("pitch_*.png"):
        old.unlink()
    main_notes = next(iter(variants.values()))
    rd.plot_pitch_pages(pt, tr.midi_corr, tr.smooth, main_notes, sa_midi, out / "plots", page_s=args.page_s,
                        simple=variants.get("simple"), onsets=onsets)
    rd.plot_histogram(tr.hist, sa_pc, tr.key_str, out / "plots" / "pitch_class_hist.png")
    log(f"notation -> {out / ('04_notation_' + next(iter(variants)) + '.txt')}")
    if done("notes"):
        return finish(out, summary, tr, total_s, tempo, pt, variants, onsets)

    # ---------------------------------------------------------- 5 render
    log(f"render instrument audio ({args.instrument})")
    from .instruments import render as render_instrument
    voc_m = aio.to_mono(vocals)
    voc_m = voc_m / (np.abs(voc_m).max() + 1e-9)
    orig = aio.to_mono(audio)
    orig = orig / (np.abs(orig).max() + 1e-9)
    for name, nl in variants.items():
        harm = render_instrument(nl, total_s, args.instrument, SR, transpose=args.transpose,
                                 tune_cents=tuning * 100)  # detuned to match the recording
        harm_m = harm.mean(axis=0)
        n = min(len(voc_m), len(harm_m))
        aio.write_wav(out / f"05_harmonium_{name}.wav", harm, SR)
        aio.write_wav(out / f"05_vocals+harmonium_{name}.wav", np.stack([voc_m[:n] * 0.8, harm_m[:n] * 0.6]), SR)  # L=voice R=instrument
        aio.write_wav(out / f"05_original+harmonium_{name}.wav", np.clip(orig[:n] * 0.7 + harm_m[:n] * 0.45, -1, 1), SR)
    return finish(out, summary, tr, total_s, tempo, pt, variants, onsets)


def finish(out, summary, tr, total_s, tempo, pt, variants, onsets):
    notes = next(iter(variants.values()))
    summary.update(duration_s=total_s, tuning_cents=tr.tuning * 100, key=tr.key_str, scale=tr.scale_str,
                   key_candidates=[(f"{nt.NOTE_NAMES[t]} {m}", round(s, 3)) for s, t, m in tr.keys[:4]],
                   sa=nt.NOTE_NAMES[tr.sa_pc], sa_midi=tr.sa_midi,
                   sa_candidates=[(nt.NOTE_NAMES[pc], round(s, 3)) for s, pc in tr.sa_rank[:4]],
                   tempo_bpm=tempo, n_onsets=len(onsets), voiced_fraction=float(np.mean(pt["voiced"])),
                   n_notes={k: len(v) for k, v in variants.items()},
                   range=[nt.midi_name(min(x.midi for x in notes)), nt.midi_name(max(x.midi for x in notes))],
                   pitch_class_hist={nt.NOTE_NAMES[i]: round(float(h), 3) for i, h in enumerate(tr.hist)})
    with open(out / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    log(f"done -> {out.resolve()}")
    return out


def main(argv=None):
    d = TranscribeParams()  # defaults shared with the GUI
    ap = argparse.ArgumentParser(description="Song -> isolated vocals -> harmonium notes",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("input", help="audio file (mp3, m4a, wav, flac, ogg, opus, ...) or a GUI selection folder")
    ap.add_argument("-o", "--out", help="output folder (default outputs/<name>)")
    ap.add_argument("--redo-from", choices=STAGES, help="recompute from this stage on (default: reuse cache)")
    ap.add_argument("--stop-after", choices=STAGES, help="stop after this stage")
    ap.add_argument("--start", type=float, default=0.0, help="clip start (s)")
    ap.add_argument("--duration", type=float, default=0.0, help="clip length (s), 0 = full")
    g = ap.add_argument_group("separation")
    g.add_argument("--sep-model", default="htdemucs_ft", help="demucs model: htdemucs_ft (best), htdemucs (faster), mdx_extra")
    g.add_argument("--shifts", type=int, default=1, help="demucs test-time shifts (higher = better/slower)")
    g.add_argument("--no-separation", action="store_true", help="input is already a clean vocal")
    g.add_argument("--vocals", help="use this isolated vocal file instead of separating (aligned with the input/clip)")
    g = ap.add_argument_group("pitch")
    g.add_argument("--crepe-model", default="full", choices=["full", "tiny"])
    g.add_argument("--fmin", type=float, default=65.0)
    g.add_argument("--fmax", type=float, default=1100.0)
    g.add_argument("--periodicity", type=float, default=0.5, help="CREPE voicing threshold (0-1)")
    g.add_argument("--silence-db", type=float, default=-40.0, help="frames quieter than this (rel. peak) = unvoiced")
    g = ap.add_argument_group("notes (same settings as the GUI)")
    g.add_argument("--detail", type=int, default=3, choices=[1, 2, 3, 4, 5],
                   help="1 coarse, 2 relaxed, 3 normal, 4 fine, 5 very fine (sets the next five unless given)")
    g.add_argument("--min-note", type=float, default=None, help="shortest note kept (s)")
    g.add_argument("--ornament", type=float, default=None, help="absorb short passing/vibrato notes below this (s); 0 = keep")
    g.add_argument("--hysteresis", type=float, default=None, help="semitones beyond +-0.5 needed to change note")
    g.add_argument("--glide-max", type=float, default=None, help="short notes with moving pitch below this (s) are glides")
    g.add_argument("--smooth", type=float, default=None, help="pitch median smoothing (s)")
    g.add_argument("--no-split-onsets", dest="split_onsets", action="store_false", help="don't split notes at syllable onsets")
    g.add_argument("--onset-delta", type=float, default=d.onset_delta, help="syllable onset sensitivity (lower = more)")
    g.add_argument("--merge-gap", type=float, default=d.merge_gap, help="join same-pitch notes across gaps shorter than this (s)")
    g.add_argument("--scale-min", type=float, default=d.scale_min, help="pitch classes with less time share are 'out of scale'")
    g.add_argument("--snap-max", type=float, default=d.snap_max, help="out-of-scale notes shorter than this (s) snap to the scale")
    g.add_argument("--no-snap", action="store_true", help="disable scale snapping")
    g.add_argument("--simple-min", type=float, default=d.simple_min, help="shortest note in the simplified version (s)")
    g.add_argument("--no-tuning", action="store_true", help="assume A440, no tuning correction")
    g.add_argument("--tonic", default="", help="force Sa, e.g. C# or Db (default: auto)")
    g.add_argument("--notes-csv", help="skip transcription: render this notes CSV (e.g. edited in the GUI / by hand)")
    g = ap.add_argument_group("output")
    g.add_argument("--instrument", default="harmonium", help="harmonium, accordion, reed_organ, shehnai, violin, flute, piano, sitar, synth")
    g.add_argument("--transpose", type=int, default=0, help="semitones to shift instrument/MIDI output")
    g.add_argument("--page-s", type=float, default=30.0, help="seconds per pitch plot page")
    run(ap.parse_args(argv))


if __name__ == "__main__":
    main()
