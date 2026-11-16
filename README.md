# audio2notes

Turn a song recording into **harmonium / sargam notation** for its vocal melody.

audio2notes isolates the singer's voice from the music, tracks the sung pitch, and transcribes
it into notes. You get Indian sargam (S r R g G m M P d D n N) next to Western key names,
plus MIDI and an instrument rendering you can play along with the original. A local web GUI
lets you pick a section of the song, see the vocal pitch with the notes drawn over it, and fix
the transcription note by note or passage by passage.

```
song.mp3 ──► isolate vocals ──► track pitch ──► notes ──► sargam / MIDI / instrument audio
             (Demucs)           (CREPE)         tuning, Sa, syllables, ornaments
```

## Features

- **Any common format in:** mp3, m4a/aac, flac, ogg, opus, wav, wma, even video files (via ffmpeg).
- **Vocal isolation** with Demucs (`htdemucs_ft`), which strips out instruments, drums and noise.
- **Pitch tracking** with CREPE, with voicing and loudness gating tuned for separated vocals.
- **Transcription built for Indian vocal music:**
  - corrects for singers who aren't tuned to A440
  - estimates the tonic (Sa) and the scale actually used
  - splits notes at syllable onsets
  - treats meend (glides) and kan/vibrato as ornaments rather than separate notes
  - snaps short out-of-scale glide notes onto the scale
  - offers a *detailed* version (what was sung) and a *simple* version (main held notes, what an accompanist would play)
- **Detail control:** five levels from *Coarse* to *Very fine*, applied to the whole selection or just one passage.
- **Note editor (GUI):** split at the playhead (both halves re-fit to the sung pitch), merge, delete,
  shift, fit to vocals, move edges, add notes, re-transcribe a range, undo/redo.
- **Natural-sounding playback:** real instrument samples (harmonium-like free reeds, shehnai, violin,
  flute, piano, sitar …), detuned to the singer's tuning so they blend with the recording.
- **Inspectable pipeline:** every stage writes its output, and every stage can be the starting point (CLI).

## Requirements

- **Python 3.13.** Python 3.14 is too new for parts of the audio stack.
- **[ffmpeg](https://ffmpeg.org/download.html)** on your `PATH`.
- About 2 GB of disk for Python packages and models. No GPU needed; a CUDA GPU makes separation much faster.

## Installation

```bash
git clone https://github.com/Sourav-Kulkarni/audio2notes.git
cd audio2notes
py -3.13 -m venv .venv                     # macOS/Linux: python3.13 -m venv .venv
.venv\Scripts\activate                     # macOS/Linux: source .venv/bin/activate

pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu   # or the CUDA build
pip install -r requirements.txt
python scripts/download_assets.py          # instrument samples (31 MB)
```

Demucs and CREPE model weights download automatically the first time they're needed.

## GUI

```bash
python app.py            # Windows: double-click start_gui.bat
```
This opens <http://127.0.0.1:7860>.

1. **Choose a file.**
2. **Select a part** by dragging the region handles on the waveform, then click **Extract vocals**.
   Every later step works only on this selection. Expect roughly 1–2× the selection length on a modern CPU.
3. **Inspect the vocal pitch.** You can play the original, the vocals or the music alone, and the playhead follows on the plot.
   Pick a **Detail** level, then click **Extract notation**. Notes are overlaid with sargam labels and Sa marked;
   hover a note for its timing and how far (in cents) the singer was from it.
4. **Refine.**
   - **Single notes:** click a note to select it; split it at the playhead (`S`), merge (`M`), delete (`Del`), move it up or down (`↑`/`↓`),
     fit it to the vocals (`F`), set its start or end to the playhead (`[`/`]`), or add a note (`N`).
   - **Passages:** switch to *Select range*, drag across a passage and **Re-transcribe** it (`R`) at its own detail level.
   - **Apply:** click **Render edits** to update the instrument audio, notation text and downloads (MIDI / CSV / TXT).

## Command line

The command line uses the same engine and defaults as the GUI.

```bash
python run.py song.mp3                                   # full song, all stages
python run.py song.mp3 --start 30 --duration 45          # just a clip
python run.py song.mp3 --stop-after separate             # only decode + isolate vocals
python run.py song.mp3 --redo-from notes --detail 4 --tonic A    # re-notate from cached pitch
python run.py song.mp3 --vocals my_vocals.wav            # bring your own isolated vocals
python run.py vocal_only.wav --no-separation             # input is already a clean vocal
python run.py outputs/gui/<session>/sel_<a>_<b>_<model>  # continue from a GUI selection
python run.py <same input> --notes-csv edited.csv --instrument violin   # render an edited note list
python run.py -h                                         # all options
```

`--notes-csv` accepts the CSVs written by the CLI or downloaded from the GUI. Times are relative
to the clip or selection, and you can edit the file in any spreadsheet; it needs `start_s,end_s,midi`
or a `western` column such as `F#4`.

### Outputs (`outputs/<song>/`)

| stage | method | files |
|---|---|---|
| 1 decode | ffmpeg → 44.1 kHz stereo | `01_input.wav` |
| 2 separate | Demucs `htdemucs_ft` | `02_vocals.wav`, `02_accompaniment.wav`, `02_stem_*.wav` |
| 3 pitch | CREPE (full) + Viterbi, voicing gating | `03_pitch.csv`, `03_pitch.npz` |
| 4 notes | onsets, tuning, segmentation, ornament/glide handling, Sa + scale, simplification | `04_onsets.csv`, `04_notation_{detailed,simple}.txt`, `04_notes_*.{csv,mid}`, `plots/*.png` |
| 5 render | SoundFont instrument samples, tuned to the singer | `05_harmonium_*.wav`, `05_vocals+harmonium_*.wav` (L voice / R instrument), `05_original+harmonium_*.wav` |

`summary.json` records the detected tuning, the Sa and key candidates, the tempo and every parameter used.

Example notation (`04_notation_simple.txt`):
```
Tonic (Sa) = A   (madhya Sa = harmonium key A3)
Scale (notes used): S R g G m P D n
[00:04.11 - 00:12.16]
  sargam : m   D    n   R' - - -  S'  D    G    D -    S' - - - -  D
  keys   : D4  F#4  G4  B4 - - -  A4  F#4  C#4  F#4 -  A4 - - - -  F#4
```
Lowercase letters are komal and `M` is tivra Ma. `N.` means the lower octave, `S'` the upper octave, and `-` a held note.

### Detail levels (`--detail` / GUI slider)

| level | shortest note | ornament absorb | hysteresis | glide filter | smoothing |
|---|---|---|---|---|---|
| 1 coarse | 0.25 s | 0.30 s | 0.35 st | 0.30 s | 0.09 s |
| 2 relaxed | 0.14 | 0.20 | 0.28 | 0.25 | 0.07 |
| 3 normal | 0.08 | 0.15 | 0.20 | 0.20 | 0.05 |
| 4 fine | 0.05 | 0.07 | 0.12 | 0.10 | 0.03 |
| 5 very fine | 0.03 | 0 | 0.06 | 0 | 0.02 |

Explicit flags (`--min-note`, `--ornament`, `--hysteresis`, `--glide-max`, `--smooth`) override the preset.

### What to tweak when

| symptom | try |
|---|---|
| instruments leak into the vocals | `--sep-model htdemucs` or `mdx_extra`, `--shifts 2` |
| notes missing in quiet/breathy passages | lower `--periodicity` (0.35) or `--silence-db` (-50) |
| spurious notes in silences | raise `--periodicity` (0.6–0.7) |
| ornaments missing / too busy | `--detail 4`–`5` / `--detail 1`–`2` |
| held notes chopped into repeats | raise `--onset-delta` (0.2–0.3) or `--no-split-onsets` |
| repeated syllables merged into one note | lower `--onset-delta` (0.06–0.08) |
| a real accidental got snapped away | `--no-snap` or lower `--scale-min` |
| wrong Sa | `--tonic <note>` (see the candidates printed / `plots/pitch_class_hist.png`) |

## Project layout

```
app.py                  Flask server for the GUI (background jobs, progress polling)
run.py                  CLI entry point -> audio2notes/pipeline.py
audio2notes/
  audio_io.py           ffmpeg decoding, wav I/O
  separate.py           Demucs vocal isolation (with progress callback)
  pitch.py              CREPE pitch tracking + voicing gate
  notes.py              segmentation, onsets, tuning, Sa/key, scale snapping, simplification
  transcribe.py         shared transcription entry point, defaults and detail presets
  engine.py             GUI session logic (extract / notate / refine / render edits)
  pipeline.py           CLI stages with cached intermediate files
  render.py             MIDI, CSV, notation text, diagnostic plots
  sf2synth.py           small pure-numpy SoundFont sample player
  instruments.py        instrument presets
web/index.html          the GUI (wavesurfer.js + Plotly, vendored in web/vendor/)
scripts/download_assets.py   fetches the instrument sample bank
tests/make_synthetic.py      synthetic test song with a known melody
```

## Limitations

- Sa (tonic) detection is a heuristic, and short selections can be ambiguous. Check the candidates and override if needed.
- Separation quality decides everything downstream. Dense mixes, chorus vocals or heavy reverb make it harder.
- Notation is pitch and timing only. There's no rhythm quantization to taal or beats yet.
- Only the melody is transcribed; there are no chords or drone.

## Credits

- [Demucs](https://github.com/facebookresearch/demucs) (Meta AI, MIT) for source separation
- [CREPE](https://github.com/marl/crepe) via [torchcrepe](https://github.com/maxrmorrison/torchcrepe) (MIT) for pitch tracking
- [librosa](https://librosa.org/) (ISC), [pretty_midi](https://github.com/craffel/pretty-midi) (MIT)
- [GeneralUser GS](https://schristiancollins.com/generaluser.php) by S. Christian Collins, the instrument samples
- [wavesurfer.js](https://wavesurfer.xyz/) (BSD-3) and [Plotly.js](https://plotly.com/javascript/) (MIT), vendored in `web/vendor/`

Only use recordings you have the rights to. Don't commit copyrighted audio to this repo (`.gitignore` excludes audio files and outputs).

## License

MIT, see [LICENSE](LICENSE). Bundled/downloaded third-party components keep their own licenses (see Credits).
