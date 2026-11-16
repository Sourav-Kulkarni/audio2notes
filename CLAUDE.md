# CLAUDE.md

Guidance for Claude Code when working in this repository.

## What this is
Song → Demucs vocal isolation → CREPE pitch → note transcription (sargam + Western) → MIDI/notation +
SoundFont instrument rendering. Two front ends over one engine: a Flask + browser GUI (`app.py`,
`web/index.html`) and a staged CLI (`run.py` → `audio2notes/pipeline.py`).

## Commands
```
.venv\Scripts\python app.py [--no-browser] [--port 7860]     # GUI at http://127.0.0.1:7860
.venv\Scripts\python run.py <audio|gui-selection-dir> [opts]   # CLI, see -h
.venv\Scripts\python scripts\download_assets.py              # instrument SoundFont -> third_party/
.venv\Scripts\python tests\make_synthetic.py test_audio\synthetic.wav   # known-melody test song
```
- Python **3.13** venv in `.venv` (3.14 breaks parts of the audio stack). CPU-only torch is fine.
- ffmpeg must be on PATH (all decoding goes through it).
- The GUI server caches Python modules: **restart `app.py` after changing backend code**; for
  `web/index.html` a browser refresh is enough. Sessions live in memory, so a restart means
  reloading the file and re-extracting.
- Quick JS syntax check: extract the `<script>` block of `web/index.html` and run `node --check`.

## Architecture (where things live)
- `transcribe.py` is the **single source of truth** for transcription: `TranscribeParams` defaults,
  `DETAIL_PRESETS` / `DETAIL_LEVELS`, and `transcribe()`. The GUI fetches presets and defaults from
  `/api/config`; the CLI builds its argparse defaults from `TranscribeParams`. Add new knobs here first,
  then expose them in both `pipeline.py` (argparse) and `web/index.html` (`p-<name>` input).
- `notes.py`: segmentation (`segment_notes`), glide/ornament absorption, onset detection and
  splitting, tuning estimate, Sa/key heuristics, `snap_to_scale`, `simplify`. `Note.onset=True` means
  "starts at a syllable": merging helpers must never merge such a note into the previous one.
- `engine.py` (`Session`): GUI logic, i.e. extract (cached per selection in `outputs/gui/<sid>/sel_*`),
  notate, refine (re-transcribe a sub-range with fixed tuning/Sa/scale), render_edited.
- `pipeline.py`: CLI stages decode → separate → pitch → notes → render, cached as numbered files;
  supports `--redo-from`, `--stop-after`, `--vocals`, `--no-separation`, `--notes-csv`, and a GUI
  selection folder as input.
- `sf2synth.py`: own minimal SF2 parser/player (pure numpy). Running external synth binaries
  (FluidSynth) was deliberately avoided. `instruments.py` maps names → SoundFont layers and falls back
  to the additive synth in `render.py` if the SoundFont is missing.
- `separate.py` / `pitch.py` take `progress` callbacks (Demucs `callback`, an unrolled torchcrepe loop)
  that feed the GUI progress bars.

## Conventions and gotchas
- **Pitch units:** `Note.midi` / `midi_float` are *tuning-corrected* (A440 grid). Sung pitch =
  corrected + `tuning` (semitones). GUI plots draw in sung pitch, so notes are drawn at `midi + tuning`.
  Instrument renders are detuned by `tuning*100` cents so they blend with the recording; written
  notation and MIDI stay on the A440 grid.
- **Times:** pipeline/CSV/MIDI times are relative to the clip/selection; GUI JSON note times are
  song-absolute (`engine._note_json` adds `sel.start`).
- Keep GUI and CLI output **identical** for the same selection and settings. To verify, run the CLI
  on a GUI selection folder and compare `04_notes_*.csv` with the engine's `notate()` notes.
- JSON responses must contain plain Python types (cast numpy floats/ints).
- Never commit audio or `outputs/` (copyright): `.gitignore` covers it. `third_party/` is downloaded.
- Validate transcription changes on real vocals, not only the synthetic song: Demucs treats the
  synthetic voice unreliably.
