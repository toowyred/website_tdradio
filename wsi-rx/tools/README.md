# wsi-rx/tools — offline Auto-DJ engine

Phase 2 of the v2 rebuild. Everything in this folder runs **locally on
GLASSBOX** — not on the web. Nothing here is deployed anywhere. The engine
authors `r/<trk-xxxxxxx>/dj.dat` mix-blueprint files; the player reads
those at runtime.

**Hard boundary**: this folder does not import from, read, or write to
`webiste_TD/radio/`. v1.5 continues receiving its own maintenance patches
in parallel; the v2 engine stays isolated.

## One-time setup

```powershell
# From wsi-rx/tools/
pip install -r requirements.txt
```

**Don't touch torch / torchvision / torch-directml.** They're pinned at
`2.4.1 / 0.19.1 / 0.2.5.dev240914` system-wide so `audio-separator` works on
DirectML. Installing the deps above should leave them alone; if pip tries
to upgrade torch, abort and install the offending lib with
`--no-deps`.

See `requirements.txt` for per-library install hazards (madmom ↔ numpy 2.x,
allin1 weight download, CLAP checkpoint, etc.).

## Validating the env

```powershell
python -m engine.probe
```

Runs every stage on the first 30 s of `td-audio001` and prints a summary
table. Every `ok` means that stage is wired up end-to-end. `skip` means
the library isn't installed yet. `fail` means the library is installed
but blew up at runtime.

Common flag combos:

```powershell
python -m engine.probe --full                       # run on the full track
python -m engine.probe --only librosa,madmom        # isolate one or two stages
python -m engine.probe --include separator          # also run the heavy stem-sep probe
python -m engine.probe --track "<path to wav>"      # different source file
python -m engine.probe --json > probe.json          # machine-readable log
python -m engine.probe --verbose                    # full traceback on failures
```

## Engine pipeline (target shape — most stages not built yet)

```
 audio file
     │
     ├─► stems.py        → tracks/stems/<trk-*>/{drums,bass,vocals,other}.wav
     │                     (UVR5 / wsi-build0406.py shell — DirectML via audio-separator)
     │
     ├─► structure.py    → tracks/cf-structure.dat
     │                     (madmom beats + downbeats, allin1 sections, Essentia key,
     │                      per-stem drop / pre-drop / fake-drop detection)
     │
     ├─► embeddings.py   → tracks/cf-embeddings.npz
     │                     (CLAP per-section, Essentia-TF mood / danceability / valence)
     │
     ├─► judge.py        → R2 upload: r/<trk-*>/dj.dat
     │                     (Gemma 4:26b via Ollama — reads structure + embeddings,
     │                      emits A/B/C mix blueprints in the locked dj.dat schema)
     │
     └─► audition.py     → tracks/cf-dj-feedback.dat
                           (Tkinter GUI: "✓ slaps / ~ cooking / ✗ this ain't it" per
                            option. Feedback loops back into judge.py prompts.)
```

Each milestone lands as its own script under `engine/`. Only `probe.py`
and `config.py` exist today.

## File layout

```
wsi-rx/
├── tools/
│   ├── README.md                ← this file
│   ├── requirements.txt
│   ├── uploader.py              ← (pending) copied from v1.5 + track-ID updates
│   └── engine/
│       ├── __init__.py
│       ├── config.py            ← paths + constants (edit if your layout differs)
│       ├── probe.py             ← Phase 2a env validator (this is it)
│       ├── stems.py             ← 2b (pending)
│       ├── structure.py         ← 2c / 2d (pending)
│       ├── embeddings.py        ← 2e / 2f (pending)
│       ├── judge.py             ← 2g (pending)
│       └── audition.py          ← 2h (pending)
└── tracks/                      ← engine output lives here
    ├── id-map.dat               ← legacy td-audio00N ↔ opaque trk-* aliases (pending)
    ├── cf-structure.dat
    ├── cf-embeddings.npz
    └── cf-dj-feedback.dat
```

## Hardware notes (from memory)

- **CPU**: Ryzen 9 7950X, 16 cores / 32 threads — librosa + madmom + allin1 run comfortably here.
- **GPU**: RX 7900 XTX 24 GB via DirectML. No CUDA, no ROCm. `audio-separator` is the proven DirectML workload; other torch-based models may work via torch-directml but benchmark-to-confirm (windowed attention is broken on DML; CNN-class ops are fine).
- **VRAM budget**: Ollama's `gemma4:26b` uses 18 GB. Stop it before launching ComfyUI or the stem-separator if the card is saturated.
- **Ollama**: pre-installed, bound to `0.0.0.0:11434`. Models: `gemma4:26b` (heavy reasoning) and `gemma4:e4b` (lightweight, always-hot).
