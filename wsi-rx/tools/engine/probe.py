"""Phase 2a environment probe.

Runs each stage of the offline Auto-DJ engine on ONE track, measures timing,
reports per-stage status + output shapes, and prints a summary table at the
end. Designed to be run repeatedly while you're installing / debugging libs
— each stage is isolated so one failure doesn't cascade.

Usage:
  python -m engine.probe                            # default: td-audio001, 30s snippet
  python -m engine.probe --full                     # full-track timing
  python -m engine.probe --track /path/to/song.wav  # any audio file
  python -m engine.probe --only librosa,clap        # only named stages
  python -m engine.probe --skip separator           # all stages except named
  python -m engine.probe --json                     # machine-readable dump
"""

from __future__ import annotations

import argparse
import importlib
import sys
import time
import traceback
from pathlib import Path

try:
    from rich.console import Console
    from rich.table import Table
    HAVE_RICH = True
except ImportError:
    HAVE_RICH = False

# On Windows the default console is cp1252 and can't print some unicode
# characters rich likes to emit (dim row dashes, box-draw, etc.). Force
# stdout/stderr to utf-8 + ask the terminal to accept it. No-op on real
# UTF-8 terminals; silent on platforms that don't need it.
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except AttributeError:
        pass  # pre-3.7 shim — not our target.

# Support being run as `python probe.py` (no package context) or
# `python -m engine.probe` (proper package). Patch sys.path if needed.
try:
    from . import config  # package-relative
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from engine import config  # type: ignore

console = Console() if HAVE_RICH else None


# ── Stage registry ────────────────────────────────────────────────────────
#
# Each stage is a function that returns a short string (the "shape" /
# sanity value) OR raises. The decorator wires up timing + status tracking
# into the STAGES list we print at the end.

STAGES: list[dict] = []


def _log(msg: str, kind: str = 'info') -> None:
    if HAVE_RICH and console:
        style = {'ok': 'green', 'skip': 'yellow', 'fail': 'red', 'info': 'cyan'}.get(kind, 'white')
        console.print(f'[{style}]{msg}[/{style}]')
    else:
        print(msg)


def stage(label: str):
    """Decorator: run fn, time it, record status. ImportError → skip, else fail."""
    def wrap(fn):
        def run(*args, **kwargs):
            prefix = f'• {label:<28}'
            if HAVE_RICH and console:
                console.print(f'[bold]{prefix}[/bold]', end=' ')
            else:
                print(prefix, end=' ', flush=True)
            t0 = time.perf_counter()
            try:
                detail = fn(*args, **kwargs) or ''
                dt = time.perf_counter() - t0
                if HAVE_RICH and console:
                    console.print(f'[green]ok[/green]   {dt:6.2f}s  [dim]{detail}[/dim]')
                else:
                    print(f'ok    {dt:6.2f}s  {detail}')
                STAGES.append({'name': label, 'status': 'ok', 'duration_s': round(dt, 3), 'detail': detail})
                return detail
            except ImportError as e:
                missing = getattr(e, 'name', None) or str(e)
                if HAVE_RICH and console:
                    console.print(f'[yellow]skip[/yellow] (not installed: {missing})')
                else:
                    print(f'skip  (not installed: {missing})')
                STAGES.append({'name': label, 'status': 'skip', 'duration_s': None, 'detail': f'not installed: {missing}'})
            except Exception as e:
                dt = time.perf_counter() - t0
                err = f'{type(e).__name__}: {e}'
                if HAVE_RICH and console:
                    console.print(f'[red]FAIL[/red] {dt:6.2f}s  {err}')
                else:
                    print(f'FAIL  {dt:6.2f}s  {err}')
                STAGES.append({'name': label, 'status': 'fail', 'duration_s': round(dt, 3), 'detail': err})
                if '--verbose' in sys.argv:
                    traceback.print_exc()
        run.label = label
        return run
    return wrap


# ── Individual probes ─────────────────────────────────────────────────────

@stage('numpy + scipy + soundfile')
def probe_numpy_stack():
    import numpy as np
    import scipy
    import soundfile as sf  # noqa: F401
    # Cheap GEMM smoke — confirms BLAS is actually wired up.
    a = np.random.rand(512, 512).astype(np.float32)
    _ = a @ a.T
    return f'numpy {np.__version__}, scipy {scipy.__version__}'


@stage('torch + torch-directml')
def probe_torch():
    import torch
    import torch_directml as tdm
    # Confirm DirectML device is reachable (can still fail at use-time; this
    # is just an optimistic probe).
    n = tdm.device_count()
    name = tdm.device_name(0) if n else 'NONE'
    return f'torch {torch.__version__}, dml devices={n} [{name}]'


@stage('librosa (load + beats + chroma)')
def probe_librosa(audio_path: Path, duration: float | None):
    import librosa
    import numpy as np
    y, sr = librosa.load(str(audio_path), duration=duration, sr=None)
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
    chroma = librosa.feature.chroma_cens(y=y, sr=sr)
    return (f'{len(y)/sr:5.1f}s@{sr}Hz, tempo={float(np.atleast_1d(tempo)[0]):.1f}, '
            f'{len(beats)} beats, chroma {chroma.shape}')


@stage('madmom (DBN beats + downbeats)')
def probe_madmom(audio_path: Path, duration: float | None):
    # madmom historically pins numpy<2; current installs with numpy 2.x may
    # need `pip install "madmom @ git+https://github.com/CPJKU/madmom.git"`
    # to pick up the post-0.16.1 patches that restore numpy-2 compat.
    from madmom.features.beats import RNNBeatProcessor, DBNBeatTrackingProcessor
    from madmom.features.downbeats import RNNDownBeatProcessor, DBNDownBeatTrackingProcessor
    # madmom operates on files, not arrays. Use the source file directly.
    # It does not natively support truncation, so a full-file run is unavoidable
    # here; --full mode doesn't change madmom's behaviour.
    src = str(audio_path)
    beats_proc = RNNBeatProcessor()
    beats_tracker = DBNBeatTrackingProcessor(fps=100)
    beats = beats_tracker(beats_proc(src))
    down_proc = RNNDownBeatProcessor()
    down_tracker = DBNDownBeatTrackingProcessor(beats_per_bar=[3, 4], fps=100)
    downs = down_tracker(down_proc(src))
    return f'{len(beats)} beats, {len(downs)} downbeats'


@stage('librosa segmentation (structure)')
def probe_structure(audio_path: Path, duration: float | None):
    # allin1 (Beat Transformer wrapper) would be ideal but blocks on this
    # stack: it depends on `natten` which needs CUDA toolchain to build +
    # a newer torch than our DirectML pin allows. msaf has its own legacy
    # scipy-API bit-rot. So we roll our own structure detector using
    # librosa primitives — still usable, fully deterministic, no deep-learning
    # deps. Milestone 2c will elaborate this with proper section labelling.
    import librosa
    import numpy as np
    y, sr = librosa.load(str(audio_path), duration=duration, sr=None)
    # Chroma-CENS is the most robust feature for structural similarity.
    chroma = librosa.feature.chroma_cens(y=y, sr=sr)
    # Enhanced recurrence matrix → path-enhanced similarity → agglomerative
    # clustering picks segment boundaries.
    rec = librosa.segment.recurrence_matrix(chroma, mode='affinity', sym=True)
    # Target ~8 sections for a typical 3-5 min track.
    target_sections = max(4, int((len(y) / sr) / 30))
    bounds_frames = librosa.segment.agglomerative(chroma, k=target_sections)
    bounds_sec = librosa.frames_to_time(bounds_frames, sr=sr)
    return f'{len(bounds_sec)} boundaries @ {bounds_sec.round(1).tolist()[:6]}...'


@stage('laion-clap (text↔audio embedding)')
def probe_clap(audio_path: Path):
    import laion_clap
    import numpy as np
    model = laion_clap.CLAP_Module(enable_fusion=False)
    model.load_ckpt()  # downloads on first run
    # Small batch: one audio file + three text prompts. Embedding space is shared.
    audio_emb = model.get_audio_embedding_from_filelist([str(audio_path)], use_tensor=False)
    text_emb = model.get_text_embedding(
        ['dark bass drop', 'euphoric lead synth', 'tension build-up'],
        use_tensor=False,
    )
    sim = (np.asarray(audio_emb) @ np.asarray(text_emb).T)[0]
    labels = 'drop/lead/build'
    return f'audio {np.asarray(audio_emb).shape}, text {np.asarray(text_emb).shape}, sim[{labels}]={sim.round(2).tolist()}'


@stage('librosa key (Krumhansl-Kessler)')
def probe_key(audio_path: Path, duration: float | None):
    # Essentia's KeyExtractor + EDMA profile would give stronger electronic-
    # music-tuned key detection, but essentia has no Windows wheel + fails
    # to build from source on this setup (setup.py bug, Linux-only build
    # artefacts). Fall back to librosa chroma + Krumhansl-Kessler profiles
    # (same algorithm used in the v1 analyze-beats.py). Milestone 2c will
    # consider running Essentia via WSL or switching to a key-detection
    # model via tensorflow/onnx if better accuracy is needed.
    import librosa
    import numpy as np
    KK_MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
    KK_MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
    NAMES = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']
    y, sr = librosa.load(str(audio_path), duration=duration, sr=None)
    chroma = librosa.feature.chroma_cens(y=y, sr=sr).mean(axis=1)
    best = (-1, -1, -1)  # (score, key_idx, 0=maj,1=min)
    for i in range(12):
        rot_maj = np.roll(KK_MAJOR, i)
        rot_min = np.roll(KK_MINOR, i)
        score_maj = float(np.corrcoef(chroma, rot_maj)[0, 1])
        score_min = float(np.corrcoef(chroma, rot_min)[0, 1])
        if score_maj > best[0]: best = (score_maj, i, 0)
        if score_min > best[0]: best = (score_min, i, 1)
    mode = 'maj' if best[2] == 0 else 'min'
    return f'key={NAMES[best[1]]}{mode}  correlation={best[0]:.3f}'


@stage('audio-separator (stem sep via DirectML)')
def probe_separator(audio_path: Path):
    # Heaviest stage by far — skip by default unless --include separator.
    # When run, it must actually allocate GPU memory via DirectML.
    import audio_separator  # noqa: F401
    # We don't run a full separation here; just confirm the package imports
    # and has a functional interface. Full timing lives in engine/stems.py.
    return 'imports ok (run engine/stems.py for actual sep timing)'


@stage('ollama client → gemma4:e4b')
def probe_ollama():
    import ollama
    client = ollama.Client(host=config.OLLAMA_HOST)
    resp = client.generate(model=config.OLLAMA_MODEL, prompt='Reply with exactly one word: working.', options={'num_predict': 5})
    txt = (resp.get('response') or '').strip()[:40]
    return f'[{config.OLLAMA_MODEL}] → {txt!r}'


@stage('mir_eval (metrics lib)')
def probe_mir_eval():
    import mir_eval
    return f'mir_eval {getattr(mir_eval, "__version__", "?")}'


# ── Runner ───────────────────────────────────────────────────────────────

ALL_STAGES = {
    'numpy':      (probe_numpy_stack, []),
    'torch':      (probe_torch, []),
    'librosa':    (probe_librosa, ['track', 'duration']),
    'madmom':     (probe_madmom, ['track', 'duration']),
    'structure':  (probe_structure, ['track', 'duration']),
    'key':        (probe_key, ['track', 'duration']),
    'clap':       (probe_clap, ['track']),
    'separator':  (probe_separator, ['track']),
    'ollama':     (probe_ollama, []),
    'mir_eval':   (probe_mir_eval, []),
}
DEFAULT_STAGES = ['numpy', 'torch', 'librosa', 'madmom', 'structure', 'key', 'clap', 'ollama', 'mir_eval']
# `separator` omitted from default — heavy. Request explicitly via --only or --include.


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Phase 2a Auto-DJ engine env probe.')
    p.add_argument('--track', type=Path, default=None, help='Path to test audio file. Default: td-audio001.')
    p.add_argument('--full', action='store_true', help='Run on the full track (default: 30s snippet).')
    p.add_argument('--only', type=str, default=None, help='Comma-separated stage list (overrides default set).')
    p.add_argument('--skip', type=str, default=None, help='Comma-separated stages to skip from default set.')
    p.add_argument('--include', type=str, default=None, help='Comma-separated stages to ADD to default set (e.g. separator).')
    p.add_argument('--json', action='store_true', help='Emit machine-readable JSON dump at end.')
    p.add_argument('--verbose', action='store_true', help='Print full tracebacks for failures.')
    return p.parse_args()


def resolve_stages(args: argparse.Namespace) -> list[str]:
    if args.only:
        return [s.strip() for s in args.only.split(',') if s.strip()]
    stages = list(DEFAULT_STAGES)
    if args.include:
        for s in (x.strip() for x in args.include.split(',')):
            if s and s not in stages:
                stages.append(s)
    if args.skip:
        skip = {x.strip() for x in args.skip.split(',')}
        stages = [s for s in stages if s not in skip]
    # Validate
    for s in stages:
        if s not in ALL_STAGES:
            raise SystemExit(f'Unknown stage: {s!r}. Valid: {sorted(ALL_STAGES)}')
    return stages


def main() -> int:
    args = parse_args()
    track = args.track or config.default_track()
    duration = None if args.full else 30.0
    stages = resolve_stages(args)

    _log('-' * 72)
    _log(f'  WSI RX · Phase 2a env probe', kind='info')
    _log(f'  track:    {track}')
    _log(f'  mode:     {"full-track" if args.full else "30s snippet"}')
    _log(f'  stages:   {", ".join(stages)}')
    _log('-' * 72)

    for name in stages:
        fn, needs = ALL_STAGES[name]
        kwargs = {}
        if 'track' in needs:
            kwargs['audio_path'] = track
        if 'duration' in needs:
            kwargs['duration'] = duration
        fn(**kwargs)

    # ── Summary ──
    _log('\n' + '-' * 72)
    if HAVE_RICH and console:
        tbl = Table(title='summary', show_lines=False)
        tbl.add_column('stage'); tbl.add_column('status'); tbl.add_column('time'); tbl.add_column('detail', overflow='fold')
        for s in STAGES:
            status = s['status']
            color = {'ok': 'green', 'skip': 'yellow', 'fail': 'red'}.get(status, 'white')
            dur = '—' if s['duration_s'] is None else f'{s["duration_s"]:.2f}s'
            tbl.add_row(s['name'], f'[{color}]{status}[/{color}]', dur, s['detail'] or '')
        console.print(tbl)
    else:
        print(f'{"stage":<32} {"status":<6} {"time":<8} detail')
        for s in STAGES:
            dur = '—' if s['duration_s'] is None else f'{s["duration_s"]:.2f}s'
            print(f'{s["name"]:<32} {s["status"]:<6} {dur:<8} {s["detail"]}')

    # Totals
    ok  = sum(1 for s in STAGES if s['status'] == 'ok')
    skip = sum(1 for s in STAGES if s['status'] == 'skip')
    fail = sum(1 for s in STAGES if s['status'] == 'fail')
    total_time = sum(s['duration_s'] or 0 for s in STAGES if s['status'] == 'ok')
    _log(f'\n  {ok} ok  ·  {skip} skip  ·  {fail} fail  ·  total time: {total_time:.2f}s', kind='info')

    if args.json:
        import json
        print('\n' + json.dumps({'track': str(track), 'full': args.full, 'stages': STAGES}, indent=2))

    return 0 if fail == 0 else 1


if __name__ == '__main__':
    raise SystemExit(main())
