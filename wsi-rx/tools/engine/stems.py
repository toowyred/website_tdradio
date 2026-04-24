"""Stem-separation orchestrator.

Uses audio_separator (UVR5's headless engine) with `htdemucs_ft.yaml` to
produce 4 stems per track: drums, bass, vocals, other. Output layout:

    <repo>/wsi-rx/tracks/stems/<track-basename>/{drums,bass,vocals,other}.wav

Design notes:
- Loads the Separator ONCE per run, processes N tracks against it. Amortizes
  the ~5-10 s model-load cost across the batch.
- DirectML-backed (use_directml=True) — validated path on GLASSBOX per
  memory + wsi_separator_worker.py which uses the same invocation.
- Resume-safe: `--all` skips any track whose output folder already has all
  4 stems. `--force` re-runs everything.
- No subprocess IPC (unlike wsi_separator_worker.py) — this is a standalone
  CLI, we don't need the STATUS-line protocol the WSI-OG GUI used.

Usage:
  python -m engine.stems --track <path-to-audio>
  python -m engine.stems --all
  python -m engine.stems --all --force
"""

from __future__ import annotations

import argparse
import gc
import logging
import os
import shutil
import sys
import time
from pathlib import Path

try:
    from . import config
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from engine import config  # type: ignore


# The Demucs v4 fine-tuned model gives us all 4 stems in one pass. A further
# ensemble (MDX-Net + Demucs) yields slightly cleaner vocals but doubles the
# runtime; Phase 2b runs Demucs-only. The ensemble path lives in
# wsi_separator_worker.py if we ever need it later.
DEMUCS_MODEL = 'htdemucs_ft.yaml'
STEMS = ('drums', 'bass', 'vocals', 'other')

# Portable model cache — audio-separator auto-downloads missing models here.
# Match the env var convention wsi-build0406.py established.
MODEL_DIR = os.environ.get(
    'AUDIO_SEPARATOR_MODEL_DIR',
    str(Path.home() / 'audio-separator-models'),
)

# Extensions considered source audio when walking V1_TRACKS_DIR.
AUDIO_EXTS = ('.wav', '.mp3', '.flac', '.m4a')


def _stem_dir_for(track_path: Path) -> Path:
    return config.TRACKS_DIR / 'stems' / track_path.stem


def _is_complete(stem_dir: Path) -> bool:
    """All 4 canonical stem files present with non-zero size."""
    if not stem_dir.is_dir():
        return False
    present = {p.stem.lower(): p for p in stem_dir.glob('*.wav')}
    return all(s in present and present[s].stat().st_size > 0 for s in STEMS)


def _normalize_stem_name(produced_filename: str) -> str | None:
    """audio-separator names stems like 'song_(drums)_htdemucs_ft.wav'. Map
    these back to the canonical {drums,bass,vocals,other}. Return None if the
    produced file doesn't match any of the 4 stems we care about."""
    low = produced_filename.lower()
    for s in STEMS:
        if s in low:
            return s
    return None


def _load_separator(output_dir: Path):
    """Initialise the audio-separator Separator with the Demucs model.
    `output_dir` MUST be passed here — the instance attribute is fixed at
    construction time; reassigning `sep.output_dir` later is ignored and
    files silently land in CWD (learned the hard way on the first smoke
    test). We use a shared scratch folder for the whole batch and then
    distribute produced stems to per-track folders in post-processing."""
    from audio_separator.separator import Separator  # local import: optional dep
    os.makedirs(MODEL_DIR, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    sep = Separator(
        model_file_dir=MODEL_DIR,
        output_dir=str(output_dir),
        output_format='WAV',
        use_directml=True,
        log_level=logging.WARNING,
    )
    sep.load_model(DEMUCS_MODEL)
    return sep


def _source_stem_of(produced_filename: str, track_stem: str) -> str | None:
    """Filenames audio-separator emits look like
        `td-audio001-normal_(Drums)_htdemucs_ft.wav`.
    We verify the filename belongs to the current track (the basename-prefix
    match) + pull out the stem kind. Returns the canonical stem name
    (`drums`/`bass`/`vocals`/`other`) or None if it isn't one of ours."""
    low = produced_filename.lower()
    if not low.startswith(track_stem.lower()):
        return None
    return _normalize_stem_name(low)


def separate_track(sep, scratch_dir: Path, track_path: Path, force: bool = False) -> Path | None:
    """Run the separator on one track, then distribute + rename the produced
    files into `wsi-rx/tracks/stems/<track>/{drums,bass,vocals,other}.wav`.
    Returns the per-track folder on success, or None if skipped."""
    out_dir = _stem_dir_for(track_path)
    if _is_complete(out_dir) and not force:
        print(f'  skip  {track_path.stem:<28} (4 stems already present)')
        return None
    out_dir.mkdir(parents=True, exist_ok=True)

    # Snapshot the scratch-dir state before/after so we can reliably grab
    # this run's outputs even if sep.separate() returns relative paths or
    # an incomplete filename list.
    before = {p.name for p in scratch_dir.glob('*.wav')}
    t0 = time.perf_counter()
    sep.separate(str(track_path))
    dt = time.perf_counter() - t0
    after = {p.name for p in scratch_dir.glob('*.wav')}
    new_files = sorted(after - before)

    moved = {}
    for name in new_files:
        kind = _source_stem_of(name, track_path.stem)
        if kind is None:
            # Not one of our 4 stems — leave orphan file alone; it'll get
            # picked up by the next batch's before-set diff if relevant.
            continue
        src = scratch_dir / name
        dest = out_dir / f'{kind}.wav'
        if dest.exists():
            dest.unlink()
        shutil.move(str(src), str(dest))
        moved[kind] = dest

    # Report + sanity
    if not _is_complete(out_dir):
        missing = [s for s in STEMS if not (out_dir / f'{s}.wav').exists()]
        print(f'  WARN  {track_path.stem:<28} {dt:6.1f}s  missing: {missing}  (scratch: {scratch_dir})')
    else:
        total_mb = sum((out_dir / f'{s}.wav').stat().st_size for s in STEMS) / (1024 * 1024)
        print(f'  ok    {track_path.stem:<28} {dt:6.1f}s  {total_mb:6.1f} MB  →  {out_dir}')
    return out_dir


def list_all_tracks() -> list[Path]:
    """Discover every track under V1_TRACKS_DIR, deduped by logical name.
    When both td-audio001-high.wav and td-audio001-normal.mp3 exist we pick
    the .wav — stem separation quality tracks source fidelity."""
    if not config.V1_TRACKS_DIR.is_dir():
        return []
    found: list[Path] = []
    for ext in AUDIO_EXTS:
        found.extend(config.V1_TRACKS_DIR.glob(f'*{ext}'))
    # Normalise the name: td-audio001-high → td-audio001
    def _canonical_stem(p: Path) -> str:
        s = p.stem
        for suffix in ('-high', '-normal', '-off'):
            if s.endswith(suffix):
                s = s[: -len(suffix)]
                break
        return s

    best: dict[str, Path] = {}
    rank = {'.wav': 3, '.flac': 2, '.m4a': 1, '.mp3': 0}
    for p in found:
        canon = _canonical_stem(p)
        if canon not in best or rank.get(p.suffix.lower(), -1) > rank.get(best[canon].suffix.lower(), -1):
            best[canon] = p
    return [best[k] for k in sorted(best)]


def main() -> int:
    p = argparse.ArgumentParser(description='Stem separation orchestrator (Phase 2b).')
    p.add_argument('--track', type=Path, default=None, help='Separate one audio file.')
    p.add_argument('--all', action='store_true', help='Separate every track under V1_TRACKS_DIR.')
    p.add_argument('--force', action='store_true', help='Re-run even when stems exist.')
    args = p.parse_args()

    if not args.track and not args.all:
        p.error('pass --track <path> or --all')

    tracks = [args.track] if args.track else list_all_tracks()
    if not tracks:
        print('No tracks found. Check config.V1_TRACKS_DIR.')
        return 1

    out_base = config.TRACKS_DIR / 'stems'
    print('-' * 72)
    print(f'  stems        · output:  {out_base}/<name>/{{{",".join(STEMS)}}}.wav')
    print(f'  model        · {DEMUCS_MODEL}  (dir: {MODEL_DIR})')
    print(f'  tracks       · {len(tracks)} to process  {"[FORCE re-run]" if args.force else ""}')
    print('-' * 72)

    # Shared scratch dir for the whole batch. Each track's raw outputs land
    # here + get distributed to per-track folders in separate_track().
    scratch_dir = out_base / '_scratch'
    scratch_dir.mkdir(parents=True, exist_ok=True)

    print('  loading Separator...')
    load_t0 = time.perf_counter()
    sep = _load_separator(scratch_dir)
    print(f'  loaded in {time.perf_counter() - load_t0:.1f}s   scratch: {scratch_dir}')
    print()

    batch_t0 = time.perf_counter()
    ok = skip = fail = 0
    for track in tracks:
        try:
            result = separate_track(sep, scratch_dir, track, force=args.force)
            if result is None:
                skip += 1
            else:
                ok += 1
        except Exception as e:
            fail += 1
            print(f'  FAIL  {track.stem:<28} {type(e).__name__}: {e}')
        gc.collect()

    # Clean up any leftover scratch files (e.g. when a track failed mid-run
    # and the partial outputs weren't distributed).
    try:
        for p in scratch_dir.glob('*.wav'):
            p.unlink()
        scratch_dir.rmdir()
    except OSError:
        pass  # not empty = user is mid-run in another shell, leave alone

    total = time.perf_counter() - batch_t0
    print('-' * 72)
    print(f'  done   {ok} ok · {skip} skip · {fail} fail · {total:.1f}s total')
    return 0 if fail == 0 else 1


if __name__ == '__main__':
    raise SystemExit(main())
