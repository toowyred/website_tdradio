"""Structural + rhythmic pass.

For each track:
  1. madmom RNN + DBN → beats (ms) + downbeats (ms + beats-per-bar = 4 inferred).
  2. BPM segments — tempo is computed per-beat as 60000/interval; we cluster
     adjacent beats whose instantaneous tempo stays within ±2 BPM and emit
     one segment per stable region.
  3. Section boundaries via librosa.segment.agglomerative on chroma-CENS
     recurrence, snapped to the nearest downbeat so sections always start on
     the "1" of a bar.
  4. Per-section key via Krumhansl-Kessler profile correlation on the
     section's mean chroma. Output as Camelot codes (mirrors the live
     player's KEY readout format — see wsi_musickey.py in v1.5 for the
     canonical table).

Output file: wsi-rx/tracks/cf-structure.dat
Format per the approved plan:
    [track-stem]
    bpm_segments  = 124.5@0ms, 128.0@42000ms
    downbeats_ms  = 124, 2050, 3976, ...
    sections      = section_0:0-12000, section_1:12000-36000, ...
    keys          = 10A@0, 11A@42000

Section LABELS (intro / verse / chorus / drop / ...) aren't emitted here
because allin1 is blocked on this stack (needs torch >=2.5). The Gemma
judge in milestone 2g will infer labels from energy + section length +
CLAP similarity patterns. For now sections are anonymous (`section_N`).

Usage:
  python -m engine.structure --track /path/to/audio.(wav|mp3)
  python -m engine.structure --all
  python -m engine.structure --all --force          # re-analyse every track
  python -m engine.structure --all --validate       # also cross-check beats
                                                     # against cf-rhythm.dat
"""

from __future__ import annotations

import argparse
import gc
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except AttributeError:
        pass

try:
    from . import config
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from engine import config  # type: ignore


# ── Constants ──────────────────────────────────────────────────────────────

AUDIO_EXTS = ('.wav', '.mp3', '.flac', '.m4a')

# Camelot wheel mapping — mirrors wsi_musickey.py in the v1.5 code. The
# Player's KEY readout expects exactly these codes. Majors = *B, minors = *A.
_CAMELOT_MAJOR = {
    'C':  '8B',  'C#': '3B',  'Db': '3B',
    'D':  '10B', 'D#': '5B',  'Eb': '5B',
    'E':  '12B',
    'F':  '7B',  'F#': '2B',  'Gb': '2B',
    'G':  '9B',  'G#': '4B',  'Ab': '4B',
    'A':  '11B', 'A#': '6B',  'Bb': '6B',
    'B':  '1B',
}
_CAMELOT_MINOR = {
    'C':  '5A',  'C#': '12A', 'Db': '12A',
    'D':  '7A',  'D#': '2A',  'Eb': '2A',
    'E':  '9A',
    'F':  '4A',  'F#': '11A', 'Gb': '11A',
    'G':  '6A',  'G#': '1A',  'Ab': '1A',
    'A':  '8A',  'A#': '3A',  'Bb': '3A',
    'B':  '10A',
}

# Krumhansl-Kessler key-profile correlations. Copied from analyze-beats.py
# (v1.5) for continuity — values from Krumhansl 1990.
_KK_MAJOR = (6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88)
_KK_MINOR = (6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17)
_NOTE_NAMES = ('C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B')

# BPM-segment threshold — if two adjacent beats show instantaneous tempo
# within this many BPM, they're the same segment. A tempo change of 4 BPM
# is a very deliberate author decision (remixes / mashups) — anything
# smaller is detector jitter.
BPM_CLUSTER_DELTA = 2.0

# Section target count per 60 s of audio. 3-min track ≈ 9-10 sections,
# 5-min track ≈ 15. Detector decides ultimate count; this is the seed.
SECTIONS_PER_MIN = 3.0
SECTIONS_MIN = 4
SECTIONS_MAX = 24


# ── Data ───────────────────────────────────────────────────────────────────

@dataclass
class TrackStructure:
    track_id: str
    duration_s: float
    bpm_segments: list[tuple[float, float]] = field(default_factory=list)  # (bpm, t_ms)
    beats_ms: list[int] = field(default_factory=list)
    downbeats_ms: list[int] = field(default_factory=list)
    sections: list[tuple[str, int, int]] = field(default_factory=list)     # (label, start_ms, end_ms)
    keys: list[tuple[str, int]] = field(default_factory=list)              # (camelot, t_ms)


# ── Helpers ────────────────────────────────────────────────────────────────

def _camelot_of(root: str, mode: str) -> str:
    table = _CAMELOT_MAJOR if mode == 'maj' else _CAMELOT_MINOR
    return table.get(root, '?')


def _key_of_chroma(chroma_mean) -> str:
    """Krumhansl-Kessler correlation on a 12-bin chroma vector. Returns a
    Camelot code. `chroma_mean` is a length-12 numpy array."""
    import numpy as np
    major = np.asarray(_KK_MAJOR, dtype=float)
    minor = np.asarray(_KK_MINOR, dtype=float)
    best_score = -1.0
    best_root = 'C'
    best_mode = 'maj'
    for i in range(12):
        rm = np.roll(major, i)
        rn = np.roll(minor, i)
        sm = float(np.corrcoef(chroma_mean, rm)[0, 1])
        sn = float(np.corrcoef(chroma_mean, rn)[0, 1])
        if sm > best_score:
            best_score, best_root, best_mode = sm, _NOTE_NAMES[i], 'maj'
        if sn > best_score:
            best_score, best_root, best_mode = sn, _NOTE_NAMES[i], 'min'
    return _camelot_of(best_root, best_mode)


def _cluster_bpm_segments(beats_s: Iterable[float], delta: float = BPM_CLUSTER_DELTA) -> list[tuple[float, float]]:
    """Turn a list of beat timestamps (seconds) into `(bpm, start_ms)` segments.
    Adjacent beats whose instantaneous tempo differs by < `delta` BPM are
    merged; a new segment starts where the gap exceeds the threshold."""
    import numpy as np
    beats = list(beats_s)
    if len(beats) < 4:
        return []
    intervals = np.diff(np.asarray(beats, dtype=float))
    inst_bpm = 60.0 / intervals
    segments: list[tuple[float, float]] = [(float(inst_bpm[0]), int(beats[0] * 1000))]
    run_bpm = [float(inst_bpm[0])]
    run_start_idx = 0
    for i in range(1, len(inst_bpm)):
        if abs(inst_bpm[i] - np.mean(run_bpm)) > delta:
            # Finalize current run, seed new run at beat i.
            segments[-1] = (round(float(np.mean(run_bpm)), 1), segments[-1][1])
            segments.append((float(inst_bpm[i]), int(beats[i] * 1000)))
            run_bpm = [float(inst_bpm[i])]
            run_start_idx = i
        else:
            run_bpm.append(float(inst_bpm[i]))
    # Finalize last run
    segments[-1] = (round(float(np.mean(run_bpm)), 1), segments[-1][1])
    return segments


def _snap_to_downbeats(boundary_s: Iterable[float], downbeats_s: list[float]) -> list[float]:
    """For each agglomerative boundary, snap to the nearest downbeat so
    sections always start on bar-1. If no downbeats are available, return
    the boundaries unchanged."""
    if not downbeats_s:
        return list(boundary_s)
    import bisect
    out: list[float] = []
    for b in boundary_s:
        i = bisect.bisect_left(downbeats_s, b)
        candidates = []
        if i < len(downbeats_s):
            candidates.append(downbeats_s[i])
        if i > 0:
            candidates.append(downbeats_s[i - 1])
        if candidates:
            out.append(min(candidates, key=lambda x: abs(x - b)))
        else:
            out.append(b)
    # Dedup — two nearby boundaries can snap to the same downbeat.
    deduped = []
    for v in out:
        if not deduped or abs(v - deduped[-1]) > 0.25:
            deduped.append(v)
    return deduped


# ── Stage runners ──────────────────────────────────────────────────────────

def _beats_and_downbeats(audio_path: Path):
    """madmom DBN beat + downbeat tracker. Returns (beats_s, downbeats_s) in
    seconds. Both arrays are sorted ascending."""
    from madmom.features.beats import RNNBeatProcessor, DBNBeatTrackingProcessor
    from madmom.features.downbeats import RNNDownBeatProcessor, DBNDownBeatTrackingProcessor
    src = str(audio_path)
    beats = DBNBeatTrackingProcessor(fps=100)(RNNBeatProcessor()(src))
    downs_raw = DBNDownBeatTrackingProcessor(beats_per_bar=[3, 4], fps=100)(RNNDownBeatProcessor()(src))
    # downs_raw is shape (N, 2) — columns [time_s, beat_in_bar]. We want
    # only the "1"s (downbeats).
    downbeats = [float(row[0]) for row in downs_raw if int(row[1]) == 1]
    return list(map(float, beats)), downbeats


def _chroma_sections(audio_path: Path, downbeats_s: list[float], duration_s: float):
    """Agglomerative segmentation on chroma-CENS recurrence, snapped to
    downbeats. Returns list of (label, start_s, end_s) — labels are
    anonymous `section_N`, Gemma judge labels them later."""
    import librosa
    import numpy as np
    y, sr = librosa.load(str(audio_path), sr=None)
    chroma = librosa.feature.chroma_cens(y=y, sr=sr)
    target_n = int(round(duration_s / 60.0 * SECTIONS_PER_MIN))
    k = max(SECTIONS_MIN, min(SECTIONS_MAX, target_n))
    bounds_frames = librosa.segment.agglomerative(chroma, k=k)
    bounds_s = librosa.frames_to_time(bounds_frames, sr=sr).tolist()
    # Ensure 0 and duration are in the boundary set.
    if bounds_s[0] > 0.5:
        bounds_s.insert(0, 0.0)
    if bounds_s[-1] < duration_s - 0.5:
        bounds_s.append(duration_s)
    bounds_s = _snap_to_downbeats(bounds_s, downbeats_s)
    # Need duration at end for last section's right-edge.
    if bounds_s[-1] < duration_s - 0.5:
        bounds_s.append(duration_s)
    sections = []
    for i, (a, b) in enumerate(zip(bounds_s[:-1], bounds_s[1:])):
        if b - a < 2.0:
            continue  # drop sub-2s segments; probably boundary-snap artefacts
        sections.append((f'section_{i}', a, b))
    return y, sr, chroma, sections


def _sectional_keys(y, sr, chroma, sections: list[tuple[str, float, float]]) -> list[tuple[str, float]]:
    """Per-section chroma mean → K-K → Camelot. Also computes a single
    "base" key at t=0 which is the whole-track key for comparison."""
    import librosa
    import numpy as np
    # Times in the chroma array for each section
    out: list[tuple[str, float]] = []
    for label, a, b in sections:
        a_frame = int(librosa.time_to_frames(a, sr=sr))
        b_frame = int(librosa.time_to_frames(b, sr=sr))
        if b_frame <= a_frame:
            continue
        section_chroma = chroma[:, a_frame:b_frame].mean(axis=1)
        cam = _key_of_chroma(section_chroma)
        out.append((cam, a))
    # Dedup consecutive identical keys — only emit key CHANGES.
    collapsed: list[tuple[str, float]] = []
    for cam, t in out:
        if not collapsed or collapsed[-1][0] != cam:
            collapsed.append((cam, t))
    return collapsed


# ── Per-track orchestrator ─────────────────────────────────────────────────

def analyse_track(audio_path: Path) -> TrackStructure:
    import librosa
    # Duration — grabbed cheaply from librosa without reloading.
    duration_s = float(librosa.get_duration(path=str(audio_path)))

    t = time.perf_counter()
    beats_s, downbeats_s = _beats_and_downbeats(audio_path)
    t_beats = time.perf_counter() - t

    t = time.perf_counter()
    y, sr, chroma, sections = _chroma_sections(audio_path, downbeats_s, duration_s)
    t_sections = time.perf_counter() - t

    t = time.perf_counter()
    keys = _sectional_keys(y, sr, chroma, sections)
    t_keys = time.perf_counter() - t

    bpm_segments = _cluster_bpm_segments(beats_s)

    # Release big arrays before the next track in the batch.
    del y, chroma
    gc.collect()

    print(f'    beats    {len(beats_s):4d}  downbeats {len(downbeats_s):3d}   {t_beats:6.1f}s')
    print(f'    sections {len(sections):4d}  target-k  {int(round(duration_s/60*SECTIONS_PER_MIN)):3d}   {t_sections:6.1f}s')
    print(f'    keys     {len(keys):4d}  (deduped from {len(sections)})   {t_keys:6.1f}s')

    return TrackStructure(
        track_id=audio_path.stem,
        duration_s=duration_s,
        bpm_segments=[(bpm, t_ms) for bpm, t_ms in bpm_segments],
        beats_ms=[int(b * 1000) for b in beats_s],
        downbeats_ms=[int(d * 1000) for d in downbeats_s],
        sections=[(lbl, int(a * 1000), int(b * 1000)) for lbl, a, b in sections],
        keys=[(cam, int(t_sec * 1000)) for cam, t_sec in keys],
    )


# ── Serialization ──────────────────────────────────────────────────────────

def render_block(s: TrackStructure) -> str:
    lines = [f'[{s.track_id}]']
    lines.append(f'duration_ms   = {int(s.duration_s * 1000)}')
    lines.append(
        'bpm_segments  = '
        + ', '.join(f'{bpm:.1f}@{t}ms' for bpm, t in s.bpm_segments)
    )
    lines.append('downbeats_ms  = ' + ', '.join(str(v) for v in s.downbeats_ms))
    lines.append(
        'sections      = '
        + ', '.join(f'{lbl}:{a}-{b}' for lbl, a, b in s.sections)
    )
    lines.append(
        'keys          = '
        + ', '.join(f'{cam}@{t}' for cam, t in s.keys)
    )
    return '\n'.join(lines)


def write_structure_file(path: Path, blocks: dict[str, str]) -> None:
    """Overwrites the file with one block per track, sorted by track_id.
    `blocks` is {track_id: rendered-block-text}."""
    path.parent.mkdir(parents=True, exist_ok=True)
    body = '\n\n'.join(blocks[k] for k in sorted(blocks))
    header = (
        '# cf-structure.dat — generated by engine/structure.py\n'
        '# Format per track: [track_id] followed by key=value lines\n'
        '# bpm_segments: list of "bpm@start_ms" (1 or more tempo regions)\n'
        '# downbeats_ms: comma-separated bar-1 timestamps\n'
        '# sections:     label:start_ms-end_ms, where label = section_N for now\n'
        '#               (Gemma judge relabels to intro/verse/chorus/drop later)\n'
        '# keys:         Camelot@start_ms (only emitted on key change)\n'
        '\n'
    )
    path.write_text(header + body + '\n', encoding='utf-8')


def parse_existing(path: Path) -> dict[str, str]:
    """Read an existing cf-structure.dat back into {track_id: block-text} so
    we can resume-skip tracks already analysed."""
    if not path.is_file():
        return {}
    text = path.read_text(encoding='utf-8')
    blocks: dict[str, str] = {}
    current_id = None
    current_lines: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith('#'):
            continue
        if stripped.startswith('[') and stripped.endswith(']'):
            if current_id is not None and current_lines:
                blocks[current_id] = '\n'.join([f'[{current_id}]'] + current_lines)
            current_id = stripped[1:-1]
            current_lines = []
        elif current_id is not None and '=' in stripped:
            current_lines.append(line)
    if current_id is not None and current_lines:
        blocks[current_id] = '\n'.join([f'[{current_id}]'] + current_lines)
    return blocks


# ── Discovery ──────────────────────────────────────────────────────────────

def list_all_tracks() -> list[Path]:
    """Same logic as stems.py — all supported audio, deduped by canonical
    stem with .wav preferred over lossy formats."""
    if not config.V1_TRACKS_DIR.is_dir():
        return []
    found: list[Path] = []
    for ext in AUDIO_EXTS:
        found.extend(config.V1_TRACKS_DIR.glob(f'*{ext}'))

    def _canonical_stem(p: Path) -> str:
        s = p.stem
        for suffix in ('-high', '-normal', '-off'):
            if s.endswith(suffix):
                s = s[: -len(suffix)]
                break
        return s

    rank = {'.wav': 3, '.flac': 2, '.m4a': 1, '.mp3': 0}
    best: dict[str, Path] = {}
    for p in found:
        canon = _canonical_stem(p)
        if canon not in best or rank.get(p.suffix.lower(), -1) > rank.get(best[canon].suffix.lower(), -1):
            best[canon] = p
    return [best[k] for k in sorted(best)]


# ── Validator (optional) ───────────────────────────────────────────────────

def _validate_beats(s: TrackStructure, rhythm_path: Path) -> str | None:
    """Cross-check our beat estimates against v1.5 cf-rhythm.dat (hand-
    refined ground truth). Returns a short diagnostic string or None."""
    if not rhythm_path.is_file():
        return None
    import bisect
    # cf-rhythm.dat line format: <id>:<bpm*100>:<key>:<beats_ms,...>[:<flash>[:<hidden>]]
    reference_beats: list[int] | None = None
    for line in rhythm_path.read_text(encoding='utf-8').splitlines():
        if not line.strip() or line.startswith('#'):
            continue
        parts = line.split(':')
        if not parts[0].strip().startswith(s.track_id.split('-normal')[0].split('-high')[0]):
            continue
        if len(parts) < 4:
            continue
        try:
            reference_beats = [int(x) for x in parts[3].split(',') if x.strip()]
        except ValueError:
            return None
        break
    if not reference_beats:
        return None
    # For each our-beat, find the nearest reference-beat, compute offset.
    ours = sorted(s.beats_ms)
    refs = sorted(reference_beats)
    offsets = []
    for b in ours:
        i = bisect.bisect_left(refs, b)
        candidates = []
        if i < len(refs): candidates.append(refs[i])
        if i > 0: candidates.append(refs[i - 1])
        if candidates:
            offsets.append(min(abs(b - c) for c in candidates))
    if not offsets:
        return None
    offsets.sort()
    median = offsets[len(offsets) // 2]
    flag = '  **MISMATCH**' if median > 50 else ''
    return f'median beat offset vs cf-rhythm.dat: {median} ms{flag} (n={len(offsets)}, reference n={len(refs)})'


# ── Main ──────────────────────────────────────────────────────────────────

def main() -> int:
    p = argparse.ArgumentParser(description='Phase 2c structural pass.')
    p.add_argument('--track', type=Path, help='Analyse one audio file.')
    p.add_argument('--all', action='store_true', help='Analyse every track under V1_TRACKS_DIR.')
    p.add_argument('--force', action='store_true', help='Re-analyse even if track is in cf-structure.dat.')
    p.add_argument('--validate', action='store_true', help='Cross-check beat timings against v1.5 cf-rhythm.dat.')
    p.add_argument('--out', type=Path, default=config.TRACKS_DIR / 'cf-structure.dat', help='Output path.')
    args = p.parse_args()

    if not args.track and not args.all:
        p.error('pass --track <path> or --all')

    tracks = [args.track] if args.track else list_all_tracks()
    if not tracks:
        print('No tracks found. Check config.V1_TRACKS_DIR.')
        return 1

    existing = parse_existing(args.out)
    print('-' * 72)
    print(f'  structure    · output:  {args.out}')
    print(f'  tracks       · {len(tracks)} candidate(s), {len(existing)} already analysed  {"[FORCE]" if args.force else ""}')
    print(f'  validate     · {"cf-rhythm.dat cross-check ON" if args.validate else "off"}')
    print('-' * 72)

    blocks = dict(existing)
    batch_t0 = time.perf_counter()
    ok = skip = fail = 0

    rhythm_path = config.V1_TRACKS_DIR / 'cf-rhythm.dat' if args.validate else None

    for track in tracks:
        track_id = track.stem
        if track_id in blocks and not args.force:
            print(f'  skip  {track_id:<28} (already in cf-structure.dat)')
            skip += 1
            continue
        try:
            print(f'  --    {track_id}')
            s = analyse_track(track)
            blocks[s.track_id] = render_block(s)
            # Write after every track so a crash mid-batch doesn't lose work.
            write_structure_file(args.out, blocks)
            if rhythm_path:
                diag = _validate_beats(s, rhythm_path)
                if diag:
                    print(f'    {diag}')
            print(f'  ok    {track_id}')
            ok += 1
        except Exception as e:
            fail += 1
            print(f'  FAIL  {track_id:<28} {type(e).__name__}: {e}')
            if '--verbose' in sys.argv:
                import traceback; traceback.print_exc()
        gc.collect()

    total = time.perf_counter() - batch_t0
    print('-' * 72)
    print(f'  done  {ok} ok · {skip} skip · {fail} fail · {total:.1f}s total')
    return 0 if fail == 0 else 1


if __name__ == '__main__':
    raise SystemExit(main())
