"""Phase 2e — semantic CLAP embeddings per section.

Reads cf-structure.dat to get section boundaries, extracts each section's
audio (~30 s clip around its midpoint), runs LAION-CLAP to produce a 512-dim
audio embedding, and stores one embedding per section to cf-embeddings.npz.

Also pre-computes a handful of text-embedding "queries" (e.g. "dark bass
drop", "euphoric lead", "tension build") so downstream stages can project
section embeddings onto those semantic axes without re-running CLAP per
query. The resulting similarity scalars are what the Gemma judge (2g)
reasons over when it's assembling A/B/C mix blueprints.

Output file: wsi-rx/tracks/cf-embeddings.npz
Schema (numpy savez):
    tracks          : ndarray[str]        shape (N,)     track_id per entry
    section_indices : ndarray[int]        shape (N,)     section_N (0-based)
    start_ms        : ndarray[int]        shape (N,)     section start
    end_ms          : ndarray[int]        shape (N,)     section end
    embeddings      : ndarray[float32]    shape (N, 512)
    query_texts     : ndarray[str]        shape (Q,)     the text prompts
    query_embeddings: ndarray[float32]    shape (Q, 512)

Usage:
  python -m engine.embeddings --all              # embed every section in cf-structure.dat
  python -m engine.embeddings --track <stem>     # embed one track's sections
  python -m engine.embeddings --all --force      # recompute everything
"""

from __future__ import annotations

import argparse
import gc
import sys
import time
from pathlib import Path

if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except AttributeError:
        pass

try:
    from . import config
    from .structure import parse_existing, list_all_tracks
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from engine import config  # type: ignore
    from engine.structure import parse_existing, list_all_tracks  # type: ignore


# Section clip length fed to CLAP. CLAP was trained on ~10 s clips but
# handles longer windows gracefully; 30 s centred on the section midpoint
# gives the text/audio alignment enough context for structural vibes like
# "build-up" to register without getting swamped by long outros.
CLIP_SECONDS = 30.0

# These are the semantic axes the judge reasons over. Add whatever vibes
# the author thinks the DJ judge needs to compare sections on. Keep
# descriptions musical, not technical — CLAP was trained on audio-caption
# pairs, not EQ specs.
DEFAULT_QUERY_TEXTS = [
    'dark bass drop',
    'euphoric melodic lead',
    'tension build-up with risers',
    'soft breakdown with sparse vocals',
    'four-on-the-floor driving chorus',
    'ambient pad intro',
    'cinematic outro with reverb tail',
]


def _load_clap():
    """Singleton-style CLAP loader. Returns a configured CLAP_Module."""
    import laion_clap
    model = laion_clap.CLAP_Module(enable_fusion=False)
    # First call downloads ~600 MB to ~/.cache/clap. Subsequent calls reuse
    # the cached checkpoint.
    model.load_ckpt()
    return model


def _section_boundaries_from_block(block_text: str) -> list[tuple[int, int, int]]:
    """Pull (section_idx, start_ms, end_ms) tuples from one cf-structure.dat
    block. Format of the sections line:
        sections      = section_0:0-12000, section_1:12000-36000, ...
    """
    out: list[tuple[int, int, int]] = []
    for raw_line in block_text.splitlines():
        line = raw_line.strip()
        if not line.startswith('sections'):
            continue
        _, _, rhs = line.partition('=')
        for i, part in enumerate(rhs.split(',')):
            part = part.strip()
            if not part:
                continue
            # "section_0:0-12000" — split label:range
            label, _, rng = part.partition(':')
            a, _, b = rng.partition('-')
            try:
                out.append((i, int(a), int(b)))
            except ValueError:
                continue
        break
    return out


def _resolve_audio_path(track_id: str) -> Path | None:
    """Find the source audio file for a track_id (which is just the basename
    stem, e.g. 'td-audio001-high'). Falls back across supported extensions."""
    for ext in ('.wav', '.mp3', '.flac', '.m4a'):
        candidate = config.V1_TRACKS_DIR / f'{track_id}{ext}'
        if candidate.is_file():
            return candidate
    return None


def _extract_clip(audio_path: Path, start_ms: int, end_ms: int) -> 'tuple[np.ndarray, int]':  # noqa: F821
    """Load up to CLIP_SECONDS centred on the section midpoint. Returns
    (mono float32 array, sample_rate). Pads with zeros if the section is
    shorter than CLIP_SECONDS."""
    import librosa
    import numpy as np
    mid_s = (start_ms + end_ms) / 2000.0
    section_len_s = (end_ms - start_ms) / 1000.0
    half_clip = CLIP_SECONDS / 2.0
    clip_start_s = max(0.0, mid_s - half_clip)
    clip_dur_s = min(CLIP_SECONDS, section_len_s) if section_len_s < CLIP_SECONDS else CLIP_SECONDS
    y, sr = librosa.load(str(audio_path), sr=48000, offset=clip_start_s, duration=clip_dur_s, mono=True)
    # CLAP expects exactly 10 s or a multiple thereof — 30 s fits. Pad if short.
    target_len = int(CLIP_SECONDS * sr)
    if len(y) < target_len:
        y = np.pad(y, (0, target_len - len(y)), mode='constant')
    return y.astype('float32'), sr


def _embed_sections(model, audio_path: Path, sections: list[tuple[int, int, int]]):
    """Run CLAP on each section clip. Returns list of (section_idx, start_ms,
    end_ms, 512-dim embedding)."""
    import numpy as np
    out = []
    clips: list['np.ndarray'] = []  # noqa: F821
    meta: list[tuple[int, int, int]] = []
    for idx, a_ms, b_ms in sections:
        y, _ = _extract_clip(audio_path, a_ms, b_ms)
        clips.append(y)
        meta.append((idx, a_ms, b_ms))
    # CLAP's get_audio_embedding_from_data batches in memory — safe up to a
    # dozen or so 30-s clips per call.
    # data shape must be (batch, samples) per laion_clap docs.
    if not clips:
        return out
    batch = np.stack(clips, axis=0)
    emb = model.get_audio_embedding_from_data(x=batch, use_tensor=False)
    for (idx, a_ms, b_ms), e in zip(meta, emb):
        out.append((idx, a_ms, b_ms, np.asarray(e, dtype='float32')))
    return out


def main() -> int:
    p = argparse.ArgumentParser(description='Phase 2e CLAP semantic embeddings.')
    p.add_argument('--track', type=str, default=None, help='One track_id to embed (as it appears in cf-structure.dat).')
    p.add_argument('--all', action='store_true', help='Embed every track present in cf-structure.dat.')
    p.add_argument('--force', action='store_true', help='Re-embed even when the track is already in cf-embeddings.npz.')
    p.add_argument('--in', dest='in_path', type=Path, default=config.TRACKS_DIR / 'cf-structure.dat', help='Input structure file.')
    p.add_argument('--out', type=Path, default=config.TRACKS_DIR / 'cf-embeddings.npz', help='Output .npz path.')
    args = p.parse_args()

    if not args.track and not args.all:
        p.error('pass --track <track_id> or --all')

    if not args.in_path.is_file():
        print(f'ERROR: structure file not found — {args.in_path}. Run `python -m engine.structure --all` first.')
        return 1

    blocks = parse_existing(args.in_path)
    if not blocks:
        print(f'ERROR: no track blocks in {args.in_path}.')
        return 1

    # Pick the target set
    if args.track:
        if args.track not in blocks:
            print(f'ERROR: track_id {args.track!r} not found in {args.in_path}. Known: {sorted(blocks)[:5]}...')
            return 1
        target_ids = [args.track]
    else:
        target_ids = sorted(blocks)

    # Load existing embeddings (for resume-skip) — kept in memory, re-written
    # after every track's compute completes so crashes don't lose work.
    import numpy as np
    already = set()
    existing = {
        'tracks': [], 'section_indices': [], 'start_ms': [], 'end_ms': [],
        'embeddings': [], 'query_texts': [], 'query_embeddings': [],
    }
    if args.out.is_file():
        try:
            npz = np.load(args.out, allow_pickle=False)
            for k in existing:
                if k in npz.files:
                    existing[k] = npz[k].tolist()
            already = set(existing['tracks']) if not args.force else set()
        except Exception as e:
            print(f'  warn: couldn\'t read existing {args.out}: {e}  (starting fresh)')

    print('-' * 72)
    print(f'  embeddings   · in:      {args.in_path}')
    print(f'  embeddings   · out:     {args.out}')
    print(f'  tracks       · {len(target_ids)} candidate(s), {len(already)} already embedded  {"[FORCE]" if args.force else ""}')
    print('-' * 72)

    todo = [tid for tid in target_ids if tid not in already]
    if not todo:
        print('  nothing to do — everything already embedded. Pass --force to re-run.')
        return 0

    print('  loading CLAP...')
    t0 = time.perf_counter()
    model = _load_clap()
    print(f'  loaded in {time.perf_counter() - t0:.1f}s')

    # Compute query embeddings ONCE per run — cheap (~0.3s) but good to log.
    t0 = time.perf_counter()
    query_embs = model.get_text_embedding(DEFAULT_QUERY_TEXTS, use_tensor=False)
    print(f'  query texts  · {len(DEFAULT_QUERY_TEXTS)} prompts embedded in {time.perf_counter() - t0:.1f}s')
    existing['query_texts'] = list(DEFAULT_QUERY_TEXTS)
    existing['query_embeddings'] = np.asarray(query_embs, dtype='float32').tolist()
    print()

    batch_t0 = time.perf_counter()
    ok = skip = fail = 0
    for track_id in todo:
        audio = _resolve_audio_path(track_id)
        if audio is None:
            print(f'  skip  {track_id:<28} (source audio not found under V1_TRACKS_DIR)')
            skip += 1
            continue
        sections = _section_boundaries_from_block(blocks[track_id])
        if not sections:
            print(f'  skip  {track_id:<28} (no sections in structure block)')
            skip += 1
            continue
        try:
            t = time.perf_counter()
            embs = _embed_sections(model, audio, sections)
            dt = time.perf_counter() - t
            # Append to the master lists. If --force is set, strip old rows
            # for this track first.
            if args.force:
                keep_mask = [t != track_id for t in existing['tracks']]
                for k in ('tracks', 'section_indices', 'start_ms', 'end_ms', 'embeddings'):
                    existing[k] = [v for v, keep in zip(existing[k], keep_mask) if keep]
            for idx, a_ms, b_ms, e in embs:
                existing['tracks'].append(track_id)
                existing['section_indices'].append(int(idx))
                existing['start_ms'].append(int(a_ms))
                existing['end_ms'].append(int(b_ms))
                existing['embeddings'].append(e.tolist())
            # Write after every track so a crash doesn't lose prior work.
            np.savez(
                args.out,
                tracks=np.asarray(existing['tracks']),
                section_indices=np.asarray(existing['section_indices'], dtype='int64'),
                start_ms=np.asarray(existing['start_ms'], dtype='int64'),
                end_ms=np.asarray(existing['end_ms'], dtype='int64'),
                embeddings=np.asarray(existing['embeddings'], dtype='float32'),
                query_texts=np.asarray(existing['query_texts']),
                query_embeddings=np.asarray(existing['query_embeddings'], dtype='float32'),
            )
            print(f'  ok    {track_id:<28} {dt:6.1f}s  {len(embs)} sections')
            ok += 1
        except Exception as e:
            fail += 1
            print(f'  FAIL  {track_id:<28} {type(e).__name__}: {e}')
        gc.collect()

    total = time.perf_counter() - batch_t0
    print('-' * 72)
    print(f'  done  {ok} ok · {skip} skip · {fail} fail · {total:.1f}s total')
    return 0 if fail == 0 else 1


if __name__ == '__main__':
    raise SystemExit(main())
