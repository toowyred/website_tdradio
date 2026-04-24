#!/usr/bin/env python3
# ─────────────────────────────────────────────────────────────────
# analyze-beats.py — offline BPM / key / beat-grid extractor + manual
# flash-beat refiner for WSI Radio tracks.
#
# Deps:
#     pip install librosa matplotlib pygame
#
# Run:
#     python analyze-beats.py          → opens the main menu
#     python analyze-beats.py --auto   → headless: run Auto on every track
#     python analyze-beats.py --refine [track-id]  → jump into Refine UI
#
# OUTPUT FORMAT (cf-rhythm.dat):
#
#     Required fields:
#       <id> : <bpm-field> : <key> : <beats_ms,beats_ms,...>
#     Optional 5th field (flash indices into the beats list):
#       <id> : <bpm-field> : <key> : <beats...> : <idx,idx,idx,...>
#
#   <bpm-field> is a bare <bpm*100> for single-tempo tracks, or a
#   pipe-separated list of <bpm*100>@<t_ms> segments for multi-tempo.
#
#   Example — auto output, no explicit flashes (default every-4th applies):
#     td-audio001:12450:Am:512,1024,1536,2048,2560,3072,3584,4096
#
#   Example — refined: flashes on beats 0, 3, 6 (not the default 0, 4):
#     td-audio001:12450:Am:512,1024,1536,2048,2560,3072,3584,4096:0,3,6
#
# PIPELINE (unchanged from v1):
#   Phase 1  30 s/15 s-hop coarse BPM scan
#   Phase 2  Boundary detect on |ΔBPM| > 4
#   Phase 3  Zoom into suspect regions with 5 s/2.5 s-hop
#   Phase 4  Per-segment beat_track() with start_bpm prior
#   Phase 5  Merge segments < 10 s and close-BPM (< 2 BPM delta)
#
# REFINE UI:
#   Waveform + every beat rendered as a vertical tick.
#   Bright tall tick = flash beat. Dim short tick = regular beat.
#   Click a beat to toggle. SPACE plays/pauses. Live playback cursor.
#   SAVE writes back in-place (one line in cf-rhythm.dat updated).
# ─────────────────────────────────────────────────────────────────

import sys
import re
import argparse
import time
from pathlib import Path

try:
    import numpy as np
    import librosa
except ImportError:
    print("error: librosa is required.  pip install librosa", file=sys.stderr)
    sys.exit(1)

HERE = Path(__file__).resolve().parent
OUT  = HERE / "cf-rhythm.dat"

SUFFIX_RE = re.compile(r"-(normal|high|off)\.(mp3|wav)$", re.I)
AUDIO_RE  = re.compile(r"\.(mp3|wav)$", re.I)

KK_MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
KK_MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

PRIORITY = ["-high.wav", "-high.mp3", "-normal.mp3", "-off.mp3"]

# Tuning knobs for segmentation.
CHUNK_S, HOP_S           = 30.0, 15.0
FINE_CHUNK_S, FINE_HOP_S =  5.0,  2.5
CHANGE_THRESHOLD         =  4.0
MERGE_THRESHOLD          =  2.0
MIN_SEG_S                = 10.0

# Default flash rule: every 4th beat starting at index 0 (bar downbeats).
DEFAULT_FLASH_STEP = 4


# ═══════════════════════════════════════════════════════════════
# FILE HELPERS
# ═══════════════════════════════════════════════════════════════

def track_id(name: str):
    m = SUFFIX_RE.search(name)
    if m:
        return name[: m.start()]
    if AUDIO_RE.search(name):
        return AUDIO_RE.sub("", name)
    return None


def pick_best(candidates, tid, path):
    lname = path.name.lower()
    rank = next((i for i, suf in enumerate(PRIORITY) if lname.endswith(suf)), 99)
    prev = candidates.get(tid)
    if prev is None or rank < prev[0]:
        candidates[tid] = (rank, path)


def scan_folder():
    """Return { track_id → (rank, Path) } for every audio file in HERE."""
    candidates = {}
    for p in HERE.iterdir():
        if not p.is_file() or p.name == OUT.name or not AUDIO_RE.search(p.name):
            continue
        tid = track_id(p.name)
        if tid:
            pick_best(candidates, tid, p)
    return candidates


def find_audio_for(tid):
    """Return the best-priority audio path for a track id (or None)."""
    candidates = scan_folder()
    entry = candidates.get(tid)
    return entry[1] if entry else None


def default_flashes(n_beats):
    """Default flash indices: every 4th beat starting at 0."""
    return list(range(0, n_beats, DEFAULT_FLASH_STEP))


# ═══════════════════════════════════════════════════════════════
# LINE PARSE / ENCODE
# ═══════════════════════════════════════════════════════════════

def parse_line(line):
    """Parse one cf-rhythm.dat line into a record dict, or None.

    Fields (colon-separated):
      1. <track-id>
      2. <bpm-field>           bare <bpm*100>  OR  <bpm*100>@<t_ms>|…   (segments)
      3. <key>                 e.g. Am / C#m / F#
      4. <beats_ms>            comma-separated beat timestamps in ms
      5. <flash_indices>       OPTIONAL. Present (even empty) → authoritative:
                               those indices flash, NOTHING else does. Absent
                               (4-field line) → default every-4th rule applies.
      6. <hidden_indices>      OPTIONAL. Beats the Player should NOT render at
                               all (no color mark, no flash). Distinct from
                               flashes: hidden beats stay in the beats list
                               (for BPM derivation + Refine reference) but
                               show up as 0-flag on the Player.
    """
    line = line.strip()
    if not line or line.startswith("#"):
        return None
    # maxsplit=5 to keep 6-field lines intact (colon only splits the first 5 times).
    parts = line.split(":", 5)
    if len(parts) < 4:
        return None
    tid, bpm_field, key, beats_field = parts[0], parts[1], parts[2], parts[3]

    # BPM / segments
    if "|" in bpm_field or "@" in bpm_field:
        segments = []
        for seg in bpm_field.split("|"):
            at = seg.find("@")
            try:
                bpm = int(seg[:at] if at > 0 else seg) / 100
                start = int(seg[at + 1:]) / 1000 if at > 0 else 0.0
            except ValueError:
                continue
            segments.append((start, bpm))
        if not segments:
            return None
        segments.sort(key=lambda s: s[0])
        primary = segments[0][1]
    else:
        try:
            primary = int(bpm_field) / 100
        except ValueError:
            return None
        segments = [(0.0, primary)]

    # Beats (ms → seconds)
    beats = []
    for x in beats_field.split(","):
        x = x.strip()
        if not x:
            continue
        try:
            beats.append(int(x) / 1000)
        except ValueError:
            continue

    def parse_indices(field):
        out = set()
        for x in field.split(","):
            x = x.strip()
            if x.lstrip("-").isdigit():
                v = int(x)
                if 0 <= v < len(beats):
                    out.add(v)
        return sorted(out)

    # Flashes — present (incl. empty) means authoritative. Absent = default rule.
    if len(parts) >= 5:
        flashes = parse_indices(parts[4])
    else:
        flashes = default_flashes(len(beats))

    # Hidden — only meaningful when the 6th field exists. Absent = nothing hidden.
    hidden = parse_indices(parts[5]) if len(parts) >= 6 else []

    return {
        "tid": tid, "bpm": primary, "key": key,
        "beats": beats, "segments": segments,
        "flashes": flashes, "hidden": hidden,
    }


def encode_line(record):
    """Build a single cf-rhythm.dat line from a record dict. Emits the minimum
    number of fields needed: 4 for unrefined tracks, 5 when flash pattern is
    non-default, 6 when there are also hidden beats."""
    tid = record["tid"]
    segments = record["segments"]
    bpm = record["bpm"]
    key = record["key"]
    beats = record["beats"]
    flashes = sorted(record.get("flashes", default_flashes(len(beats))))
    hidden  = sorted(record.get("hidden", []))

    if len(segments) <= 1:
        bpm_field = str(int(round(bpm * 100)))
    else:
        bpm_field = "|".join(
            f"{int(round(b * 100))}@{int(round(t * 1000))}"
            for t, b in segments
        )
    beats_ms = ",".join(str(int(round(b * 1000))) for b in beats)
    line = f"{tid}:{bpm_field}:{key}:{beats_ms}"

    is_default_flash = (flashes == default_flashes(len(beats)))

    if hidden:
        # Need both optional fields present so the 6th field lines up. The 5th
        # field always carries an explicit flash list when hidden is set —
        # otherwise a parser seeing 6 fields with an empty 5th would interpret
        # "zero flashes" instead of "default every-4th".
        flash_field = ",".join(str(i) for i in flashes)
        hidden_field = ",".join(str(i) for i in hidden)
        line += f":{flash_field}:{hidden_field}"
    elif not is_default_flash:
        line += ":" + ",".join(str(i) for i in flashes)
    # else: 4-field compact form; default every-4th applies
    return line


def load_all_records():
    """Parse cf-rhythm.dat into { track_id → record }."""
    out = {}
    if not OUT.exists():
        return out
    for line in OUT.read_text(encoding="utf-8").splitlines():
        rec = parse_line(line)
        if rec:
            out[rec["tid"]] = rec
    return out


def save_record(record):
    """Replace one track's line in cf-rhythm.dat (preserve every other)."""
    all_recs = load_all_records()
    all_recs[record["tid"]] = record
    lines = [encode_line(all_recs[tid]) for tid in sorted(all_recs)]
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ═══════════════════════════════════════════════════════════════
# ANALYSIS PIPELINE (unchanged from v1)
# ═══════════════════════════════════════════════════════════════

def detect_key(y, sr):
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    profile = chroma.mean(axis=1)
    best_corr, best_key = -np.inf, ""
    for shift in range(12):
        rotated = np.roll(profile, -shift)
        cmaj = float(np.dot(rotated, KK_MAJOR))
        cmin = float(np.dot(rotated, KK_MINOR))
        if cmaj > best_corr:
            best_corr, best_key = cmaj, NOTE_NAMES[shift]
        if cmin > best_corr:
            best_corr, best_key = cmin, NOTE_NAMES[shift] + "m"
    return best_key


def bpm_of(y_slice, sr, start_bpm=None):
    try:
        kwargs = {"y": y_slice, "sr": sr}
        if start_bpm:
            kwargs["start_bpm"] = float(start_bpm)
        tempo, _ = librosa.beat.beat_track(**kwargs)
        return float(np.squeeze(tempo))
    except Exception:
        return 0.0


def coarse_scan(y, sr):
    total = len(y) / sr
    out = []
    t = 0.0
    while t < total:
        end = min(t + CHUNK_S, total)
        if end - t < 5.0:
            break
        yc = y[int(t * sr): int(end * sr)]
        out.append((t, end, bpm_of(yc, sr)))
        t += HOP_S
    return out


def refine_boundary(y, sr, t_start, t_end, bpm_before, bpm_after):
    total = len(y) / sr
    t = t_start
    end_cap = min(t_end, total)
    while t + FINE_CHUNK_S <= end_cap:
        yc = y[int(t * sr): int((t + FINE_CHUNK_S) * sr)]
        bpm = bpm_of(yc, sr)
        if bpm > 0 and abs(bpm - bpm_after) < abs(bpm - bpm_before):
            return t
        t += FINE_HOP_S
    return (t_start + t_end) / 2.0


def detect_segments(y, sr, coarse):
    if not coarse:
        return []
    segments = [(0.0, coarse[0][2])]
    for i in range(1, len(coarse)):
        current_bpm = segments[-1][1]
        chunk_bpm = coarse[i][2]
        if chunk_bpm <= 0 or current_bpm <= 0:
            continue
        if abs(chunk_bpm - current_bpm) > CHANGE_THRESHOLD:
            t_before = coarse[i - 1][0]
            t_after = coarse[i][1]
            pivot = refine_boundary(y, sr, t_before, t_after, current_bpm, chunk_bpm)
            segments.append((pivot, chunk_bpm))
    return segments


def merge_close_segments(segments, total):
    if len(segments) <= 1:
        return segments
    keep = [segments[0]]
    for i in range(1, len(segments)):
        t, bpm = segments[i]
        next_t = segments[i + 1][0] if i + 1 < len(segments) else total
        if (next_t - t) < MIN_SEG_S:
            continue
        keep.append((t, bpm))
    collapsed = [keep[0]]
    for i in range(1, len(keep)):
        t, bpm = keep[i]
        _, last_bpm = collapsed[-1]
        if abs(bpm - last_bpm) < MERGE_THRESHOLD:
            continue
        collapsed.append((t, bpm))
    return collapsed


def beats_for_segment(y, sr, seg_start, seg_end, seg_bpm):
    start_i = int(seg_start * sr)
    end_i = int(seg_end * sr)
    if end_i - start_i < sr * 2:
        return []
    yc = y[start_i:end_i]
    try:
        _, frames = librosa.beat.beat_track(y=yc, sr=sr, start_bpm=float(seg_bpm))
        rel = librosa.frames_to_time(frames, sr=sr)
        return [seg_start + float(b) for b in rel]
    except Exception:
        return []


def analyze(path):
    y, sr = librosa.load(str(path), sr=None, mono=True)
    total = len(y) / sr

    coarse = coarse_scan(y, sr)
    segments = detect_segments(y, sr, coarse)
    segments = merge_close_segments(segments, total)

    if not segments:
        bpm = bpm_of(y, sr)
        segments = [(0.0, bpm)] if bpm > 0 else []

    all_beats = []
    for i, (seg_start, seg_bpm) in enumerate(segments):
        seg_end = segments[i + 1][0] if i + 1 < len(segments) else total
        all_beats.extend(beats_for_segment(y, sr, seg_start, seg_end, seg_bpm))
    all_beats.sort()

    if segments:
        durations = []
        for i, (s, _) in enumerate(segments):
            e = segments[i + 1][0] if i + 1 < len(segments) else total
            durations.append(e - s)
        primary_bpm = segments[int(np.argmax(durations))][1]
    else:
        primary_bpm = 0.0

    key = detect_key(y, sr)
    return primary_bpm, key, all_beats, segments


def describe_segments(segments, total):
    if len(segments) <= 1:
        bpm = segments[0][1] if segments else 0.0
        return f"{bpm:6.1f} BPM"
    pieces = []
    for i, (s, bpm) in enumerate(segments):
        e = segments[i + 1][0] if i + 1 < len(segments) else total
        pieces.append(f"{bpm:.0f}@{int(s)}–{int(e)}s")
    return " + ".join(pieces)


# ═══════════════════════════════════════════════════════════════
# AUTO-RUN (batch) — the pre-existing behavior
# ═══════════════════════════════════════════════════════════════

def auto_run(force=False, only_track=None):
    seen = {} if force else load_all_records()
    candidates = scan_folder()
    ids = sorted(candidates)
    if only_track:
        if only_track not in candidates:
            print(f"no audio file found for {only_track} in {HERE.name}/", file=sys.stderr)
            return 1
        ids = [only_track]
        seen.pop(only_track, None)

    if not ids:
        print(f"no audio files found in {HERE.name}/")
        return 0

    new_recs, skipped = [], 0
    for tid in ids:
        if tid in seen:
            skipped += 1
            continue
        _, path = candidates[tid]
        print(f"  scan  {tid}  ({path.name}) …", end="", flush=True)
        try:
            primary_bpm, key, beats, segments = analyze(path)
        except Exception as e:
            print(f"  FAIL: {e}")
            continue
        total = librosa.get_duration(path=str(path))
        summary = describe_segments(segments, total)
        rec = {
            "tid": tid, "bpm": primary_bpm, "key": key,
            "beats": beats, "segments": segments,
            "flashes": default_flashes(len(beats)),
            "hidden": [],
        }
        new_recs.append(rec)
        tag = f"  key {key:<3}  {len(beats)} beats"
        if len(segments) > 1:
            print(f"  ⚡ {summary}{tag}  (multi-tempo)")
        else:
            print(f"  {summary}{tag}")

    if skipped:
        print(f"  (skipped {skipped} already-analyzed track{'s' if skipped != 1 else ''})")

    if not new_recs:
        print("nothing new to write.")
        return 0

    # Merge with existing + sort
    all_recs = load_all_records() if not force else {}
    for rec in new_recs:
        all_recs[rec["tid"]] = rec
    lines = [encode_line(all_recs[tid]) for tid in sorted(all_recs)]
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {OUT.name}  ({len(new_recs)} new, {len(all_recs)} total)")
    return 0


# ═══════════════════════════════════════════════════════════════
# GUI — lazy imports (only when --auto isn't used)
# ═══════════════════════════════════════════════════════════════

def _import_gui():
    """Import tkinter / matplotlib / pygame only when we actually need them."""
    mods = {}
    try:
        import tkinter as _tk
        from tkinter import ttk as _ttk, messagebox as _mb
        mods["tk"], mods["ttk"], mods["messagebox"] = _tk, _ttk, _mb
    except ImportError:
        print("error: tkinter is required for the GUI.", file=sys.stderr)
        sys.exit(1)
    try:
        import matplotlib
        matplotlib.use("TkAgg")
        from matplotlib.figure import Figure as _Fig
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg as _FCT
        mods["Figure"], mods["FigureCanvasTkAgg"] = _Fig, _FCT
    except ImportError:
        print("error: matplotlib is required for the Refine GUI.  pip install matplotlib", file=sys.stderr)
        sys.exit(1)
    try:
        import pygame as _pg
        _pg.mixer.init()
        mods["pygame"] = _pg
    except ImportError:
        print("error: pygame is required for audio playback.  pip install pygame", file=sys.stderr)
        sys.exit(1)
    globals().update(mods)


# ── Refine window ─────────────────────────────────────────────

class RefineWindow:
    """Per-track waveform + clickable beat markers + playback cursor."""

    WAVE_COLOR   = "#555"
    BEAT_COLOR   = "#de7029"    # matches TD accent (dark_orange theme)
    FLASH_COLOR  = "#ffb070"
    HIDDEN_COLOR = "#444"       # muted tick — stays visible in Refine, not on Player
    DRAW_COLOR   = "#66aaff"    # preview colour for user-added ticks
    CURSOR_COLOR = "#00ff88"
    BG           = "#0a0a0a"

    _HELP_DEFAULT = ("click beat = toggle FLASH   •   right-click = hide/show tick   •   "
                     "SPACE = play/pause   •   ← → = seek ±2s   •   "
                     "Ctrl+wheel = zoom   •   wheel = pan   •   + / − / 0 = zoom / fit")
    _HELP_PICKING = ("▶ CLICK a tick to mark it as a FLASH   "
                     "(auto-fills every 4th in BOTH directions)   •   Esc = cancel")
    _HELP_DRAWING = ("✎ LEFT-CLICK on the waveform to ADD a new tick   "
                     "(too close to an existing tick → skipped)   •   Esc = exit draw mode")

    def __init__(self, audio_path, record, on_saved=None):
        self.audio_path = audio_path
        self.record = record
        self.flashes = set(record.get("flashes", default_flashes(len(record.get("beats", [])))))
        self.hidden  = set(record.get("hidden", []))
        self.playing = False
        # ── Playback state machine ──
        #   _pos          — our tracked position in the track (seconds). Canonical
        #                   while paused/stopped; derived from get_pos() while playing.
        #   _play_start   — the offset we most recently passed to pygame.play(start=…).
        #                   get_pos() is relative to that call, so current audio time =
        #                   _play_start + get_pos() / 1000.
        #   _was_paused   — true if the last break in playback was pause() (so the
        #                   next Space should unpause), false if it was stop() / seek()
        #                   (next Space should play(start=_pos)). pygame.unpause()
        #                   resumes from pause point; it can't seek.
        self._pos         = 0.0
        self._play_start  = 0.0
        self._was_paused  = False
        # Pick-mode: "click a single beat and auto-fill every-4th from there."
        # While armed, the next canvas click becomes the downbeat seed instead
        # of a plain flash toggle. Snapshot lets Esc restore the prior flashes.
        self._picking_flash    = False
        self._pick_snapshot    = None
        # Draw-mode: "left-click anywhere on the waveform adds a tick there".
        # Does not auto-exit — stays active so the user can drop several ticks
        # in a row (useful when listening for off-grid hits). Esc exits.
        self._drawing          = False
        self.on_saved = on_saved

        # Load audio for waveform view (downsampled later for plotting).
        self.y, self.sr = librosa.load(str(audio_path), sr=None, mono=True)
        self.duration = len(self.y) / self.sr

        # View state — Ctrl+wheel zooms horizontally, plain wheel pans.
        # FL-Studio-ish: cursor-centric zoom, smooth auto-follow while playing.
        self.view_xlim = (0.0, self.duration)
        self.autopan   = True

        # Tell pygame to load the file (it streams from disk).
        pygame.mixer.music.load(str(audio_path))

        self._build_ui()
        self._draw()
        self._tick_id = self.win.after(60, self._tick)

    # ── layout ────────────────────────────────────────────────

    def _build_ui(self):
        self.win = tk.Toplevel()
        self.win.title(f"Refine — {self.record['tid']}")
        self.win.geometry("1200x620")
        self.win.configure(bg=self.BG)
        self.win.protocol("WM_DELETE_WINDOW", self._cancel)

        # Top info bar
        top = tk.Frame(self.win, bg=self.BG)
        top.pack(side=tk.TOP, fill=tk.X, padx=10, pady=6)
        seg_note = f"  •  {len(self.record['segments'])} segments" if len(self.record["segments"]) > 1 else ""
        info = (f"{self.record['tid']}   "
                f"{self.record['bpm']:.1f} BPM{seg_note}   "
                f"key {self.record['key']}   "
                f"{len(self.record['beats'])} beats")
        tk.Label(top, text=info, fg="#ddd", bg=self.BG, font=("Courier New", 10, "bold")).pack(side=tk.LEFT)
        self.help_label = tk.Label(top,
                 text=self._HELP_DEFAULT,
                 fg="#777", bg=self.BG, font=("Courier New", 8))
        self.help_label.pack(side=tk.RIGHT)

        # Matplotlib figure
        self.fig = Figure(figsize=(11, 4), dpi=100, facecolor=self.BG)
        self.ax  = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.win)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10)
        self.canvas.mpl_connect("button_press_event", self._on_click)
        self.canvas.mpl_connect("scroll_event",       self._on_scroll)

        # Controls
        ctrl = tk.Frame(self.win, bg=self.BG)
        ctrl.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=8)

        self.btn_play = tk.Button(ctrl, text="▶  PLAY", width=10, command=self._toggle_play,
                                  bg="#222", fg="#ddd", activebackground="#333", activeforeground="#fff",
                                  relief="flat")
        self.btn_play.pack(side=tk.LEFT, padx=3)
        tk.Button(ctrl, text="■  STOP", width=8, command=self._stop,
                  bg="#222", fg="#ddd", activebackground="#333", activeforeground="#fff",
                  relief="flat").pack(side=tk.LEFT, padx=3)

        self.time_label = tk.Label(ctrl, text="0:00 / 0:00", fg="#999", bg=self.BG,
                                   font=("Courier New", 10))
        self.time_label.pack(side=tk.LEFT, padx=14)

        tk.Button(ctrl, text="reset: every 4th", command=self._reset_default,
                  bg="#222", fg="#aaa", relief="flat").pack(side=tk.LEFT, padx=3)
        tk.Button(ctrl, text="clear all", command=self._clear_flashes,
                  bg="#222", fg="#aaa", relief="flat").pack(side=tk.LEFT, padx=3)
        # Pick-then-fill: clears flashes, arms the next canvas click as the
        # downbeat seed, then auto-fills every-4th (both directions). Esc bails.
        tk.Button(ctrl, text="pick 1st flash ▸", command=self._arm_pick_flash,
                  bg="#2a1a0e", fg="#ffb070", activebackground="#3a220f",
                  activeforeground="#ffb070", relief="flat").pack(side=tk.LEFT, padx=3)
        # Draw-mode: left-click to add ticks anywhere on the waveform. Useful
        # when the analyzer missed an obvious hit. Stays armed until Esc.
        self.btn_draw = tk.Button(ctrl, text="draw tick ✎", command=self._toggle_draw,
                  bg="#0e1a2a", fg="#66aaff", activebackground="#0f223a",
                  activeforeground="#66aaff", relief="flat")
        self.btn_draw.pack(side=tk.LEFT, padx=3)
        # Unhide everything — right-clicks are reversible one-by-one, but when
        # the user wants to start over this is the escape hatch.
        tk.Button(ctrl, text="show hidden", command=self._show_all_hidden,
                  bg="#222", fg="#aaa", relief="flat").pack(side=tk.LEFT, padx=3)

        # Zoom controls — FL-Studio style. Buttons + keyboard shortcuts + mouse.
        tk.Label(ctrl, text="│", fg="#333", bg=self.BG,
                 font=("TkDefaultFont", 10)).pack(side=tk.LEFT, padx=6)
        tk.Button(ctrl, text="－", width=3, command=self._zoom_out,
                  bg="#222", fg="#ddd", relief="flat",
                  font=("TkDefaultFont", 10, "bold")).pack(side=tk.LEFT, padx=1)
        tk.Button(ctrl, text="＋", width=3, command=self._zoom_in,
                  bg="#222", fg="#ddd", relief="flat",
                  font=("TkDefaultFont", 10, "bold")).pack(side=tk.LEFT, padx=1)
        tk.Button(ctrl, text="fit", width=4, command=self._zoom_fit,
                  bg="#222", fg="#aaa", relief="flat").pack(side=tk.LEFT, padx=3)

        tk.Button(ctrl, text="CANCEL", width=9, command=self._cancel,
                  bg="#222", fg="#aaa", relief="flat").pack(side=tk.RIGHT, padx=3)
        tk.Button(ctrl, text="SAVE", width=9, command=self._save,
                  bg=self.BEAT_COLOR, fg="#000", activebackground="#ffb070", relief="flat",
                  font=("TkDefaultFont", 9, "bold")).pack(side=tk.RIGHT, padx=3)

        # Keyboard shortcuts
        self.win.bind("<space>",   lambda e: self._toggle_play())
        self.win.bind("s",         lambda e: self._stop())
        # Esc first cancels pick-flash mode (if armed), only then closes the window.
        self.win.bind("<Escape>",  lambda e: self._on_escape())
        # Zoom + pan keyboard parity with FL Studio-style mouse ops.
        self.win.bind("<plus>",    lambda e: self._zoom_in())
        self.win.bind("<equal>",   lambda e: self._zoom_in())
        self.win.bind("<minus>",   lambda e: self._zoom_out())
        self.win.bind("<Key-0>",   lambda e: self._zoom_fit())
        # Arrow keys = audio seek (±2 s). The view follows the playhead if
        # zoomed in (see _seek's autopan clause). Use mouse wheel for panning
        # the view without moving the playhead.
        self.win.bind("<Left>",    lambda e: self._seek(-2.0))
        self.win.bind("<Right>",   lambda e: self._seek(+2.0))
        self.win.focus_set()

    # ── drawing ───────────────────────────────────────────────

    def _draw(self):
        self.ax.clear()
        self.ax.set_facecolor(self.BG)
        for spine in self.ax.spines.values():
            spine.set_color("#333")
        self.ax.tick_params(axis="both", colors="#888", labelsize=8)
        self.ax.set_xlabel("time (s)", color="#888", fontsize=8)

        # Downsample the waveform for plotting — we don't need per-sample detail.
        target_points = 6000
        step = max(1, len(self.y) // target_points)
        y_sub = self.y[::step]
        t_sub = np.linspace(0, self.duration, len(y_sub))
        self.ax.plot(t_sub, y_sub, color=self.WAVE_COLOR, linewidth=0.5)
        # Preserve the user's current zoom/pan across redraws — without this,
        # every click-to-toggle would reset the view back to "fit all".
        self.ax.set_xlim(*self.view_xlim)
        self.ax.set_ylim(-1.2, 1.5)

        # Beat ticks — three tiers for visual separation:
        #   hidden  = faded grey dashed stub  (won't show on the Player)
        #   regular = dim orange short tick
        #   flash   = bright orange tall tick + ▼ marker
        beats = self.record["beats"]
        hidden_idx = sorted(i for i in self.hidden  if 0 <= i < len(beats))
        flash_idx  = sorted(i for i in self.flashes if 0 <= i < len(beats) and i not in self.hidden)
        reg_idx    = [i for i in range(len(beats))
                      if i not in self.flashes and i not in self.hidden]

        if hidden_idx:
            xs = [beats[i] for i in hidden_idx]
            self.ax.vlines(xs, 0.78, 0.92, colors=self.HIDDEN_COLOR,
                           linewidth=1.0, linestyles="dashed", alpha=0.55)
        if reg_idx:
            xs = [beats[i] for i in reg_idx]
            self.ax.vlines(xs, 0.6, 1.05, colors=self.BEAT_COLOR, alpha=0.45, linewidth=1.0)
        if flash_idx:
            xs = [beats[i] for i in flash_idx]
            self.ax.vlines(xs, 0.3, 1.4, colors=self.FLASH_COLOR, linewidth=2.2)
            # Dot on top for easy visual count
            self.ax.scatter(xs, [1.42] * len(xs), color=self.FLASH_COLOR, s=18, marker="v")

        # Segment boundaries — vertical dashed lines if multi-segment.
        if len(self.record["segments"]) > 1:
            for start, bpm in self.record["segments"]:
                if start > 0:
                    self.ax.axvline(start, color="#66aaff", linestyle="--", alpha=0.4, linewidth=1)
                    self.ax.text(start + 0.5, -1.1, f"{bpm:.0f}", color="#66aaff", fontsize=8, alpha=0.7)

        # Playback cursor.
        self._cursor = self.ax.axvline(0, color=self.CURSOR_COLOR, linewidth=1, alpha=0.9)

        # Status line in the upper-right corner — flash / hidden / total.
        status = f"{len(flash_idx)} flash"
        if hidden_idx:
            status += f" · {len(hidden_idx)} hidden"
        status += f" · {len(beats)} beats"
        self._count_text = self.ax.text(
            0.99, 0.97, status,
            transform=self.ax.transAxes, ha="right", va="top",
            color="#aaa", fontsize=9, family="monospace",
        )

        self.canvas.draw_idle()

    # ── interaction ───────────────────────────────────────────

    def _on_click(self, event):
        if event.inaxes != self.ax or event.xdata is None:
            return
        beats = self.record["beats"]
        click_t = event.xdata
        is_right = (getattr(event, "button", 1) == 3)

        # ── Draw-mode: LEFT-click ADDS a new tick anywhere on the waveform.
        #   Stays armed — user can drop several in a row. Right-click still
        #   falls through to the hide/show handler below.
        if self._drawing and not is_right:
            self._add_beat(click_t)
            return

        if not beats:
            return
        # Nearest-beat lookup — tolerance proportional to local beat spacing.
        i = min(range(len(beats)), key=lambda i: abs(beats[i] - click_t))
        if 0 < i < len(beats) - 1:
            tol = max(0.08, min(beats[i] - beats[i - 1], beats[i + 1] - beats[i]) * 0.5)
        else:
            tol = 0.25
        if abs(beats[i] - click_t) > tol:
            return

        # ── Right-click: toggle HIDDEN on nearest tick. Applies in every mode.
        # Hidden beats stay in the beats list (analysis still uses them) but
        # render as 0-flag on the Player — no color mark, no flash.
        if is_right:
            if i in self.hidden:
                self.hidden.discard(i)
            else:
                self.hidden.add(i)
                self.flashes.discard(i)   # hidden beats can't also flash
            self._draw()
            return

        # ── Pick-flash mode: seed auto-fill in BOTH directions.
        # Starting at beat `i`, fill every 4th index the same modulo-4 class
        # in the full track — so earlier indices (i-4, i-8, …) AND later ones
        # (i+4, i+8, …) all become flashes. One click = one full bar grid.
        if self._picking_flash:
            step = DEFAULT_FLASH_STEP
            self.flashes = set(range(i % step, len(beats), step))
            # Hidden beats cannot flash — respect existing hides.
            self.flashes -= self.hidden
            self._exit_pick_flash()
            self._draw()
            return

        # ── Normal: toggle FLASH on the clicked beat. Hidden ticks can't flash.
        if i in self.hidden:
            return
        if i in self.flashes:
            self.flashes.discard(i)
        else:
            self.flashes.add(i)
        self._draw()

    # ── beat mutation helpers ─────────────────────────────────

    def _add_beat(self, t_sec):
        """Insert a new beat at `t_sec`, preserving sort order. Shift existing
        flash / hidden indices that sit after the insertion point.
        Rejected when the click is within 60 ms of an existing beat to avoid
        duplicate / near-duplicate entries."""
        beats = self.record["beats"]
        # Clamp to a valid range.
        t_sec = max(0.0, min(self.duration - 0.005, float(t_sec)))
        # Reject near-duplicates (60 ms tolerance — tighter than click-toggle
        # tolerance since the user is deliberately trying to place a new beat).
        if beats:
            nearest = min(beats, key=lambda b: abs(b - t_sec))
            if abs(nearest - t_sec) < 0.06:
                return
        # Find insertion index (beats stay sorted ascending).
        insert_at = 0
        while insert_at < len(beats) and beats[insert_at] < t_sec:
            insert_at += 1
        # Shift flash + hidden indices ≥ insertion point by +1.
        self.flashes = {(f + 1 if f >= insert_at else f) for f in self.flashes}
        self.hidden  = {(h + 1 if h >= insert_at else h) for h in self.hidden}
        beats.insert(insert_at, t_sec)
        self._draw()

    def _show_all_hidden(self):
        """Un-hide every beat."""
        if not self.hidden:
            return
        self.hidden = set()
        self._draw()

    # ── pick-flash mode helpers ───────────────────────────────

    def _arm_pick_flash(self):
        """Enter pick-mode: snapshot current flashes, clear them all, wait
        for the next canvas click to seed the every-4th auto-fill in both
        directions."""
        if self._drawing:
            self._exit_draw()
        self._pick_snapshot = set(self.flashes)
        self.flashes = set()
        self._picking_flash = True
        self.help_label.config(text=self._HELP_PICKING, fg=self.FLASH_COLOR,
                               font=("Courier New", 9, "bold"))
        # Crosshair cursor tells the user "you're picking something now".
        try:
            self.canvas.get_tk_widget().configure(cursor="crosshair")
        except Exception:
            pass
        self._draw()

    def _exit_pick_flash(self):
        """Leave pick-mode. Called after a successful pick OR on Esc cancel."""
        self._picking_flash = False
        self._pick_snapshot = None
        self.help_label.config(text=self._HELP_DEFAULT, fg="#777",
                               font=("Courier New", 8))
        try:
            self.canvas.get_tk_widget().configure(cursor="")
        except Exception:
            pass

    def _cancel_pick_flash(self):
        """Esc while armed: restore the flashes the user had before arming."""
        if not self._picking_flash:
            return
        if self._pick_snapshot is not None:
            self.flashes = set(self._pick_snapshot)
        self._exit_pick_flash()
        self._draw()

    def _on_escape(self):
        """Esc cascade: pick-mode → draw-mode → close window."""
        if self._picking_flash:
            self._cancel_pick_flash()
        elif self._drawing:
            self._exit_draw()
        else:
            self._cancel()

    # ── draw-mode helpers ─────────────────────────────────────

    def _toggle_draw(self):
        """The 'draw tick ✎' button. Flips draw-mode on/off — any other active
        exclusive mode (pick-flash) is cancelled first."""
        if self._picking_flash:
            self._cancel_pick_flash()
        if self._drawing:
            self._exit_draw()
        else:
            self._arm_draw()

    def _arm_draw(self):
        self._drawing = True
        self.help_label.config(text=self._HELP_DRAWING, fg=self.DRAW_COLOR,
                               font=("Courier New", 9, "bold"))
        try:
            self.canvas.get_tk_widget().configure(cursor="pencil")
        except Exception:
            pass
        self.btn_draw.config(bg="#1a3a5a", fg="#ffffff")

    def _exit_draw(self):
        self._drawing = False
        self.help_label.config(text=self._HELP_DEFAULT, fg="#777",
                               font=("Courier New", 8))
        try:
            self.canvas.get_tk_widget().configure(cursor="")
        except Exception:
            pass
        self.btn_draw.config(bg="#0e1a2a", fg="#66aaff")

    def _current_time(self):
        """Return the current audio playback time in seconds."""
        if self.playing:
            ms = pygame.mixer.music.get_pos()
            if ms < 0:
                return self._play_start
            return self._play_start + ms / 1000.0
        return self._pos

    def _toggle_play(self):
        """Space / PLAY button. Proper pause-and-resume, never a hidden restart."""
        if self.playing:
            # Pause: freeze the position so the next unpause resumes from here.
            self._pos = self._current_time()
            pygame.mixer.music.pause()
            self._was_paused = True
            self.playing = False
            self.btn_play.config(text="▶  PLAY")
        else:
            if self._was_paused:
                # Strict resume: pygame continues from the exact pause point.
                pygame.mixer.music.unpause()
            else:
                # Fresh start — from 0 on first press, or from the seeked-to
                # position if the user arrow-seeked while stopped/paused.
                pygame.mixer.music.play(start=float(self._pos))
                self._play_start = self._pos
            self._was_paused = False
            self.playing = True
            self.btn_play.config(text="⏸  PAUSE")

    def _stop(self):
        pygame.mixer.music.stop()
        self.playing = False
        self._pos = 0.0
        self._play_start = 0.0
        self._was_paused = False
        self.btn_play.config(text="▶  PLAY")
        self._cursor.set_xdata([0, 0])
        self.time_label.config(text=f"0:00 / {self._fmt(self.duration)}")
        self.canvas.draw_idle()

    def _seek(self, delta_s):
        """Arrow-key seek — jump ±N seconds. Preserves playing/paused state:
        seek while playing continues playing from the new position; seek while
        paused or stopped stays halted at the new position (next Space plays
        from there). Clamped to [0, duration − 0.1]."""
        new_pos = max(0.0, min(self.duration - 0.1, self._current_time() + delta_s))
        self._pos = new_pos
        # pygame.mixer.music has no seek primitive; we stop + replay at offset.
        pygame.mixer.music.stop()
        if self.playing:
            pygame.mixer.music.play(start=float(new_pos))
            self._play_start = new_pos
            # _was_paused stays False (we started fresh with play()).
        else:
            # Stay halted at new_pos. unpause() wouldn't honor the seek.
            self._was_paused = False
            self._play_start = new_pos
        # Reflect the new position visually right away — don't wait for _tick.
        self._cursor.set_xdata([new_pos, new_pos])
        self.time_label.config(text=f"{self._fmt(new_pos)} / {self._fmt(self.duration)}")
        # Keep the seeked position inside the current view if zoomed in.
        lo, hi = self.view_xlim
        if self.autopan and (hi - lo) < self.duration - 0.01:
            if new_pos < lo or new_pos > hi:
                span = hi - lo
                new_lo = max(0.0, new_pos - span * 0.35)
                new_hi = min(self.duration, new_lo + span)
                if new_hi - new_lo < span:
                    new_lo = max(0.0, new_hi - span)
                self._set_view(new_lo, new_hi, redraw=False)
        self.canvas.draw_idle()

    def _tick(self):
        if self.playing:
            t = self._current_time()
            if t >= self.duration - 0.02:
                # Natural end — reset to stopped state so the next Space starts fresh.
                self._stop()
            else:
                self._cursor.set_xdata([t, t])
                self.time_label.config(text=f"{self._fmt(t)} / {self._fmt(self.duration)}")
                # Auto-pan: when zoomed in and the cursor nears the right
                # edge of the view, slide the window so playback stays on
                # screen. Only runs when zoomed (view narrower than track).
                lo, hi = self.view_xlim
                span = hi - lo
                if self.autopan and span < self.duration - 0.01:
                    edge = lo + span * 0.85   # start panning at 85% across
                    if t >= edge:
                        new_lo = max(0.0, t - span * 0.35)
                        new_hi = min(self.duration, new_lo + span)
                        if new_hi - new_lo < span:
                            new_lo = max(0.0, new_hi - span)
                        self._set_view(new_lo, new_hi, redraw=False)
                self.canvas.draw_idle()
        self._tick_id = self.win.after(60, self._tick)

    @staticmethod
    def _fmt(s):
        return f"{int(s // 60)}:{int(s % 60):02d}"

    def _reset_default(self):
        self.flashes = set(default_flashes(len(self.record["beats"])))
        self._draw()

    def _clear_flashes(self):
        self.flashes = set()
        self._draw()

    # ── zoom / pan ────────────────────────────────────────────
    # View is expressed as (lo, hi) seconds on the x-axis. All zoom / pan
    # ops go through _set_view so clamping + redraw live in one place.

    MIN_VIEW_SPAN_S = 1.5   # don't zoom in past ~1.5 s visible (finger-clickable)
    ZOOM_STEP       = 1.35  # per button-press / per wheel notch

    def _set_view(self, lo, hi, redraw=True):
        """Clamp to [0, duration] and at least MIN_VIEW_SPAN_S wide; push to axes."""
        lo = max(0.0, float(lo))
        hi = min(self.duration, float(hi))
        span = hi - lo
        if span < self.MIN_VIEW_SPAN_S:
            mid = (lo + hi) / 2
            half = self.MIN_VIEW_SPAN_S / 2
            lo = max(0.0, mid - half)
            hi = min(self.duration, lo + self.MIN_VIEW_SPAN_S)
            if hi - lo < self.MIN_VIEW_SPAN_S:
                lo = max(0.0, hi - self.MIN_VIEW_SPAN_S)
        if hi - lo > self.duration:
            lo, hi = 0.0, self.duration
        self.view_xlim = (lo, hi)
        self.ax.set_xlim(lo, hi)
        if redraw:
            self.canvas.draw_idle()

    def _zoom_at(self, factor, center_t):
        """Multiply the zoom by `factor` (>1 zoom in, <1 zoom out), keeping
        `center_t` anchored in place (FL Studio-style cursor-centric zoom)."""
        lo, hi = self.view_xlim
        span = hi - lo
        new_span = span / factor
        # Preserve the relative x-position of center_t within the view so the
        # mouse cursor stays over the same audio frame after the zoom.
        frac = (center_t - lo) / span if span > 0 else 0.5
        new_lo = center_t - frac * new_span
        new_hi = new_lo + new_span
        self._set_view(new_lo, new_hi)

    def _on_scroll(self, event):
        """Ctrl + wheel → zoom at cursor. Plain wheel → horizontal pan."""
        if event.inaxes != self.ax:
            return
        # matplotlib's scroll_event reports modifier keys via event.key
        # ('control', 'shift', 'ctrl+shift', etc.). Guard for None.
        mod = (event.key or "").lower()
        direction_up = (event.button == "up")
        if "control" in mod or "ctrl" in mod:
            if event.xdata is None:
                return
            factor = self.ZOOM_STEP if direction_up else 1.0 / self.ZOOM_STEP
            self._zoom_at(factor, event.xdata)
        else:
            # Plain wheel: pan ~10% of the current view per notch.
            lo, hi = self.view_xlim
            span = hi - lo
            delta = span * 0.12 * (-1 if direction_up else 1)
            self._set_view(lo + delta, hi + delta)

    def _zoom_in(self):
        """Button / keyboard zoom — centered on the playback cursor if
        playing, else on the current view center."""
        lo, hi = self.view_xlim
        if self.playing:
            try:
                center = pygame.mixer.music.get_pos() / 1000.0
                if center < lo or center > hi:
                    center = (lo + hi) / 2
            except Exception:
                center = (lo + hi) / 2
        else:
            center = (lo + hi) / 2
        self._zoom_at(self.ZOOM_STEP, center)

    def _zoom_out(self):
        lo, hi = self.view_xlim
        self._zoom_at(1.0 / self.ZOOM_STEP, (lo + hi) / 2)

    def _zoom_fit(self):
        self._set_view(0.0, self.duration)

    def _pan(self, fraction):
        """Keyboard-driven pan: fraction of the current view span."""
        lo, hi = self.view_xlim
        span = hi - lo
        delta = span * fraction
        self._set_view(lo + delta, hi + delta)

    def _save(self):
        # Keep flash / hidden sets disjoint at save time (defensive — _on_click
        # already maintains this, but a one-line guard here protects against
        # any future code path that doesn't).
        self.flashes -= self.hidden
        self.record["flashes"] = sorted(self.flashes)
        self.record["hidden"]  = sorted(self.hidden)
        # record["beats"] was mutated in-place by _add_beat, no extra work.
        save_record(self.record)
        messagebox.showinfo("Saved", f"Updated {self.record['tid']} in {OUT.name}")
        if self.on_saved:
            self.on_saved(self.record)
        self._teardown()

    def _cancel(self):
        self._teardown()

    def _teardown(self):
        try:
            self.win.after_cancel(self._tick_id)
        except Exception:
            pass
        try:
            pygame.mixer.music.stop()
        except Exception:
            pass
        self.win.destroy()


# ── Track chooser ─────────────────────────────────────────────

def show_refine_chooser(parent):
    records = load_all_records()
    if not records:
        messagebox.showwarning("No data",
            f"{OUT.name} doesn't exist yet.\n\nRun 'Auto Run All' first to "
            "analyze your audio files, then come back to Refine.")
        return

    win = tk.Toplevel(parent)
    win.title("Refine — pick a track")
    win.geometry("580x440")
    win.configure(bg="#0a0a0a")

    tk.Label(win, text="PICK A TRACK TO REFINE",
             fg="#ddd", bg="#0a0a0a", font=("TkDefaultFont", 10, "bold"),
             pady=12).pack(side=tk.TOP)

    list_frame = tk.Frame(win, bg="#0a0a0a")
    list_frame.pack(fill=tk.BOTH, expand=True, padx=14, pady=(0, 8))
    scroll = tk.Scrollbar(list_frame)
    scroll.pack(side=tk.RIGHT, fill=tk.Y)
    listbox = tk.Listbox(
        list_frame, font=("Courier New", 10),
        bg="#111", fg="#ddd", selectbackground="#de7029", selectforeground="#000",
        activestyle="none", relief="flat", highlightthickness=0,
        yscrollcommand=scroll.set,
    )
    listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    scroll.config(command=listbox.yview)

    ids_sorted = sorted(records.keys())
    for tid in ids_sorted:
        rec = records[tid]
        default_flashes_set = set(default_flashes(len(rec["beats"])))
        is_custom = (set(rec["flashes"]) != default_flashes_set) or bool(rec.get("hidden"))
        tag = "◆" if is_custom else " "
        seg = f"+{len(rec['segments'])-1}seg" if len(rec["segments"]) > 1 else "     "
        hidden_count = len(rec.get("hidden", []))
        hidden_tag = f" · {hidden_count:>2}h" if hidden_count else "     "
        listbox.insert(tk.END,
            f" {tag}  {tid:<18}  {rec['bpm']:6.1f} BPM  {rec['key']:<3}  "
            f"{len(rec['beats']):4d} beats  {len(rec['flashes']):3d} flashes"
            f"{hidden_tag}  {seg}")

    # Legend
    tk.Label(win, text="◆ = custom flash pattern (refined)    |    blank = default every-4th",
             fg="#777", bg="#0a0a0a", font=("Courier New", 8)).pack(side=tk.TOP, pady=(0, 6))

    def open_selected(_e=None):
        sel = listbox.curselection()
        if not sel:
            return
        tid = ids_sorted[sel[0]]
        audio = find_audio_for(tid)
        if not audio:
            messagebox.showerror("Audio missing",
                f"No audio file for {tid} in {HERE.name}/.\n"
                "The record exists in cf-rhythm.dat but the audio has been removed or renamed.")
            return
        win.destroy()
        RefineWindow(audio, records[tid])

    btns = tk.Frame(win, bg="#0a0a0a")
    btns.pack(side=tk.BOTTOM, fill=tk.X, padx=14, pady=10)
    tk.Button(btns, text="CANCEL", command=win.destroy,
              bg="#222", fg="#aaa", relief="flat", width=10).pack(side=tk.RIGHT, padx=3)
    tk.Button(btns, text="REFINE →", command=open_selected,
              bg="#de7029", fg="#000", relief="flat", width=12,
              font=("TkDefaultFont", 9, "bold")).pack(side=tk.RIGHT, padx=3)
    listbox.bind("<Double-Button-1>", open_selected)
    listbox.bind("<Return>", open_selected)


# ── Main menu ─────────────────────────────────────────────────

def show_main_menu():
    _import_gui()
    root = tk.Tk()
    root.title("analyze-beats")
    root.geometry("420x300")
    root.configure(bg="#0a0a0a")

    tk.Label(root, text="WSI RADIO", fg="#de7029", bg="#0a0a0a",
             font=("TkDefaultFont", 14, "bold"), pady=4).pack(pady=(24, 0))
    tk.Label(root, text="beat analyzer", fg="#888", bg="#0a0a0a",
             font=("Courier New", 9)).pack(pady=(0, 16))

    def do_auto():
        root.withdraw()
        try:
            auto_run()
            messagebox.showinfo("Done", f"Analysis complete.\nSee {OUT.name}.")
        finally:
            root.deiconify()

    def do_refine():
        root.withdraw()
        show_refine_chooser(root)
        root.after(100, root.deiconify)  # brings the menu back after chooser closes

    btn_kwargs = dict(bg="#de7029", fg="#000", activebackground="#ffb070",
                      relief="flat", font=("TkDefaultFont", 10, "bold"), width=22, pady=6)
    sub_kwargs = dict(bg="#222", fg="#ddd", activebackground="#333", activeforeground="#fff",
                      relief="flat", font=("TkDefaultFont", 9), width=22, pady=4)

    tk.Button(root, text="AUTO  —  SCAN ALL",   command=do_auto,   **btn_kwargs).pack(pady=4)
    tk.Button(root, text="REFINE  —  PICK ONE", command=do_refine, **btn_kwargs).pack(pady=4)
    tk.Button(root, text="quit",                command=root.destroy, **sub_kwargs).pack(pady=(18, 0))

    tk.Label(root, text=f"output: {OUT.name}", fg="#555", bg="#0a0a0a",
             font=("Courier New", 8)).pack(side=tk.BOTTOM, pady=8)

    root.mainloop()


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(description="Offline BPM/key/beat analysis for WSI Radio tracks.")
    ap.add_argument("--auto",   action="store_true", help="Headless: run Auto on all tracks, no menu.")
    ap.add_argument("--refine", metavar="ID", nargs="?", const="",
                    help="Open Refine UI (optionally pre-selecting track-id).")
    ap.add_argument("--force",  action="store_true", help="(--auto) re-analyze every track.")
    ap.add_argument("--track",  metavar="ID", help="(--auto) analyze only this track-id.")
    args = ap.parse_args()

    if args.auto:
        return auto_run(force=args.force, only_track=args.track)

    if args.refine is not None:
        _import_gui()
        root = tk.Tk()
        root.withdraw()
        if args.refine:
            records = load_all_records()
            if args.refine not in records:
                print(f"track-id '{args.refine}' not found in {OUT.name}", file=sys.stderr)
                return 1
            audio = find_audio_for(args.refine)
            if not audio:
                print(f"no audio file for {args.refine} in {HERE.name}/", file=sys.stderr)
                return 1
            RefineWindow(audio, records[args.refine])
        else:
            show_refine_chooser(root)
        root.mainloop()
        return 0

    # Default: interactive menu.
    show_main_menu()
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
