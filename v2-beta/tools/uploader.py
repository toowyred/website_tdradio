#!/usr/bin/env python3
"""
TD Radio — Uploader + Marker tools

One Tkinter app with two tabs:

  UPLOAD — drop an audio file → auto-rename next td-audio0NN → convert to
           normal (192 kbps MP3) and/or high (WAV) → append a scaffolded
           block to cf-signal.dat. Source file is never moved/deleted.

  MARK   — pick an existing track → play → tap Space at moments → capture
           timestamps for signal / print / skip / origin etc. → review in
           a table → append to cf-signal.dat. Auto-backup before writing.

Dependencies:
  - Python 3.10+
  - Tkinter (stdlib)
  - ffmpeg on PATH             (required for UPLOAD tab conversion)
  - pygame  (optional)         — enables MARK tab playback.
                                 Install:  pip install pygame

The script auto-detects the radio/ folder (parent of radio/tools/).
"""

from __future__ import annotations

import os
import re
import sys
import time
import shutil
import pathlib
import subprocess
import threading
from datetime import datetime
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

# ── Paths ──────────────────────────────────────────────────────────────
HERE        = pathlib.Path(__file__).resolve().parent
RADIO_DIR   = HERE.parent                          # radio/
TRACKS_DIR  = RADIO_DIR / 'tracks'
SIGNAL_PATH = RADIO_DIR / 'cf-signal.dat'
BACKUP_DIR  = RADIO_DIR / 'tools' / '_backups'     # rotating signal.dat backups

# ── External tool check ────────────────────────────────────────────────
FFMPEG = shutil.which('ffmpeg')

# ── Optional pygame for MARK tab playback ──────────────────────────────
try:
    import pygame
    pygame.mixer.init()
    PYGAME_OK = True
except Exception as _e:
    pygame = None
    PYGAME_OK = False


# ══════════════════════════════════════════════════════════════════════
# HELPERS — filesystem, naming, signal scaffold
# ══════════════════════════════════════════════════════════════════════
NAME_RE = re.compile(r'^td-audio(\d{3})-(normal|high|off)\.(mp3|wav)$', re.IGNORECASE)


def next_audio_number() -> int:
    """Scan tracks/ for existing td-audio0NN files and return NN+1 (min 1)."""
    used = set()
    if TRACKS_DIR.is_dir():
        for f in TRACKS_DIR.iterdir():
            m = NAME_RE.match(f.name)
            if m:
                used.add(int(m.group(1)))
    return max(used) + 1 if used else 1


def next_audio_name() -> str:
    return f'td-audio{next_audio_number():03d}'


def list_existing_tracks() -> list[str]:
    """Return unique track stems like 'td-audio003' present in tracks/."""
    stems = set()
    if TRACKS_DIR.is_dir():
        for f in TRACKS_DIR.iterdir():
            m = NAME_RE.match(f.name)
            if m:
                stems.add(f'td-audio{m.group(1)}')
    return sorted(stems)


def track_file_for_stem(stem: str) -> pathlib.Path | None:
    """Return the first playable file for a stem (prefer normal.mp3)."""
    if not TRACKS_DIR.is_dir():
        return None
    # preference order: normal.mp3, high.wav, off.mp3
    for suffix in ('-normal.mp3', '-high.wav', '-off.mp3'):
        p = TRACKS_DIR / f'{stem}{suffix}'
        if p.exists():
            return p
    return None


def ensure_backup() -> pathlib.Path | None:
    """Copy cf-signal.dat into tools/_backups/ with a timestamped name."""
    if not SIGNAL_PATH.exists():
        return None
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d-%H%M%S')
    dst = BACKUP_DIR / f'cf-signal.{ts}.dat.bak'
    shutil.copy2(SIGNAL_PATH, dst)
    return dst


# ── ffmpeg wrappers ────────────────────────────────────────────────────
def run_ffmpeg(args: list[str], log) -> bool:
    log('  $ ffmpeg ' + ' '.join(args[1:]))
    try:
        p = subprocess.run(
            args, capture_output=True, text=True, encoding='utf-8', errors='replace',
            creationflags=(subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0),
        )
    except FileNotFoundError:
        log('  ! ffmpeg not found on PATH')
        return False
    if p.returncode != 0:
        tail = (p.stderr or '').strip().splitlines()[-3:]
        for ln in tail:
            log('  ! ' + ln)
        return False
    return True


def convert_to_normal_mp3(src: pathlib.Path, dst: pathlib.Path, log) -> bool:
    """192 kbps CBR MP3, 44.1 kHz stereo, no Xing header (matches existing td-audio*)."""
    return run_ffmpeg([
        FFMPEG, '-y', '-i', str(src),
        '-codec:a', 'libmp3lame', '-b:a', '192k',
        '-ar', '44100', '-ac', '2',
        '-write_xing', '0',
        str(dst),
    ], log)


def make_high_wav(src: pathlib.Path, dst: pathlib.Path, log) -> bool:
    """If source is already WAV, copy as-is. Else transcode to 16-bit 44.1k stereo."""
    if src.suffix.lower() == '.wav':
        shutil.copy2(src, dst)
        log(f'  → copied source WAV → {dst.name}')
        return True
    return run_ffmpeg([
        FFMPEG, '-y', '-i', str(src),
        '-codec:a', 'pcm_s16le', '-ar', '44100', '-ac', '2',
        str(dst),
    ], log)


# ── cf-signal.dat scaffold + write ─────────────────────────────────────
def scaffold_block(name: str, signal: str, signal_offline: str,
                   srt_mode: str, srt_filename: str,
                   has_normal: bool, has_high: bool) -> str:
    """Produce a new signal.dat block matching the existing convention."""
    m = re.search(r'(\d+)$', name)
    num = m.group(1).zfill(3) if m else '???'
    lines: list[str] = []
    lines.append('')
    lines.append('=================================================================')
    lines.append('#')
    lines.append('#')
    lines.append(f'# {num}')
    lines.append('=================================================================')
    if has_high:
        lines.append(f'File: {name}-high.wav')
    if has_normal:
        lines.append(f'File: {name}-normal.mp3')
    lines.append(f'offline_file\t={name}-off.mp3')
    lines.append('WebAudio_track\t=')
    lines.append('WebAudio_artist\t=TWISTED DUALITY')
    lines.append('WebAudio_album\t=')
    lines.append('#')
    lines.append(f'# ── base values variables ─────────────────────────────{num}')
    lines.append(f'signal\t\t={signal}')
    lines.append(f'signal_offline\t={signal_offline}')
    lines.append('decoder\t\t=TWISTED DUALITY')
    lines.append('decoder_offline\t=TWISTED DUALITY')
    lines.append('status\t\t=noncompiled')
    lines.append('status_offline\t=unknown')
    lines.append('access\t\t=none')
    lines.append('access_offline\t=unknown')
    lines.append('origin\t\t=TWISTED DUALITY')
    lines.append('origin_offline\t=0')
    lines.append('vibe\t\t\t=')
    lines.append('vibe_offline\t=0')
    if srt_mode in ('0', '1'):
        srt_val = srt_mode if not srt_filename else f'{srt_mode}; {srt_filename}'
        lines.append(f'srt\t\t\t={srt_val}')
    lines.append('#')
    lines.append(f'# ── prints ────────────────────────────────────────────{num}')
    lines.append('')
    lines.append(f'# ── timeline ──────────────────────────────────────────{num}')
    lines.append('')
    lines.append('=================================================================')
    return '\n'.join(lines) + '\n'


def append_to_signal(block: str, log) -> bool:
    if not SIGNAL_PATH.exists():
        log(f'! cf-signal.dat not found at {SIGNAL_PATH}')
        return False
    backup = ensure_backup()
    if backup:
        log(f'  ✓ backup saved: {backup.name}')
    with open(SIGNAL_PATH, 'a', encoding='utf-8') as f:
        f.write(block)
    log('  ✓ appended block to cf-signal.dat')
    return True


def append_marks_to_track_block(track_stem: str, marks: list[dict], log) -> bool:
    """Merge new timeline lines into the existing block for track_stem.

    Strategy: find the block in cf-signal.dat containing `File: {stem}-*`.
    Insert each mark as a signal.dat line before the closing `=====` delimiter
    of that block. Backups first.
    """
    if not SIGNAL_PATH.exists():
        log(f'! cf-signal.dat not found at {SIGNAL_PATH}')
        return False

    with open(SIGNAL_PATH, 'r', encoding='utf-8') as f:
        text = f.read()

    # Find block delimiter positions
    delim_re = re.compile(r'={5,}')
    delims = [(m.start(), m.end()) for m in delim_re.finditer(text)]
    if len(delims) < 2:
        log('! not enough block delimiters to locate a track block')
        return False

    # Find the block containing "File: td-audio003-..." for our stem
    file_re = re.compile(rf'^File:\s*{re.escape(track_stem)}-', re.MULTILINE)
    file_match = file_re.search(text)
    if not file_match:
        log(f'! no block with File: {track_stem}-* found in cf-signal.dat')
        return False

    file_pos = file_match.start()
    # The closing delimiter is the first delim whose start is AFTER file_pos
    close_delim = next((d for d in delims if d[0] > file_pos), None)
    if not close_delim:
        log('! could not find closing ===== for that block')
        return False
    insert_at = close_delim[0]

    # Build the lines to insert
    mark_lines = []
    for m in marks:
        t = m['time']
        # Pretty-print integer seconds as integer, else one-decimal
        t_str = f'{int(round(t))}' if abs(t - round(t)) < 0.05 else f'{t:.1f}'
        if m['type'] == 'print':
            txt = m.get('value', '')
            duration = m.get('duration')
            dpart = f'; d={duration}' if duration else ''
            mark_lines.append(f'print(sec{t_str})\t={txt}{dpart}')
        elif m['type'] == 'skip':
            mark_lines.append(f'# (scan-skip entry @ sec{t_str}) — wire to scan-skip.dat later')
        else:
            val = m.get('value', '')
            mark_lines.append(f'{m["type"]}(sec{t_str})\t={val}')

    addition = '# ── marks appended ' + datetime.now().strftime('%Y-%m-%d %H:%M') + ' ──\n'
    addition += '\n'.join(mark_lines) + '\n'

    # Backup first
    backup = ensure_backup()
    if backup:
        log(f'  ✓ backup saved: {backup.name}')

    new_text = text[:insert_at] + addition + text[insert_at:]
    with open(SIGNAL_PATH, 'w', encoding='utf-8') as f:
        f.write(new_text)
    log(f'  ✓ inserted {len(marks)} mark line(s) into {track_stem} block')
    return True


# ══════════════════════════════════════════════════════════════════════
# UPLOAD TAB
# ══════════════════════════════════════════════════════════════════════
class UploadTab(ttk.Frame):
    def __init__(self, master):
        super().__init__(master, padding=12)
        self.source_path: pathlib.Path | None = None
        self._build()
        self._refresh_auto_name()

    def _build(self):
        # Source file row
        frame_src = ttk.LabelFrame(self, text='SOURCE AUDIO', padding=8)
        frame_src.pack(fill='x')
        self.lbl_src = ttk.Label(frame_src, text='(no file selected — drop one or browse)',
                                 foreground='#888')
        self.lbl_src.pack(side='left', fill='x', expand=True)
        ttk.Button(frame_src, text='Browse…', command=self.on_browse).pack(side='right')

        # Name + Signal fields
        frame_meta = ttk.LabelFrame(self, text='TRACK INFO', padding=8)
        frame_meta.pack(fill='x', pady=(10, 0))
        # Name
        row = ttk.Frame(frame_meta); row.pack(fill='x', pady=2)
        ttk.Label(row, text='Name:', width=18).pack(side='left')
        self.name_var = tk.StringVar(value='')
        ttk.Entry(row, textvariable=self.name_var).pack(side='left', fill='x', expand=True)
        ttk.Button(row, text='⟳', width=3, command=self._refresh_auto_name).pack(side='left', padx=(4, 0))
        # Signal
        row = ttk.Frame(frame_meta); row.pack(fill='x', pady=2)
        ttk.Label(row, text='Signal:', width=18).pack(side='left')
        self.signal_var = tk.StringVar(value='')
        ttk.Entry(row, textvariable=self.signal_var).pack(side='left', fill='x', expand=True)
        # Signal offline
        row = ttk.Frame(frame_meta); row.pack(fill='x', pady=2)
        ttk.Label(row, text='Signal (offline):', width=18).pack(side='left')
        self.signal_off_var = tk.StringVar(value='ID')
        ttk.Entry(row, textvariable=self.signal_off_var).pack(side='left', fill='x', expand=True)

        # SRT options
        row = ttk.Frame(frame_meta); row.pack(fill='x', pady=(6, 2))
        ttk.Label(row, text='SRT:', width=18).pack(side='left')
        self.srt_var = tk.StringVar(value='0')
        ttk.Radiobutton(row, text='off', variable=self.srt_var, value='0').pack(side='left')
        ttk.Radiobutton(row, text='on',  variable=self.srt_var, value='1').pack(side='left')
        self.srt_samename_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(row, text='same name', variable=self.srt_samename_var,
                        command=self._on_srt_toggle).pack(side='left', padx=(10, 4))
        self.srt_file_var = tk.StringVar(value='')
        self.srt_file_entry = ttk.Entry(row, textvariable=self.srt_file_var, state='disabled')
        self.srt_file_entry.pack(side='left', fill='x', expand=True)

        # Outputs
        frame_out = ttk.LabelFrame(self, text='OUTPUTS', padding=8)
        frame_out.pack(fill='x', pady=(10, 0))
        self.out_normal_var = tk.BooleanVar(value=True)
        self.out_high_var   = tk.BooleanVar(value=True)
        ttk.Checkbutton(frame_out, text='normal (192 kbps MP3)', variable=self.out_normal_var).pack(side='left')
        ttk.Checkbutton(frame_out, text='high (WAV)',            variable=self.out_high_var  ).pack(side='left', padx=(14, 0))
        self.append_block_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(frame_out, text='append block to cf-signal.dat',
                        variable=self.append_block_var).pack(side='right')

        # Action + log
        action_row = ttk.Frame(self); action_row.pack(fill='x', pady=(10, 4))
        self.btn_go = ttk.Button(action_row, text='CONVERT & APPEND', command=self.on_go)
        self.btn_go.pack(side='left')
        self.lbl_ffmpeg = ttk.Label(action_row, text='', foreground='#888')
        self.lbl_ffmpeg.pack(side='right')
        self._update_ffmpeg_status()

        frame_log = ttk.LabelFrame(self, text='LOG', padding=4)
        frame_log.pack(fill='both', expand=True, pady=(6, 0))
        self.log_txt = tk.Text(frame_log, height=14, bg='#111', fg='#e6e6e6',
                               insertbackground='#e6e6e6', font=('Courier New', 10))
        self.log_txt.pack(fill='both', expand=True)

        # Drag-and-drop via the standard dnd extension (tkinterdnd2 is optional)
        self.bind_all('<Control-o>', lambda e: self.on_browse())

    # ── helpers ──
    def _update_ffmpeg_status(self):
        if FFMPEG:
            self.lbl_ffmpeg.configure(text='ffmpeg: ok', foreground='#4caf50')
        else:
            self.lbl_ffmpeg.configure(text='ffmpeg: NOT FOUND on PATH', foreground='#f44336')
            self.btn_go.configure(state='disabled')

    def _refresh_auto_name(self):
        self.name_var.set(next_audio_name())

    def _on_srt_toggle(self):
        if self.srt_samename_var.get():
            self.srt_file_entry.configure(state='disabled')
            self.srt_file_var.set('')
        else:
            self.srt_file_entry.configure(state='normal')

    def log(self, line: str):
        self.log_txt.insert('end', line + '\n')
        self.log_txt.see('end')
        self.log_txt.update_idletasks()

    def on_browse(self):
        f = filedialog.askopenfilename(
            title='Pick an audio file',
            filetypes=[('Audio', '*.wav *.flac *.mp3 *.aiff *.m4a *.ogg'), ('All', '*.*')],
        )
        if f:
            self.source_path = pathlib.Path(f)
            self.lbl_src.configure(text=self.source_path.name, foreground='#e6e6e6')

    def on_go(self):
        if not self.source_path:
            messagebox.showwarning('TD Radio', 'Pick a source audio file first.'); return
        name = self.name_var.get().strip()
        if not re.match(r'^td-audio\d{3}$', name):
            messagebox.showwarning('TD Radio', 'Name must look like td-audio007.'); return
        if not (self.out_normal_var.get() or self.out_high_var.get()):
            messagebox.showwarning('TD Radio', 'Pick at least one output.'); return

        # disable button during work
        self.btn_go.configure(state='disabled')
        self.log_txt.delete('1.0', 'end')
        self.log(f'→ source: {self.source_path}')

        TRACKS_DIR.mkdir(parents=True, exist_ok=True)
        ok = True
        if self.out_normal_var.get():
            dst = TRACKS_DIR / f'{name}-normal.mp3'
            if not convert_to_normal_mp3(self.source_path, dst, self.log):
                ok = False; self.log(f'! failed normal conversion')
            else:
                self.log(f'  ✓ {dst.name}')
        if ok and self.out_high_var.get():
            dst = TRACKS_DIR / f'{name}-high.wav'
            if not make_high_wav(self.source_path, dst, self.log):
                ok = False; self.log(f'! failed high conversion')
            else:
                self.log(f'  ✓ {dst.name}')
        if ok and self.append_block_var.get():
            srt_file = ''
            if self.srt_var.get() == '1' and not self.srt_samename_var.get():
                srt_file = self.srt_file_var.get().strip()
            block = scaffold_block(
                name=name,
                signal=self.signal_var.get().strip() or 'ID',
                signal_offline=self.signal_off_var.get().strip() or 'ID',
                srt_mode=self.srt_var.get(),
                srt_filename=srt_file,
                has_normal=self.out_normal_var.get(),
                has_high=self.out_high_var.get(),
            )
            append_to_signal(block, self.log)

        self.log('')
        self.log('✓ DONE' if ok else '✗ FAILED')
        self.btn_go.configure(state='normal')
        self._refresh_auto_name()


# ══════════════════════════════════════════════════════════════════════
# MARK TAB
# ══════════════════════════════════════════════════════════════════════
MARK_TYPES = ['signal', 'decoder', 'origin', 'status', 'access', 'vibe', 'print', 'skip']


class MarkTab(ttk.Frame):
    def __init__(self, master):
        super().__init__(master, padding=12)
        self.stem: str | None = None
        self.audio_path: pathlib.Path | None = None
        self.duration_s: float = 0.0
        self.play_start_wall: float | None = None   # perf_counter() when play began
        self.play_offset_at_start: float = 0.0      # audio time when play() was called
        self.paused = True
        self.marks: list[dict] = []
        self._build()
        self._tick()

    # ── UI ──
    def _build(self):
        if not PYGAME_OK:
            ttk.Label(self, text='MARK TAB DISABLED — pygame not installed.\n'
                                 'Run: pip install pygame',
                      foreground='#f44336', justify='center', padding=30).pack(fill='both', expand=True)
            return

        # Track picker row
        row = ttk.Frame(self); row.pack(fill='x')
        ttk.Label(row, text='Track:').pack(side='left')
        self.track_cbo = ttk.Combobox(row, values=list_existing_tracks(), state='readonly', width=18)
        self.track_cbo.pack(side='left', padx=(6, 6))
        ttk.Button(row, text='⟳', width=3, command=self._refresh_tracks).pack(side='left')
        ttk.Button(row, text='Load', command=self.on_load).pack(side='left', padx=(10, 0))

        # Transport
        row = ttk.Frame(self); row.pack(fill='x', pady=(10, 0))
        ttk.Button(row, text='▶ Play / Pause', command=self.on_toggle_play).pack(side='left')
        ttk.Button(row, text='■ Stop',         command=self.on_stop).pack(side='left', padx=(6, 0))
        self.time_lbl = ttk.Label(row, text='--:--.- / --:--.-', font=('Courier New', 11))
        self.time_lbl.pack(side='right')

        # Seek slider
        self.seek_var = tk.DoubleVar(value=0)
        self.seek = ttk.Scale(self, from_=0, to=100, orient='horizontal', variable=self.seek_var,
                              command=self._on_seek_drag)
        self.seek.pack(fill='x', pady=(4, 0))
        self.seek.bind('<ButtonRelease-1>', lambda e: self._on_seek_commit())

        # Active-marker radio + print text
        row = ttk.LabelFrame(self, text='ACTIVE MARKER  (Space = capture now)', padding=6)
        row.pack(fill='x', pady=(12, 0))
        self.active_type = tk.StringVar(value='signal')
        radio_row = ttk.Frame(row); radio_row.pack(fill='x')
        for t in MARK_TYPES:
            ttk.Radiobutton(radio_row, text=t, value=t, variable=self.active_type).pack(side='left', padx=2)
        row2 = ttk.Frame(row); row2.pack(fill='x', pady=(6, 0))
        ttk.Label(row2, text='Value:').pack(side='left')
        self.value_var = tk.StringVar(value='')
        ttk.Entry(row2, textvariable=self.value_var).pack(side='left', fill='x', expand=True, padx=6)
        ttk.Label(row2, text='d=').pack(side='left')
        self.dur_var = tk.StringVar(value='')
        ttk.Entry(row2, textvariable=self.dur_var, width=4).pack(side='left')

        # Captured table
        frame_tbl = ttk.LabelFrame(self, text='CAPTURED', padding=4)
        frame_tbl.pack(fill='both', expand=True, pady=(10, 0))
        cols = ('time', 'type', 'value', 'd')
        self.tv = ttk.Treeview(frame_tbl, columns=cols, show='headings', height=10)
        for c, w in zip(cols, (80, 80, 420, 40)):
            self.tv.heading(c, text=c.upper())
            self.tv.column(c, width=w, anchor='w')
        self.tv.pack(fill='both', expand=True, side='left')
        sb = ttk.Scrollbar(frame_tbl, orient='vertical', command=self.tv.yview)
        sb.pack(side='right', fill='y')
        self.tv.configure(yscrollcommand=sb.set)

        # Action row
        row = ttk.Frame(self); row.pack(fill='x', pady=(8, 0))
        ttk.Button(row, text='Delete selected', command=self.on_delete).pack(side='left')
        ttk.Button(row, text='Clear all',       command=self.on_clear).pack(side='left', padx=(6, 0))
        ttk.Button(row, text='APPEND TO cf-signal.dat', command=self.on_commit).pack(side='right')

        # Log
        self.log_txt = tk.Text(self, height=6, bg='#111', fg='#e6e6e6',
                               insertbackground='#e6e6e6', font=('Courier New', 9))
        self.log_txt.pack(fill='x', pady=(6, 0))

        # Key bindings — capture when focus is anywhere in this tab
        self.bind_all('<space>',       lambda e: self._capture_if_active(e))
        self.bind_all('<BackSpace>',   lambda e: self._capture_backspace(e))

    def log(self, line: str):
        self.log_txt.insert('end', line + '\n')
        self.log_txt.see('end')
        self.log_txt.update_idletasks()

    def _refresh_tracks(self):
        self.track_cbo.configure(values=list_existing_tracks())

    def on_load(self):
        stem = self.track_cbo.get()
        if not stem:
            return
        path = track_file_for_stem(stem)
        if not path:
            messagebox.showwarning('TD Radio', f'No playable file for {stem}'); return
        try:
            pygame.mixer.music.stop()
            pygame.mixer.music.load(str(path))
        except Exception as e:
            messagebox.showerror('TD Radio', f'pygame load failed:\n{e}'); return
        self.stem = stem
        self.audio_path = path
        self.duration_s = self._probe_duration(path)
        self.seek.configure(from_=0, to=max(self.duration_s, 1))
        self.paused = True
        self.play_start_wall = None
        self.play_offset_at_start = 0.0
        self.seek_var.set(0)
        self.marks.clear()
        for iid in self.tv.get_children():
            self.tv.delete(iid)
        self.log(f'loaded {stem} ({path.name}, {self.duration_s:.1f}s)')

    def _probe_duration(self, path: pathlib.Path) -> float:
        """Best-effort duration: try ffprobe, else pygame Sound.get_length (WAV only)."""
        probe = shutil.which('ffprobe')
        if probe:
            try:
                p = subprocess.run(
                    [probe, '-v', 'error', '-show_entries', 'format=duration',
                     '-of', 'default=nw=1:nk=1', str(path)],
                    capture_output=True, text=True, timeout=5,
                    creationflags=(subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0),
                )
                return float((p.stdout or '').strip() or 0)
            except Exception:
                pass
        if path.suffix.lower() == '.wav':
            try:
                return pygame.mixer.Sound(str(path)).get_length()
            except Exception:
                pass
        return 0.0

    # ── transport ──
    def _current_time(self) -> float:
        if self.paused or self.play_start_wall is None:
            return self.play_offset_at_start
        return self.play_offset_at_start + (time.perf_counter() - self.play_start_wall)

    def on_toggle_play(self):
        if not self.audio_path:
            return
        if self.paused:
            offset = self.play_offset_at_start
            try:
                pygame.mixer.music.play(start=offset)
            except Exception:
                # Some codecs reject start-offset via pygame — fall back to restart + skip
                pygame.mixer.music.play()
            self.play_start_wall = time.perf_counter()
            self.paused = False
        else:
            self.play_offset_at_start = self._current_time()
            pygame.mixer.music.pause()
            self.paused = True

    def on_stop(self):
        pygame.mixer.music.stop()
        self.paused = True
        self.play_start_wall = None
        self.play_offset_at_start = 0.0
        self.seek_var.set(0)

    def _on_seek_drag(self, val):
        # User actively dragging — display but don't commit until release
        if self.paused:
            self.play_offset_at_start = float(val)
        self.time_lbl.configure(text=f'{self._fmt(float(val))} / {self._fmt(self.duration_s)}')

    def _on_seek_commit(self):
        target = float(self.seek_var.get())
        self.play_offset_at_start = target
        if not self.paused:
            try:
                pygame.mixer.music.play(start=target)
                self.play_start_wall = time.perf_counter()
            except Exception:
                pass

    def _tick(self):
        # Update seek bar + time label while playing
        if self.audio_path and not self.paused:
            t = self._current_time()
            if self.duration_s > 0 and t >= self.duration_s:
                self.on_stop()
                t = self.duration_s
            self.seek_var.set(t)
            self.time_lbl.configure(text=f'{self._fmt(t)} / {self._fmt(self.duration_s)}')
        self.after(100, self._tick)

    @staticmethod
    def _fmt(t: float) -> str:
        if t < 0 or not t == t:  # NaN check
            return '--:--.-'
        m = int(t // 60); s = t - m * 60
        return f'{m}:{s:04.1f}'

    # ── capture ──
    def _capture_if_active(self, e):
        # Don't hijack space when an Entry has focus
        w = self.focus_get()
        if isinstance(w, (tk.Entry, ttk.Entry)):
            return
        if not self.audio_path:
            return
        self.on_capture()
        return 'break'

    def _capture_backspace(self, e):
        w = self.focus_get()
        if isinstance(w, (tk.Entry, ttk.Entry)):
            return
        # Remove last row
        items = self.tv.get_children()
        if not items:
            return
        self.tv.delete(items[-1])
        if self.marks:
            self.marks.pop()
        return 'break'

    def on_capture(self):
        t = self._current_time()
        type_ = self.active_type.get()
        value = self.value_var.get().strip()
        dur = self.dur_var.get().strip()
        mark = {'time': t, 'type': type_, 'value': value, 'duration': dur or None}
        self.marks.append(mark)
        self.tv.insert('', 'end', values=(self._fmt(t), type_, value, dur))
        self.log(f'+ {type_}@{self._fmt(t)}  {value}')

    def on_delete(self):
        sel = self.tv.selection()
        if not sel:
            return
        for iid in sel:
            idx = self.tv.index(iid)
            if 0 <= idx < len(self.marks):
                self.marks.pop(idx)
            self.tv.delete(iid)

    def on_clear(self):
        self.marks.clear()
        for iid in self.tv.get_children():
            self.tv.delete(iid)

    def on_commit(self):
        if not self.stem:
            messagebox.showwarning('TD Radio', 'Load a track first.'); return
        if not self.marks:
            messagebox.showwarning('TD Radio', 'No marks to append.'); return
        if not messagebox.askyesno('TD Radio', f'Append {len(self.marks)} mark(s) into the '
                                               f'{self.stem} block of cf-signal.dat?\n'
                                               f'Backup will be saved first.'):
            return
        # Sort by time ascending before appending
        sorted_marks = sorted(self.marks, key=lambda m: m['time'])
        if append_marks_to_track_block(self.stem, sorted_marks, self.log):
            self.log('✓ written')
        else:
            self.log('✗ aborted')


# ══════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════
def main():
    root = tk.Tk()
    root.title('TD Radio — Tools')
    root.geometry('820x640')

    # Dark-ish ttk theme
    style = ttk.Style(root)
    try:
        style.theme_use('clam')
    except Exception:
        pass
    root.configure(bg='#0a0a0a')
    style.configure('.', background='#0a0a0a', foreground='#e6e6e6', fieldbackground='#1a1a1a')
    style.configure('TLabelFrame', background='#0a0a0a', foreground='#de7029')
    style.configure('TLabelFrame.Label', background='#0a0a0a', foreground='#de7029')
    style.configure('TNotebook', background='#0a0a0a')
    style.configure('TNotebook.Tab', padding=(16, 6))
    style.configure('Treeview', background='#111', fieldbackground='#111', foreground='#e6e6e6')
    style.configure('Treeview.Heading', background='#1a1a1a', foreground='#de7029')

    nb = ttk.Notebook(root)
    nb.add(UploadTab(nb), text='UPLOAD')
    nb.add(MarkTab(nb),   text='MARK')
    nb.pack(fill='both', expand=True)

    # Footer status
    footer = ttk.Frame(root); footer.pack(fill='x', side='bottom')
    status = f'radio/ = {RADIO_DIR} · ffmpeg: {"ok" if FFMPEG else "MISSING"} · pygame: {"ok" if PYGAME_OK else "MISSING"}'
    ttk.Label(footer, text=status, foreground='#888', padding=4).pack(side='left')

    root.mainloop()


if __name__ == '__main__':
    main()
