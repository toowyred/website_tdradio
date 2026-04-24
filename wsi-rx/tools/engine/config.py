"""Paths + constants shared across engine stages."""
from pathlib import Path

# --- Repo layout ---------------------------------------------------------
# This file lives at  webiste_TDradio/wsi-rx/tools/engine/config.py
# Repo root:          webiste_TDradio/
REPO_ROOT  = Path(__file__).resolve().parents[3]
WSI_RX_DIR = REPO_ROOT / 'wsi-rx'
TOOLS_DIR  = WSI_RX_DIR / 'tools'
TRACKS_DIR = WSI_RX_DIR / 'tracks'

# --- Source tracks --------------------------------------------------------
# v2 tracks will live in WSI_RX_DIR/tracks once the engine has something to
# emit. Until then the source-of-truth audio lives in the v1.5 repo next
# door (NOT pushed to either git — R2-only + local-only).
V1_TRACKS_DIR = REPO_ROOT.parent / 'webiste_TD' / 'radio' / 'tracks'

# Default test track for probe.py + one-off benchmarks. Pick the highest-
# fidelity form available (wav > normal.mp3 > high.mp3).
def default_track() -> Path:
    for candidate in (
        V1_TRACKS_DIR / 'td-audio001-high.wav',
        V1_TRACKS_DIR / 'td-audio001-normal.mp3',
    ):
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f'No test track found. Looked in:\n'
        f'  {V1_TRACKS_DIR / "td-audio001-high.wav"}\n'
        f'  {V1_TRACKS_DIR / "td-audio001-normal.mp3"}\n'
        f'Pass --track /path/to/audio.wav to probe.py to override.'
    )

# --- External tool locations ---------------------------------------------
# wsi-build0406.py lives in the LYOKO project; the engine shells out to it
# for stem separation (per the Phase 2b plan — reuse, don't reinvent).
WSI_BUILD_0406 = Path(
    r'W:\Studio\1_Coding\1_app_LYOKO\Core\WSI\wsi-build0406.py'
)

# Ollama HTTP endpoint. Installed + running at 0.0.0.0:11434 per user memory.
OLLAMA_HOST  = 'http://localhost:11434'
OLLAMA_MODEL = 'gemma4:e4b'  # 9.6 GB always-hot; heavier reasoning uses gemma4:26b
