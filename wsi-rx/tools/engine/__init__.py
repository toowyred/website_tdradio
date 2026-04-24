"""WSI RX offline Auto-DJ engine.

Phase 2 of the v2 rebuild. Each module in this package is one stage of the
offline pipeline that authors `r/<trk-xxxxxxx>/dj.dat` mix blueprints the
browser follows blindly at runtime:

    probe.py        env validator + per-stage benchmark (Phase 2a)
    stems.py        UVR5 / wsi-build0406 stem separation wrapper (2b)
    structure.py    beats + downbeats + sections + key + drops (2c, 2d)
    embeddings.py   CLAP / MERT / Essentia-TF semantic tagging (2e, 2f)
    judge.py        Gemma 4:26b mix-blueprint author (2g)
    audition.py     Tkinter feedback loop (2h)

Nothing in this package touches webiste_TD/radio/ — v1.5 keeps its own
maintenance stream uninterrupted.
"""
