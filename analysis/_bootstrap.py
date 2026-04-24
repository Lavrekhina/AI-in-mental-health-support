from __future__ import annotations

import sys
from pathlib import Path


def ensure_src_on_path() -> None:
    """
    Make `src/` importable without requiring `pip install -e .`.

    This keeps the project runnable in teaching environments where editable installs
    may not be used.
    """
    repo_root = Path(__file__).resolve().parents[1]
    src = repo_root / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))

