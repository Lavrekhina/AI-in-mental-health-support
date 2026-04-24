"""AI in mental health support (mini-project) utilities.

This package uses a small `src/` layout. For convenience in notebooks and scripts,
key functions are re-exported at the package level.
"""

from src.aihms.data import (  # noqa: F401
    EMOTION_CANONICAL_ORDER,
    FOLLOWUP_CANONICAL_ORDER,
    RESPONSE_CANONICAL_ORDER,
    LoadResult,
    load_interactions,
)

__all__ = [
    "load_interactions",
    "LoadResult",
    "EMOTION_CANONICAL_ORDER",
    "RESPONSE_CANONICAL_ORDER",
    "FOLLOWUP_CANONICAL_ORDER",
]

