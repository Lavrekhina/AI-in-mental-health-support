from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import matplotlib as mpl


@dataclass(frozen=True)
class EmotionTheme:
    """
    Emotion-aware theme choices (calm base, distress highlight).

    These defaults are intentionally soft/low-saturation to avoid sensationalizing distress.
    """

    background: str = "#fbfcfd"
    text: str = "#1f2937"

    calm_primary: str = "#2b6cb0"  # muted blue
    calm_secondary: str = "#68d391"  # soft green
    neutral: str = "#94a3b8"  # slate

    distress: str = "#b91c1c"  # red (used sparingly)
    distress_soft: str = "#fecaca"  # pale red

    grid: str = "#e5e7eb"

    emotion_palette: Tuple[str, ...] = (
        "#3b82f6",  # anxious (blue)
        "#60a5fa",  # overwhelmed
        "#93c5fd",  # stressed
        "#475569",  # sad — darker slate so label/bar stays readable on light background
        "#a78bfa",  # lonely (lavender)
        "#34d399",  # calm (green)
        "#fbbf24",  # hopeful (warm amber)
    )

    response_palette: Dict[str, str] = None  # set in __post_init__

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "response_palette",
            {
                "empathetic": "#34d399",  # green
                "supportive": "#60a5fa",  # blue
                "neutral": "#94a3b8",  # slate
                "corrective": "#fbbf24",  # amber
            },
        )


def apply_mpl_theme(theme: EmotionTheme) -> None:
    mpl.rcParams.update(
        {
            "figure.facecolor": theme.background,
            "axes.facecolor": theme.background,
            "axes.edgecolor": theme.grid,
            "axes.labelcolor": theme.text,
            "xtick.color": theme.text,
            "ytick.color": theme.text,
            "text.color": theme.text,
            "grid.color": theme.grid,
            "axes.grid": True,
            "grid.linestyle": "-",
            "grid.linewidth": 0.6,
            "axes.titleweight": "semibold",
            "axes.titlesize": 14,
            "axes.labelsize": 11,
            "font.size": 11,
        }
    )

