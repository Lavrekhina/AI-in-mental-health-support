from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Union

import pandas as pd


REQUIRED_COLUMNS: tuple[str, ...] = (
    "Date_of_interaction",
    "User_age_group",
    "Reported_emotion",
    "Sentiment_score",
    "AI_response_classification",
    "User_follow_up_behaviour",
    "Conversation_length_tokens",
)


EMOTION_CANONICAL_ORDER: tuple[str, ...] = (
    "anxious",
    "overwhelmed",
    "stressed",
    "sad",
    "lonely",
    "calm",
    "hopeful",
)


RESPONSE_CANONICAL_ORDER: tuple[str, ...] = (
    "empathetic",
    "supportive",
    "neutral",
    "corrective",
)


FOLLOWUP_CANONICAL_ORDER: tuple[str, ...] = (
    "returned",
    "dropped",
    "escalated_to_human",
)


@dataclass(frozen=True)
class LoadResult:
    df: pd.DataFrame
    warnings: tuple[str, ...]


def _norm_str(x: object) -> Optional[str]:
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return None
    s = str(x).strip()
    if not s:
        return None
    return " ".join(s.split()).lower()


def normalize_emotion_label(x: object) -> Optional[str]:
    """
    Normalize emotion labels to a small canonical vocabulary.

    The provided dataset already uses a small set, but this function is intentionally
    tolerant to common variants (case, whitespace, mild synonyms).
    """
    s = _norm_str(x)
    if s is None:
        return None

    mapping = {
        "anxiety": "anxious",
        "anxious": "anxious",
        "over whelmed": "overwhelmed",
        "overwhelmed": "overwhelmed",
        "stress": "stressed",
        "stressed": "stressed",
        "sadness": "sad",
        "sad": "sad",
        "alone": "lonely",
        "lonely": "lonely",
        "calm": "calm",
        "peaceful": "calm",
        "hope": "hopeful",
        "hopeful": "hopeful",
    }
    return mapping.get(s, s)


def normalize_category(x: object) -> Optional[str]:
    return _norm_str(x)


def validate_schema(df: pd.DataFrame, required_columns: Iterable[str] = REQUIRED_COLUMNS) -> list[str]:
    missing = [c for c in required_columns if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}. Found: {list(df.columns)}")
    return []


def load_interactions(csv_path: Union[str, Path]) -> LoadResult:
    """
    Load the interactions CSV and apply light, reproducible cleaning.

    - Parses dates
    - Normalizes categorical labels
    - Casts numeric columns
    - Adds a few derived convenience columns used by later steps
    """
    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path)
    validate_schema(df)

    warnings: list[str] = []

    # Parse + normalize columns
    df["Date_of_interaction"] = pd.to_datetime(df["Date_of_interaction"], errors="coerce")
    if df["Date_of_interaction"].isna().any():
        warnings.append("Some rows have invalid Date_of_interaction and were set to NaT.")

    df["User_age_group"] = df["User_age_group"].map(normalize_category)
    df["AI_response_classification"] = df["AI_response_classification"].map(normalize_category)
    df["User_follow_up_behaviour"] = df["User_follow_up_behaviour"].map(normalize_category)

    df["Reported_emotion_raw"] = df["Reported_emotion"]
    df["Reported_emotion"] = df["Reported_emotion"].map(normalize_emotion_label)

    df["Sentiment_score"] = pd.to_numeric(df["Sentiment_score"], errors="coerce")
    if df["Sentiment_score"].isna().any():
        warnings.append("Some rows have invalid Sentiment_score and were set to NaN.")

    df["Conversation_length_tokens"] = pd.to_numeric(df["Conversation_length_tokens"], errors="coerce")
    if df["Conversation_length_tokens"].isna().any():
        warnings.append("Some rows have invalid Conversation_length_tokens and were set to NaN.")

    # Optional columns for alternate dataset versions (kept if present).
    optional_after_cols = ["Sentiment_score_after", "Reported_emotion_after"]
    for c in optional_after_cols:
        if c in df.columns and c == "Sentiment_score_after":
            df[c] = pd.to_numeric(df[c], errors="coerce")
        if c in df.columns and c == "Reported_emotion_after":
            df[c] = df[c].map(normalize_emotion_label)

    # Derived: rough risk tier based on sentiment (purely analytical; avoid individual labeling in narrative).
    df["risk_tier"] = pd.cut(
        df["Sentiment_score"],
        bins=[-float("inf"), -0.6, -0.3, 0.3, float("inf")],
        labels=["high_distress", "moderate_distress", "mixed", "positive"],
    )

    return LoadResult(df=df, warnings=tuple(warnings))

