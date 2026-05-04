from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import seaborn as sns

from analysis._bootstrap import ensure_src_on_path

ensure_src_on_path()

from src.aihms.data import EMOTION_CANONICAL_ORDER, RESPONSE_CANONICAL_ORDER, load_interactions
from src.aihms.viz import EmotionTheme, apply_mpl_theme


def _ensure_dirs(repo_root: Path) -> tuple[Path, Path]:
    outputs = repo_root / "outputs"
    fig_dir = outputs / "figures"
    outputs.mkdir(exist_ok=True)
    fig_dir.mkdir(exist_ok=True)
    return outputs, fig_dir


def _get_after_columns(df: pd.DataFrame) -> Tuple[Optional[str], Optional[str], str]:
    """
    Returns:
      - emotion_after_col (or None)
      - sentiment_after_col (or None)
      - mode: "observed" if after-columns exist, else "proxy"
    """
    emotion_after = "Reported_emotion_after" if "Reported_emotion_after" in df.columns else None
    sentiment_after = "Sentiment_score_after" if "Sentiment_score_after" in df.columns else None
    if emotion_after is not None and sentiment_after is not None:
        return emotion_after, sentiment_after, "observed"
    return emotion_after, sentiment_after, "proxy"


def add_proxy_after(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create *proxy* post-response sentiment/emotion when the dataset lacks "after" columns.

    Rationale: we need a consistent way to demonstrate the pipeline end-to-end while
    clearly labeling results as proxy estimates (not ground-truth).
    """
    out = df.copy()

    # Proxy delta: based on follow-up behaviour + response style.
    # Returned tends to indicate engagement; escalation indicates high risk; dropped may indicate disengagement.
    followup_delta = {
        "returned": 0.08,
        "dropped": -0.04,
        "escalated_to_human": -0.10,
    }
    response_delta = {
        "empathetic": 0.05,
        "supportive": 0.03,
        "neutral": 0.00,
        "corrective": -0.02,
    }

    delta = out["User_follow_up_behaviour"].map(followup_delta).fillna(0.0) + out[
        "AI_response_classification"
    ].map(response_delta).fillna(0.0)

    out["Sentiment_score_after_proxy"] = (out["Sentiment_score"] + delta).clip(-1.0, 1.0)

    # Proxy emotion shift: bucketed on sentiment movement only (keeps it simple and transparent).
    bins = [-np.inf, -0.3, 0.3, np.inf]
    labels = ["distressed", "mixed", "positive"]
    before_bucket = pd.cut(out["Sentiment_score"], bins=bins, labels=labels)
    after_bucket = pd.cut(out["Sentiment_score_after_proxy"], bins=bins, labels=labels)
    out["Emotion_bucket_before_proxy"] = before_bucket.astype(str)
    out["Emotion_bucket_after_proxy"] = after_bucket.astype(str)

    return out


def plot_before_after_sentiment_by_response(
    fig_dir: Path, df: pd.DataFrame, theme: EmotionTheme, sentiment_after_col: str, mode: str
) -> Path:
    apply_mpl_theme(theme)

    order_r = [r for r in RESPONSE_CANONICAL_ORDER if r in set(df["AI_response_classification"])]

    plot_df = df[["AI_response_classification", "Sentiment_score", sentiment_after_col]].dropna().copy()
    plot_df = plot_df.rename(
        columns={"Sentiment_score": "before", sentiment_after_col: "after", "AI_response_classification": "response"}
    )
    plot_df = plot_df.melt(id_vars=["response"], value_vars=["before", "after"], var_name="time", value_name="sentiment")

    fig, ax = plt.subplots(figsize=(11, 5.5))
    sns.violinplot(
        data=plot_df,
        x="response",
        y="sentiment",
        hue="time",
        order=order_r,
        split=True,
        inner="quartile",
        linewidth=0.8,
        palette=[theme.neutral, theme.calm_secondary],
        ax=ax,
    )
    # Use a figure-level title + a smaller subtitle to avoid overlaps.
    fig.suptitle("Before vs after sentiment by AI response type", y=0.98, fontsize=14, fontweight="semibold")
    ax.set_xlabel("AI response classification")
    ax.set_ylabel("Sentiment score (higher = more positive)")
    # Put legend outside the plotting area to avoid overlapping annotations.
    ax.legend(title="", loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0)

    subtitle = "Observed after-sentiment from dataset" if mode == "observed" else "Proxy after-sentiment (no after-columns in CSV)"
    ax.set_title(subtitle, loc="left", fontsize=10, pad=10)

    out = fig_dir / ("sentiment_before_after_by_response.png" if mode == "observed" else "sentiment_before_after_by_response_PROXY.png")
    # Leave room on the right for the legend and on top for the suptitle.
    fig.tight_layout(rect=(0, 0, 0.82, 0.92))
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


def plot_agegroup_shift(fig_dir: Path, df: pd.DataFrame, theme: EmotionTheme, sentiment_after_col: str, mode: str) -> Path:
    apply_mpl_theme(theme)

    tmp = df[["User_age_group", "Sentiment_score", sentiment_after_col]].dropna().copy()
    tmp["delta"] = tmp[sentiment_after_col] - tmp["Sentiment_score"]

    shift = tmp.groupby("User_age_group")["delta"].agg(["mean", "median", "count"]).reset_index()
    shift = shift.sort_values("mean")

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=shift, x="User_age_group", y="mean", color=theme.calm_primary, ax=ax)
    ax.axhline(0, color=theme.grid, linewidth=1)
    ax.set_title("Which age groups show the biggest emotional shifts?")
    ax.set_xlabel("Age group")
    ax.set_ylabel("Mean sentiment change (after − before)")

    subtitle = "Observed (dataset after-columns)" if mode == "observed" else "Proxy (estimated after-sentiment)"
    ax.annotate(
        subtitle,
        xy=(0.01, 0.98),
        xycoords="axes fraction",
        ha="left",
        va="top",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=theme.grid),
    )

    out = fig_dir / ("agegroup_sentiment_shift.png" if mode == "observed" else "agegroup_sentiment_shift_PROXY.png")
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


def sankey_emotion_transition(fig_dir: Path, df: pd.DataFrame, mode: str) -> Path:
    """
    Emotion transition diagram.

    - If Reported_emotion_after exists: uses (Reported_emotion -> Reported_emotion_after)
    - Else: uses proxy buckets (Emotion_bucket_before_proxy -> Emotion_bucket_after_proxy)
    """
    if mode == "observed" and "Reported_emotion_after" in df.columns:
        source_col = "Reported_emotion"
        target_col = "Reported_emotion_after"
        title = "Emotion transitions after AI reply (observed labels)"
    else:
        source_col = "Emotion_bucket_before_proxy"
        target_col = "Emotion_bucket_after_proxy"
        title = "Emotion transitions after AI reply (proxy buckets)"

    tmp = df[[source_col, target_col]].dropna().copy()
    tmp[source_col] = tmp[source_col].astype(str)
    tmp[target_col] = tmp[target_col].astype(str)

    counts = tmp.groupby([source_col, target_col]).size().reset_index(name="count")

    sources = counts[source_col].unique().tolist()
    targets = counts[target_col].unique().tolist()
    labels = sources + [t for t in targets if t not in sources]

    label_to_idx = {lab: i for i, lab in enumerate(labels)}
    src_idx = counts[source_col].map(label_to_idx).tolist()
    tgt_idx = counts[target_col].map(label_to_idx).tolist()
    values = counts["count"].tolist()

    fig = go.Figure(
        data=[
            go.Sankey(
                node=dict(
                    pad=14,
                    thickness=16,
                    line=dict(color="rgba(0,0,0,0.15)", width=0.5),
                    label=labels,
                    color=["rgba(59,130,246,0.25)"] * len(labels),
                ),
                link=dict(
                    source=src_idx,
                    target=tgt_idx,
                    value=values,
                    color=["rgba(59,130,246,0.18)"] * len(values),
                ),
            )
        ]
    )
    fig.update_layout(title=title, font=dict(size=12), template="simple_white")

    out = fig_dir / ("emotion_transition_sankey.html" if mode == "observed" else "emotion_transition_sankey_PROXY.html")
    fig.write_html(out, include_plotlyjs="cdn")
    png_out = out.with_suffix(".png")
    try:
        fig.write_image(str(png_out), width=1280, height=720, scale=2)
    except Exception:
        # Optional: requires `kaleido` (see requirements.txt). HTML export always works.
        pass
    return out


def plot_extreme_negative(fig_dir: Path, df: pd.DataFrame, theme: EmotionTheme) -> Path:
    """
    Highlight extreme negative sentiment with color intensity and soft-edged markers.
    Uses interaction-level points (no IDs) and avoids singling out individuals in text.
    """
    apply_mpl_theme(theme)

    tmp = df[["Sentiment_score", "Conversation_length_tokens", "AI_response_classification"]].dropna().copy()

    # Use a quantile-based definition to avoid hard thresholds.
    q = float(tmp["Sentiment_score"].quantile(0.05))
    tmp["is_extreme_negative"] = tmp["Sentiment_score"] <= q

    fig, ax = plt.subplots(figsize=(10, 5.5))

    # Base cloud (soft, calm)
    ax.scatter(
        tmp["Conversation_length_tokens"],
        tmp["Sentiment_score"],
        s=18,
        alpha=0.18,
        c=theme.calm_primary,
        edgecolors="none",
    )

    # Highlight tail (soft-edge glow effect: big pale layer + smaller intense layer)
    tail = tmp[tmp["is_extreme_negative"]]
    ax.scatter(
        tail["Conversation_length_tokens"],
        tail["Sentiment_score"],
        s=160,
        alpha=0.20,
        c=theme.distress_soft,
        edgecolors="none",
    )
    ax.scatter(
        tail["Conversation_length_tokens"],
        tail["Sentiment_score"],
        s=40,
        alpha=0.50,
        c=theme.distress,
        edgecolors="none",
    )

    ax.set_title("Extreme negative sentiment interactions (tail highlight)")
    ax.set_xlabel("Conversation length (tokens)")
    ax.set_ylabel("Sentiment score")
    ax.annotate(
        "Tail defined as lowest 5% sentiment\n(visual emphasis, no individual labels)",
        xy=(0.01, 0.98),
        xycoords="axes fraction",
        ha="left",
        va="top",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=theme.grid),
    )

    out = fig_dir / "extreme_negative_sentiment_scatter.png"
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


def write_risk_tables(outputs_dir: Path, df: pd.DataFrame, sentiment_after_col: Optional[str], mode: str) -> Path:
    """
    Identify at-risk groups based on sentiment trends (aggregated; no individual labeling).
    """
    tmp = df.copy()

    # Group-level risk indicators.
    group_cols = ["User_age_group", "AI_response_classification", "Reported_emotion"]
    base = (
        tmp.groupby(group_cols, dropna=False)
        .agg(
            n=("Sentiment_score", "size"),
            mean_sentiment=("Sentiment_score", "mean"),
            share_high_distress=("risk_tier", lambda s: (s == "high_distress").mean()),
        )
        .reset_index()
    )

    if sentiment_after_col is not None:
        tmp2 = tmp.dropna(subset=["Sentiment_score", sentiment_after_col]).copy()
        tmp2["delta"] = tmp2[sentiment_after_col] - tmp2["Sentiment_score"]
        delta = tmp2.groupby(group_cols, dropna=False)["delta"].mean().reset_index(name="mean_delta")
        base = base.merge(delta, on=group_cols, how="left")
    else:
        base["mean_delta"] = np.nan

    base["delta_mode"] = mode

    out = outputs_dir / ("risk_groups_step03.csv" if mode == "observed" else "risk_groups_step03_PROXY.csv")
    base.sort_values(["share_high_distress", "mean_sentiment"], ascending=[False, True]).to_csv(out, index=False)
    return out


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    outputs_dir, fig_dir = _ensure_dirs(repo_root)

    theme = EmotionTheme()
    result = load_interactions(repo_root / "AI_mental_health_interactions.csv")
    df = result.df

    emotion_after_col, sentiment_after_col, mode = _get_after_columns(df)

    if mode == "proxy":
        df = add_proxy_after(df)
        sentiment_after_col = "Sentiment_score_after_proxy"

    p1 = plot_before_after_sentiment_by_response(fig_dir, df, theme, sentiment_after_col, mode)
    p2 = plot_agegroup_shift(fig_dir, df, theme, sentiment_after_col, mode)
    p3 = sankey_emotion_transition(fig_dir, df, mode)
    p4 = plot_extreme_negative(fig_dir, df, theme)
    p5 = write_risk_tables(outputs_dir, df, sentiment_after_col, mode)

    note = outputs_dir / "step03_notes.txt"
    if mode == "observed":
        note.write_text(
            "Step 03 ran in OBSERVED mode using Sentiment_score_after / Reported_emotion_after from the dataset.\n",
            encoding="utf-8",
        )
    else:
        note.write_text(
            "Step 03 ran in PROXY mode because the dataset has no Sentiment_score_after / Reported_emotion_after.\n"
            "Proxy after-sentiment uses a transparent, small delta based on follow-up behaviour and response style.\n"
            "Use OBSERVED mode for the final report if your Moodle CSV includes after-columns.\n",
            encoding="utf-8",
        )

    for p in [p1, p2, p3, p4, p5, note]:
        print(f"Wrote {Path(p).relative_to(repo_root)}")


if __name__ == "__main__":
    main()

