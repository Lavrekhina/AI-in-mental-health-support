from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import seaborn as sns
from wordcloud import WordCloud

from analysis._bootstrap import ensure_src_on_path

ensure_src_on_path()

from src.aihms.data import (
    EMOTION_CANONICAL_ORDER,
    FOLLOWUP_CANONICAL_ORDER,
    RESPONSE_CANONICAL_ORDER,
    load_interactions,
)
from src.aihms.viz import EmotionTheme, apply_mpl_theme


def _ensure_dirs(repo_root: Path) -> tuple[Path, Path]:
    outputs = repo_root / "outputs"
    fig_dir = outputs / "figures"
    outputs.mkdir(exist_ok=True)
    fig_dir.mkdir(exist_ok=True)
    return outputs, fig_dir


def _write_metrics(repo_root: Path, df: pd.DataFrame) -> Path:
    """
    Writes a small table of key metrics that directly answers early questions:
    - most common emotions
    - does empathetic correlate with improved follow-up (returned vs not returned)
    """
    out_path = repo_root / "outputs" / "metrics_step02.csv"

    tmp = df.copy()
    tmp["followed_up"] = tmp["User_follow_up_behaviour"].eq("returned")

    emotions = (
        tmp["Reported_emotion"]
        .value_counts()
        .rename_axis("Reported_emotion")
        .reset_index(name="count")
    )
    emotions["share"] = emotions["count"] / emotions["count"].sum()

    followup_by_response = (
        tmp.groupby("AI_response_classification", dropna=False)["followed_up"]
        .agg(["mean", "count"])
        .reset_index()
        .rename(columns={"mean": "followup_rate_returned"})
    )

    # Simple "uplift" vs neutral, for interpretability.
    neutral_rate = followup_by_response.loc[
        followup_by_response["AI_response_classification"].eq("neutral"),
        "followup_rate_returned",
    ]
    neutral_rate = float(neutral_rate.iloc[0]) if len(neutral_rate) else float("nan")
    followup_by_response["uplift_vs_neutral"] = followup_by_response["followup_rate_returned"] - neutral_rate

    # Store as a multi-section CSV (wide-ish but easy to open in Excel).
    with out_path.open("w", encoding="utf-8") as f:
        f.write("# Emotion prevalence\n")
        emotions.to_csv(f, index=False)
        f.write("\n# Follow-up rate by AI response classification (returned)\n")
        followup_by_response.to_csv(f, index=False)

    return out_path


def plot_emotion_prevalence(fig_dir: Path, df: pd.DataFrame, theme: EmotionTheme) -> Path:
    apply_mpl_theme(theme)

    order = [e for e in EMOTION_CANONICAL_ORDER if e in set(df["Reported_emotion"])]
    counts = df["Reported_emotion"].value_counts().reindex(order)

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(
        x=counts.index,
        y=counts.values,
        hue=counts.index,
        palette=list(theme.emotion_palette)[: len(counts)],
        dodge=False,
        ax=ax,
    )
    leg = ax.get_legend()
    if leg is not None:
        leg.remove()
    ax.set_title("What emotions are most common among users seeking help?")
    ax.set_xlabel("Reported emotion (normalized)")
    ax.set_ylabel("Number of interactions")
    for patch in ax.patches:
        patch.set_edgecolor("#334155")
        patch.set_linewidth(0.85)

    # Gentle annotation — top emotion only, avoids clutter.
    top_emotion = counts.idxmax()
    top_count = int(counts.max())
    ax.annotate(
        f"Most common: {top_emotion} ({top_count})",
        xy=(0.98, 0.95),
        xycoords="axes fraction",
        ha="right",
        va="top",
        fontsize=11,
        bbox=dict(boxstyle="round,pad=0.35", fc="white", ec=theme.grid),
    )

    out = fig_dir / "emotion_prevalence_bar.png"
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


def plot_emotion_wordcloud(fig_dir: Path, df: pd.DataFrame, theme: EmotionTheme) -> Path:
    freqs = df["Reported_emotion"].value_counts().to_dict()

    emotion_colors = {
        e: theme.emotion_palette[i]
        for i, e in enumerate(EMOTION_CANONICAL_ORDER)
        if i < len(theme.emotion_palette)
    }

    def _wc_color(word: str, font_size: float, position, orientation, random_state=None, **kwargs: object) -> str:
        return emotion_colors.get(str(word).lower(), theme.text)

    wc = WordCloud(
        width=1200,
        height=600,
        background_color=theme.background,
        prefer_horizontal=0.85,
        normalize_plurals=False,
        random_state=13,
        color_func=_wc_color,
        min_font_size=14,
    ).generate_from_frequencies(freqs)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    ax.set_title("Emotion cloud (anonymized, label-only)")

    out = fig_dir / "emotion_wordcloud.png"
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


def plot_followup_by_response(fig_dir: Path, df: pd.DataFrame, theme: EmotionTheme) -> tuple[Path, Path]:
    """
    Visualization (static + interactive) of interaction between AI response type and follow-up behaviour.
    """
    apply_mpl_theme(theme)

    order_r = [r for r in RESPONSE_CANONICAL_ORDER if r in set(df["AI_response_classification"])]
    order_f = [f for f in FOLLOWUP_CANONICAL_ORDER if f in set(df["User_follow_up_behaviour"])]

    ctab = pd.crosstab(
        df["AI_response_classification"],
        df["User_follow_up_behaviour"],
        normalize="index",
    ).reindex(index=order_r, columns=order_f)

    fig, ax = plt.subplots(figsize=(10, 5))
    ctab.plot(kind="bar", stacked=True, ax=ax, color=[theme.calm_primary, theme.neutral, theme.distress_soft])
    ax.set_title("Do AI response styles relate to whether users follow up?")
    ax.set_xlabel("AI response classification")
    ax.set_ylabel("Share within response type")
    ax.legend(title="User follow-up behaviour", bbox_to_anchor=(1.02, 1), loc="upper left")
    ax.set_ylim(0, 1)

    out_static = fig_dir / "followup_by_response_stacked.png"
    fig.tight_layout()
    fig.savefig(out_static, dpi=200)
    plt.close(fig)

    # Interactive version (Plotly)
    ctab_long = (
        ctab.reset_index()
        .melt(id_vars="AI_response_classification", var_name="User_follow_up_behaviour", value_name="share")
        .dropna()
    )
    fig_i = px.bar(
        ctab_long,
        x="AI_response_classification",
        y="share",
        color="User_follow_up_behaviour",
        barmode="stack",
        category_orders={
            "AI_response_classification": order_r,
            "User_follow_up_behaviour": order_f,
        },
        title="Follow-up behaviour by AI response type (interactive)",
        labels={"share": "Share within response type"},
        color_discrete_sequence=[theme.calm_primary, theme.neutral, theme.distress_soft],
    )
    fig_i.update_layout(template="simple_white", yaxis_tickformat=".0%")

    out_html = fig_dir / "followup_by_response_interactive.html"
    fig_i.write_html(out_html, include_plotlyjs="cdn")

    return out_static, out_html


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    outputs_dir, fig_dir = _ensure_dirs(repo_root)

    theme = EmotionTheme()

    result = load_interactions(repo_root / "AI_mental_health_interactions.csv")
    df = result.df

    # Metrics + early questions (textual outputs)
    metrics_path = _write_metrics(repo_root, df)

    # Visualizations required early in the storyline
    p1 = plot_emotion_prevalence(fig_dir, df, theme)
    p2 = plot_emotion_wordcloud(fig_dir, df, theme)
    p3_static, p3_html = plot_followup_by_response(fig_dir, df, theme)

    # Before/after sentiment plots will be implemented in a later step if columns exist.
    # (We keep the check here to make the pipeline self-explanatory.)
    if "Sentiment_score_after" not in df.columns:
        note = outputs_dir / "step02_notes.txt"
        note.write_text(
            "This dataset version has no Sentiment_score_after; before/after sentiment analyses are skipped.\n"
            "If your Moodle CSV includes Sentiment_score_after (and optionally Reported_emotion_after),\n"
            "later steps will automatically use them.\n",
            encoding="utf-8",
        )

    print(f"Wrote {metrics_path.relative_to(repo_root)}")
    print(f"Wrote {p1.relative_to(repo_root)}")
    print(f"Wrote {p2.relative_to(repo_root)}")
    print(f"Wrote {p3_static.relative_to(repo_root)}")
    print(f"Wrote {p3_html.relative_to(repo_root)}")


if __name__ == "__main__":
    main()

