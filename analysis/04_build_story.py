from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

from aihms.data import load_interactions


def _read_sectioned_csv(path: Path) -> dict[str, pd.DataFrame]:
    """
    Reads the custom multi-section CSV we write in step02 (sections start with '# ').
    Returns {section_title: dataframe}.
    """
    text = path.read_text(encoding="utf-8")
    parts: list[tuple[str, str]] = []
    current_title: Optional[str] = None
    current_buf: list[str] = []

    for line in text.splitlines():
        if line.startswith("# "):
            if current_title is not None and current_buf:
                parts.append((current_title, "\n".join(current_buf).strip() + "\n"))
            current_title = line[2:].strip()
            current_buf = []
        else:
            current_buf.append(line)
    if current_title is not None and current_buf:
        parts.append((current_title, "\n".join(current_buf).strip() + "\n"))

    out: dict[str, pd.DataFrame] = {}
    for title, csv_body in parts:
        csv_body = csv_body.strip()
        if not csv_body:
            continue
        from io import StringIO

        out[title] = pd.read_csv(StringIO(csv_body))
    return out


def _fmt_pct(x: float) -> str:
    if pd.isna(x):
        return "n/a"
    return f"{x*100:.1f}%"


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    outputs = repo_root / "outputs"
    fig = outputs / "figures"
    outputs.mkdir(exist_ok=True)
    fig.mkdir(exist_ok=True)

    # Load cleaned data (if available), else load raw and clean in-memory.
    clean_path = outputs / "interactions_clean.csv"
    if clean_path.exists():
        df = pd.read_csv(clean_path)
    else:
        df = load_interactions(repo_root / "AI_mental_health_interactions.csv").df

    # Inputs from previous steps (we keep this resilient: story can be built even if some files missing).
    metrics_path = outputs / "metrics_step02.csv"
    risk_proxy_path = outputs / "risk_groups_step03_PROXY.csv"
    risk_obs_path = outputs / "risk_groups_step03.csv"

    metrics = _read_sectioned_csv(metrics_path) if metrics_path.exists() else {}
    emotion_tbl = metrics.get("Emotion prevalence")
    followup_tbl = metrics.get("Follow-up rate by AI response classification (returned)")

    risk_path = risk_obs_path if risk_obs_path.exists() else risk_proxy_path if risk_proxy_path.exists() else None
    risk_tbl = pd.read_csv(risk_path) if risk_path is not None else None

    # Compute headline answers (from data we have).
    most_common_emotion = None
    if emotion_tbl is not None and len(emotion_tbl):
        top = emotion_tbl.sort_values("count", ascending=False).iloc[0]
        most_common_emotion = (str(top["Reported_emotion"]), int(top["count"]), float(top.get("share", float("nan"))))

    empathetic_followup = None
    if followup_tbl is not None and len(followup_tbl):
        row = followup_tbl.loc[followup_tbl["AI_response_classification"].astype(str).eq("empathetic")]
        if len(row):
            r = row.iloc[0]
            empathetic_followup = (
                float(r["followup_rate_returned"]),
                float(r.get("uplift_vs_neutral", float("nan"))),
                int(r["count"]),
            )

    # Story markdown
    story_path = outputs / "story.md"
    lines: list[str] = []
    lines.append("## AI in mental health support — an emotionally engaging data story\n")
    lines.append(
        "This story explores anonymized interactions between users seeking mental-health support and an AI assistant. "
        "The goal is to understand *what users feel*, *how the AI responds*, and *what happens next*—while keeping the tone "
        "calm, ethical, and non-sensational.\n"
    )

    lines.append("### 1) Which emotions are most common among users seeking help?\n")
    if most_common_emotion is not None:
        emo, cnt, share = most_common_emotion
        lines.append(f"**Most common emotion:** `{emo}` ({cnt} interactions, {_fmt_pct(share)} of the dataset).\n")
    else:
        lines.append("Emotion prevalence metrics were not found (run step 2).\n")

    lines.append("Emotion prevalence (bar):\n")
    lines.append("- `outputs/figures/emotion_prevalence_bar.png`\n")
    lines.append("Emotion cloud (wordcloud):\n")
    lines.append("- `outputs/figures/emotion_wordcloud.png`\n")

    lines.append("### 2) Does AI empathetic response correlate with improved follow-up?\n")
    lines.append(
        "We treat **follow-up** as whether the user **returned** (vs dropped or escalated). "
        "This is a behavioral proxy—useful, but not a clinical outcome.\n"
    )
    if empathetic_followup is not None:
        rate, uplift, n = empathetic_followup
        lines.append(f"**Empathetic follow-up rate (returned):** {_fmt_pct(rate)} (n={n}).\n")
        lines.append(f"**Uplift vs neutral:** {_fmt_pct(uplift)}.\n")
    else:
        lines.append("Follow-up metrics by response type were not found (run step 2).\n")

    lines.append("Response type × follow-up behaviour:\n")
    lines.append("- `outputs/figures/followup_by_response_stacked.png`\n")
    lines.append("- `outputs/figures/followup_by_response_interactive.html`\n")

    lines.append("### 3) Are certain AI response styles linked to negative emotional reinforcement?\n")
    lines.append(
        "We examine **before vs after sentiment** by AI response type. "
        "If your dataset includes `Sentiment_score_after`, plots are **observed**; otherwise this project runs a clearly-labeled **proxy** mode.\n"
    )
    lines.append("- Observed: `outputs/figures/sentiment_before_after_by_response.png`\n")
    lines.append("- Proxy: `outputs/figures/sentiment_before_after_by_response_PROXY.png`\n")

    lines.append("### 4) Which age groups show the biggest emotional shifts?\n")
    lines.append("- Observed: `outputs/figures/agegroup_sentiment_shift.png`\n")
    lines.append("- Proxy: `outputs/figures/agegroup_sentiment_shift_PROXY.png`\n")

    lines.append("### 5) How do emotions transition after the AI reply?\n")
    lines.append(
        "Transitions are shown as a Sankey diagram. "
        "Observed mode uses `Reported_emotion → Reported_emotion_after`; proxy mode uses sentiment buckets.\n"
    )
    lines.append("- Observed: `outputs/figures/emotion_transition_sankey.html`\n")
    lines.append("- Proxy: `outputs/figures/emotion_transition_sankey_PROXY.html`\n")

    lines.append("### 6) Highlighting extreme negative sentiment ethically\n")
    lines.append(
        "We highlight the **lowest 5% sentiment tail** with soft-edged markers and no individual labels, to avoid stigmatizing any interaction.\n"
    )
    lines.append("- `outputs/figures/extreme_negative_sentiment_scatter.png`\n")

    lines.append("### 7) Identifying at-risk groups (aggregated)\n")
    if risk_tbl is not None and len(risk_tbl):
        # show the top 5 groups by distress share; redact to aggregated only (already aggregated).
        top = risk_tbl.sort_values(["share_high_distress", "mean_sentiment"], ascending=[False, True]).head(5)
        lines.append(
            "Top groups by **share of high distress** (aggregated by age × response × emotion; not individuals):\n"
        )
        lines.append(top.to_markdown(index=False))
        lines.append("\n")
        lines.append(f"Full table: `{risk_path.as_posix()}`\n")
    else:
        lines.append("Risk group table not found (run step 3).\n")

    lines.append("### Appendix: full reproducible code\n")
    lines.append("- `analysis/01_data_overview.py`\n")
    lines.append("- `analysis/02_emotions_and_followup.py`\n")
    lines.append("- `analysis/03_shifts_transitions_risk.py`\n")
    lines.append("- `analysis/04_build_story.py`\n")
    lines.append("- `src/aihms/data.py`, `src/aihms/viz.py`\n")

    story_path.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")
    print(f"Wrote {story_path.relative_to(repo_root)}")


if __name__ == "__main__":
    main()

