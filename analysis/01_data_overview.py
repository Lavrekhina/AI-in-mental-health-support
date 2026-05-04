from __future__ import annotations

from pathlib import Path

from analysis._bootstrap import ensure_src_on_path

ensure_src_on_path()

from src.aihms.data import load_interactions


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    raw_path = repo_root / "AI_mental_health_interactions.csv"
    outputs_dir = repo_root / "outputs"
    outputs_dir.mkdir(exist_ok=True)

    result = load_interactions(raw_path)
    df = result.df

    overview_path = outputs_dir / "data_overview.txt"
    with overview_path.open("w", encoding="utf-8") as f:
        f.write("## Dataset overview\n")
        f.write(f"- rows: {len(df)}\n")
        f.write(f"- columns: {list(df.columns)}\n")
        f.write("\n")
        if result.warnings:
            f.write("## Warnings\n")
            for w in result.warnings:
                f.write(f"- {w}\n")
            f.write("\n")
        f.write("## Value counts (top)\n")
        for col in [
            "User_age_group",
            "Reported_emotion",
            "AI_response_classification",
            "User_follow_up_behaviour",
            "risk_tier",
        ]:
            f.write(f"\n### {col}\n")
            f.write(df[col].value_counts(dropna=False).to_string())
            f.write("\n")

    clean_csv_path = outputs_dir / "interactions_clean.csv"
    df.to_csv(clean_csv_path, index=False)

    print(f"Wrote {overview_path.relative_to(repo_root)}")
    print(f"Wrote {clean_csv_path.relative_to(repo_root)}")


if __name__ == "__main__":
    main()

