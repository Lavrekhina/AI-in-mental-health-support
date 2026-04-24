from __future__ import annotations

from pathlib import Path

from analysis._bootstrap import ensure_src_on_path

ensure_src_on_path()


FILES = [
    "analysis/00_run_all.py",
    "analysis/01_data_overview.py",
    "analysis/02_emotions_and_followup.py",
    "analysis/03_shifts_transitions_risk.py",
    "analysis/04_build_story.py",
    "analysis/05_export_html.py",
    "src/aihms/data.py",
    "src/aihms/viz.py",
]


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    outputs = repo_root / "outputs"
    outputs.mkdir(exist_ok=True)

    out = outputs / "appendix_code.md"
    parts: list[str] = []
    parts.append("## Appendix: full reproducible code\n")
    parts.append(
        "This appendix contains the full source code used to generate the analysis, figures, and story artifacts.\n"
    )

    for rel in FILES:
        p = repo_root / rel
        if not p.exists():
            continue
        parts.append(f"\n### `{rel}`\n")
        parts.append("```python")
        parts.append(p.read_text(encoding="utf-8").rstrip())
        parts.append("```")

    out.write_text("\n".join(parts).rstrip() + "\n", encoding="utf-8")
    print(f"Wrote {out.relative_to(repo_root)}")


if __name__ == "__main__":
    main()

