from __future__ import annotations

import subprocess
import sys
from pathlib import Path


STEPS = [
    "analysis.01_data_overview",
    "analysis.02_emotions_and_followup",
    "analysis.03_shifts_transitions_risk",
    "analysis.04_build_story",
    "analysis.05_export_html",
    "analysis.06_export_appendix",
]


def run_step(module: str, repo_root: Path) -> None:
    print(f"\n=== Running: python -m {module} ===")
    subprocess.run([sys.executable, "-m", module], cwd=str(repo_root), check=True)


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]

    for module in STEPS:
        run_step(module, repo_root=repo_root)

    print("\nAll steps completed.")
    print("Main artifact: outputs/story.html")


if __name__ == "__main__":
    main()

