from __future__ import annotations

from pathlib import Path

import markdown as md

from analysis._bootstrap import ensure_src_on_path

ensure_src_on_path()


CSS = """
:root{
  --bg:#fbfcfd;
  --card:#ffffff;
  --text:#111827;
  --muted:#4b5563;
  --grid:#e5e7eb;
  --calm:#2b6cb0;
  --hope:#34d399;
  --distress:#b91c1c;
  --mono: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
  --sans: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, "Apple Color Emoji", "Segoe UI Emoji";
}
html,body{background:var(--bg); color:var(--text); font-family:var(--sans); line-height:1.55;}
main{max-width: 940px; margin: 40px auto; padding: 0 18px;}
section{background:var(--card); border:1px solid var(--grid); border-radius:14px; padding: 18px 20px; margin: 14px 0;}
h2{font-size: 1.5rem; margin: 0 0 12px 0;}
h3{font-size: 1.15rem; margin: 18px 0 8px 0;}
p{margin: 10px 0;}
code{font-family:var(--mono); background:#f3f4f6; padding: 0.12rem 0.28rem; border-radius: 6px;}
pre code{display:block; padding: 12px; overflow:auto;}
blockquote{border-left: 4px solid var(--grid); padding-left: 12px; margin-left: 0; color: var(--muted);}
a{color: var(--calm); text-decoration: none;}
a:hover{text-decoration: underline;}
img{max-width:100%; border:1px solid var(--grid); border-radius: 12px; background: #fff;}
.hint{color:var(--muted); font-size:0.95rem;}
.pill{display:inline-block; padding: 2px 10px; border-radius: 999px; background: rgba(43,108,176,0.10); color: var(--calm); border: 1px solid rgba(43,108,176,0.20); font-size: 0.9rem;}
.danger{background: rgba(185,28,28,0.08); border-color: rgba(185,28,28,0.20); color: var(--distress);}
.embed{width:100%; height: 560px; border: 1px solid var(--grid); border-radius: 12px; background: #fff;}
"""


def _resolve_output_path(repo_root: Path, p: Path) -> str:
    # Keep paths relative so the report works when moving the folder.
    try:
        return str(p.relative_to(repo_root))
    except ValueError:
        return str(p)


def _inject_embeds(repo_root: Path, html: str) -> str:
    """
    If the story mentions Plotly HTML artifacts, embed them as iframes right after the paragraph/list item.
    """
    candidates = [
        repo_root / "outputs" / "figures" / "followup_by_response_interactive.html",
        repo_root / "outputs" / "figures" / "emotion_transition_sankey.html",
        repo_root / "outputs" / "figures" / "emotion_transition_sankey_PROXY.html",
    ]
    for p in candidates:
        if not p.exists():
            continue
        rel_repo = _resolve_output_path(repo_root, p)
        # story.html lives in outputs/, so use paths relative to outputs/
        rel_out = rel_repo.replace("outputs/", "", 1) if rel_repo.startswith("outputs/") else rel_repo
        marker = f"<code>{rel_repo}</code>"
        if marker in html:
            iframe = f'<div style="margin:12px 0;"><iframe class="embed" src="{rel_out}"></iframe></div>'
            html = html.replace(marker, marker + iframe)
    return html


def _inject_images(repo_root: Path, html: str) -> str:
    """
    Convert code-references to png paths into inline images.
    Example in story.md: `outputs/figures/foo.png`
    """
    import re

    pattern = re.compile(r"<code>(outputs/figures/[^<]+?\\.png)</code>")

    def repl(m: re.Match[str]) -> str:
        path_repo = m.group(1)  # e.g. outputs/figures/foo.png
        path_out = path_repo.replace("outputs/", "", 1)  # figures/foo.png
        img = f'<div style="margin:12px 0;"><img src="{path_out}" alt="{path_out}"/></div>'
        return f"<code>{path_repo}</code>{img}"

    return pattern.sub(repl, html)


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    story_md = repo_root / "outputs" / "story.md"
    if not story_md.exists():
        raise FileNotFoundError("Missing outputs/story.md. Run: python -m analysis.04_build_story")

    story_html_path = repo_root / "outputs" / "story.html"

    body = md.markdown(
        story_md.read_text(encoding="utf-8"),
        extensions=["tables", "fenced_code"],
        output_format="html5",
    )

    # Wrap each top-level section block for gentle card layout.
    # (Simple heuristic: split on <h3> while keeping the first intro in its own section.)
    body = body.replace("<h2>", "<section><h2>", 1)
    body = body.replace("</h2>", "</h2>", 1)
    body = body.replace("<h3>", "</section><section><h3>")
    body = body + "</section>"

    body = _inject_images(repo_root, body)
    body = _inject_embeds(repo_root, body)

    html = f"""<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8"/>
    <meta name="viewport" content="width=device-width, initial-scale=1"/>
    <title>AI in mental health support — data story</title>
    <style>{CSS}</style>
  </head>
  <body>
    <main>
      <div class="hint"><span class="pill">Care-focused design</span> Generated from <code>outputs/story.md</code></div>
      {body}
      <div class="hint" style="margin-top:18px;">
        Tip: keep the <code>outputs/figures/</code> folder next to this HTML so embeds render.
      </div>
    </main>
  </body>
</html>
"""

    story_html_path.write_text(html, encoding="utf-8")
    print(f"Wrote {story_html_path.relative_to(repo_root)}")


if __name__ == "__main__":
    main()

