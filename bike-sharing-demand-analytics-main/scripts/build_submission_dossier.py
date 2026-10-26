"""Build workspace copies of the HR & Tech Submission Dossier (.md and standalone .html with embedded charts)."""
from __future__ import annotations

import base64
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import config

ARTIFACT_MD = Path(
    r"C:\Users\Asus\.gemini\antigravity\brain\52db3254-3ae6-4998-8237-663a16e992c8\submission_dossier.md"
)
TARGET_MD = config.REPORTS_DIR / "HR_and_Tech_Submission_Dossier.md"
TARGET_HTML = config.REPORTS_DIR / "HR_and_Tech_Submission_Dossier.html"


def build() -> None:
    raw_md = ARTIFACT_MD.read_text(encoding="utf-8")
    # Replace artifact image paths with relative ../outputs/figures/ paths for the workspace .md file
    workspace_md = re.sub(
        r"!\[([^\]]*)\]\(/C:/Users/Asus/\.gemini/antigravity/brain/[^/]+/([^)]+\.png)\)",
        r"![\1](../outputs/figures/\2)",
        raw_md,
    )
    TARGET_MD.write_text(workspace_md, encoding="utf-8")

    # Build a clean standalone HTML with embedded base64 images
    html_body = workspace_md
    # Embed PNG images as base64 data URIs
    def _embed_img(match: re.Match) -> str:
        alt, fname = match.group(1), match.group(2)
        img_path = config.FIGURES_DIR / fname
        if not img_path.exists():
            return match.group(0)
        b64 = base64.b64encode(img_path.read_bytes()).decode("ascii")
        return (
            f'<figure><img src="data:image/png;base64,{b64}" alt="{alt}"/>'
            f"<figcaption>{alt}</figcaption></figure>"
        )

    html_body = re.sub(r"!\[([^\]]*)\]\(\.\./outputs/figures/([^)]+\.png)\)", _embed_img, html_body)

    # Convert basic markdown blocks to clean HTML for standalone viewing
    lines = html_body.splitlines()
    out_lines: list[str] = []
    in_table = False
    in_code = False
    in_list = False

    def _inline(text: str) -> str:
        text = re.sub(r"\[([^\]]+)\]\((file://[^)]+)\)", r"<code>\1</code>", text)
        text = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", text)
        text = re.sub(r"\*([^*]+)\*", r"<em>\1</em>", text)
        text = re.sub(r"`([^`]+)`", r"<code>\1</code>", text)
        return text

    for line in lines:
        if line.startswith("```"):
            if not in_code:
                in_code = True
                out_lines.append("<pre><code>")
            else:
                in_code = False
                out_lines.append("</code></pre>")
            continue
        if in_code:
            out_lines.append(line.replace("<", "&lt;").replace(">", "&gt;"))
            continue

        if line.startswith("|") and line.endswith("|"):
            cells = [c.strip() for c in line.strip("|").split("|")]
            if all(re.match(r"^:?-{2,}:?$", c) for c in cells):
                continue
            if not in_table:
                if in_list:
                    out_lines.append("</ul>")
                    in_list = False
                in_table = True
                out_lines.append("<table><thead><tr>" + "".join(f"<th>{_inline(c)}</th>" for c in cells) + "</tr></thead><tbody>")
            else:
                out_lines.append("<tr>" + "".join(f"<td>{_inline(c)}</td>" for c in cells) + "</tr>")
            continue
        elif in_table:
            out_lines.append("</tbody></table>")
            in_table = False

        if line.startswith("# "):
            if in_list:
                out_lines.append("</ul>"); in_list = False
            out_lines.append(f"<h1>{_inline(line[2:])}</h1>")
        elif line.startswith("## "):
            if in_list:
                out_lines.append("</ul>"); in_list = False
            out_lines.append(f"<h2>{_inline(line[3:])}</h2>")
        elif line.startswith("### "):
            if in_list:
                out_lines.append("</ul>"); in_list = False
            out_lines.append(f"<h3>{_inline(line[4:])}</h3>")
        elif line.strip().startswith("- ") or re.match(r"^\d+\.\s+", line.strip()):
            if not in_list:
                in_list = True
                out_lines.append("<ul>")
            item_txt = re.sub(r"^(?:-\s+|\d+\.\s+)", "", line.strip())
            out_lines.append(f"<li>{_inline(item_txt)}</li>")
        elif line.startswith(">"):
            out_lines.append(f"<blockquote>{_inline(line[1:].strip())}</blockquote>")
        elif line.strip() == "---":
            if in_list:
                out_lines.append("</ul>"); in_list = False
            out_lines.append("<hr/>")
        elif line.strip().startswith("<figure>"):
            if in_list:
                out_lines.append("</ul>"); in_list = False
            out_lines.append(line)
        elif line.strip():
            out_lines.append(f"<p>{_inline(line)}</p>")

    if in_table:
        out_lines.append("</tbody></table>")
    if in_list:
        out_lines.append("</ul>")

    full_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<title>Technical & Executive Submission Dossier — Capital Bikeshare Demand Analytics</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Arial, sans-serif;
         max-width: 1060px; margin: 36px auto; padding: 0 24px; color: #1e293b; line-height: 1.6; }}
  h1 {{ color: #0f172a; border-bottom: 3px solid #2563eb; padding-bottom: 10px; }}
  h2 {{ color: #1e3a8a; margin-top: 34px; border-bottom: 1px solid #cbd5e1; padding-bottom: 6px; }}
  h3 {{ color: #334155; margin-top: 24px; }}
  table {{ border-collapse: collapse; width: 100%; margin: 18px 0; font-size: 0.94rem; }}
  th, td {{ border: 1px solid #cbd5e1; padding: 9px 12px; text-align: left; vertical-align: top; }}
  th {{ background: #f1f5f9; font-weight: 600; color: #0f172a; }}
  tr:nth-child(even) td {{ background: #f8fafc; }}
  code {{ background: #f1f5f9; padding: 2px 5px; border-radius: 4px; font-size: 0.9em; }}
  pre {{ background: #0f172a; color: #f8fafc; padding: 14px; border-radius: 6px; overflow-x: auto; }}
  pre code {{ background: transparent; color: inherit; }}
  blockquote {{ border-left: 4px solid #2563eb; background: #eff6ff; margin: 14px 0; padding: 10px 16px; }}
  figure {{ margin: 22px 0; text-align: center; }}
  figure img {{ max-width: 100%; height: auto; border: 1px solid #e2e8f0; border-radius: 6px; }}
  figcaption {{ font-size: 0.9rem; color: #475569; margin-top: 6px; font-style: italic; }}
  hr {{ border: 0; border-top: 1px solid #cbd5e1; margin: 28px 0; }}
</style>
</head>
<body>
{chr(10).join(out_lines)}
</body>
</html>
"""
    TARGET_HTML.write_text(full_html, encoding="utf-8")
    print(f"Wrote {TARGET_MD} ({TARGET_MD.stat().st_size:,} bytes)")
    print(f"Wrote {TARGET_HTML} ({TARGET_HTML.stat().st_size:,} bytes)")


if __name__ == "__main__":
    build()

