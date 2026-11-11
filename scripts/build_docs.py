"""把带 LaTeX 公式的 Markdown 转成自包含的 HTML（MathML，不依赖 CDN/联网）。

用法：
    python -m scripts.build_docs            # 转换根目录与 docs/ 下所有含 $$ 的 md
    python -m scripts.build_docs 某文件.md

为什么用 MathML 而不是 KaTeX/MathJax：MathML 是浏览器原生支持（Chrome/Edge 109+），
HTML 里没有 <script>、不联网也能渲染，适合离线看笔记。
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from src.io import PROJECT_ROOT

STYLE = """
<style>
  :root { --ink:#1a1a1a; --muted:#5f5e5a; --line:#d8d6ce; --accent:#185FA5; }
  body { max-width: 920px; margin: 0 auto; padding: 48px 28px 96px; background:#fff; color: var(--ink);
         font-family: "Segoe UI", "Microsoft YaHei", system-ui, sans-serif; font-size: 15px; line-height: 1.8; }
  h1 { font-size: 26px; font-weight: 600; margin: 0 0 8px; letter-spacing: .2px; }
  h2 { font-size: 19px; font-weight: 600; margin: 40px 0 12px; padding-bottom: 6px; border-bottom: 1px solid var(--line); }
  h3 { font-size: 15.5px; font-weight: 600; margin: 26px 0 6px; color: #0C447C; }
  ul { padding-left: 22px; }
  li { margin: 3px 0; }
  li > strong { color: #0C447C; font-weight: 600; }
  table { display: table; border-collapse: collapse; width: 100%; margin: 14px 0 18px; font-size: 13.5px; }
  th, td { border: 1px solid var(--line); padding: 6px 10px; text-align: left; vertical-align: top; }
  th { background: #F1EFE8; font-weight: 600; }
  tbody { border: none; }
  code { background: #F4F3EE; padding: 1px 5px; border-radius: 3px; font-size: 13px;
         font-family: "Cascadia Mono", Consolas, monospace; }
  math { font-size: 1.05em; }
  math[display="block"] { display: block; width: -moz-fit-content; width: fit-content;
                          margin: 14px auto 16px; font-size: 1.12em; }
  blockquote { border-left: 3px solid var(--line); margin: 12px 0; padding: 4px 0 4px 14px; color: var(--muted); }
  hr { border: none; border-top: 1px solid var(--line); margin: 32px 0; }
  nav#TOC { background:#FAF9F5; border:1px solid var(--line); border-radius:8px;
            padding:14px 18px; margin:20px 0 28px; font-size:13.5px; }
  nav#TOC ul { list-style:none; padding-left:14px; margin:4px 0; }
  nav#TOC > ul { padding-left:0; }
  nav#TOC a { color: var(--accent); text-decoration:none; }
  nav#TOC a:hover { text-decoration: underline; }
</style>
"""


def build(md: Path) -> Path | None:
    text = md.read_text(encoding="utf-8")
    if "$$" not in text:
        return None

    header = PROJECT_ROOT / ".tmp" / "doc_header.html"
    header.parent.mkdir(parents=True, exist_ok=True)
    header.write_text(STYLE, encoding="utf-8")

    out = md.with_suffix(".html")
    subprocess.run(
        [
            "pandoc", str(md), "-o", str(out),
            "--standalone", "--math-method=mathml",
            "--toc", "--toc-depth=2",
            "-V", f"pagetitle={md.stem}",
            f"--include-in-header={header}",
        ],
        check=True,
    )
    return out


def main() -> None:
    targets = [Path(a) for a in sys.argv[1:]] or sorted(
        [*PROJECT_ROOT.glob("*.md"), *PROJECT_ROOT.glob("docs/**/*.md")])
    built = [build(p) for p in targets]
    built = [p for p in built if p]
    for p in built:
        print(f"已生成 {p.relative_to(PROJECT_ROOT)}")
    if not built:
        print("没有含 $$ 公式的 md")


if __name__ == "__main__":
    main()
